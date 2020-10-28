import os
from pathlib import Path

import click
import zarr
import torch
from tqdm.auto import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.ndimage
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from brainage.model.model3d import AgeModel3DVolume
from brainage.dataset.dataset3d import BrainDataset
from brainage.dataset.grid3d import GridPatchSampler, DataReaderHDF5

load_dotenv()
DATA = Path(os.getenv('DATA'))
CONFIG = Path(os.getenv('CONFIG'))

# regional age
@click.command()
@click.argument('checkpoint', type=click.Path(exists=True))
@click.option('--overlap', default=20)
@click.option('--batch_size', default=32)
@click.option('--chunk_size', default=50)
def regional_age_map(checkpoint, overlap=25, batch_size=32, chunk_size=50):
    patch_overlap = [overlap, overlap, overlap]
    out_dir = Path(DATA/'nako/processed/patchwise')
    out_dir.mkdir(exist_ok=True)

    print('loading model')
    print(f'checkpoint: {checkpoint}')
    model = AgeModel3DVolume.load_from_checkpoint(checkpoint, train_ds=None, val_ds=None)
    model.eval()
    device = torch.device('cuda')
    model = model.to(device)

    # setup
    data_path = model.hparams['dataset']['data']
    data_group = model.hparams['dataset']['group']
    info = model.hparams['dataset']['info']
    infocolumn = model.hparams['dataset']['column']
    val_set = model.hparams['dataset']['val']
    patch_size = np.array(model.hparams['dataset']['patch_size'])
    val_keys = [l.strip() for l in Path(val_set).open().readlines()]

    print(f'total number of keys {len(val_keys)}')
    chunk_num = len(val_keys)//chunk_size
    chunks = np.array_split(np.array(val_keys), chunk_num)

    for c, chunk in enumerate(chunks):
        print(f'chunk {c}/{chunk_num}')
        chunk_keys = list(chunk)

        print('loading data')
        print(f'ds: {data_path} - {data_group}')
        print(f'set: {val_set}')
        info_df = pd.read_csv(info, index_col=0, dtype={'key': 'string', infocolumn: np.float32})
        ds = GridPatchSampler(data_path,
                        chunk_keys[:],
                        patch_size, patch_overlap,
                        out_channels=2,
                        out_dtype=np.float32,
                        image_group=data_group,
                        ReaderClass=DataReaderHDF5,
                        pad_args={'mode': 'constant'})
        loader = DataLoader(ds, batch_size=batch_size, num_workers=1, drop_last=False, shuffle=False)

        print('processing subjects')
        pred = {'key': [], 'pos0': [], 'pos1': [], 'pos2': [], 'y': [], 'yhat0': [], 'yhat1': []}
        for sample in loader:
            print('.', end='')
            # predict
            x = sample['data'].float()
            x = x.to(device)
            position = sample['position'].float()
            y_hat = model(x, pos=None)

            # store map with predicted values
            shape = np.array(x.size())
            sample['data'] = np.einsum('ij,klm->ijklm', 
                                    y_hat.detach().cpu().numpy(),
                                    np.ones(shape[2:]))
            ds.add_processed_batch(sample)

            # store results
            for b in range(len(sample['key'])):
                key = sample['key'][b]
                y = info_df.loc[key][infocolumn]
                pred['key'].append(key)
                pred['pos0'].append(position[b,0].cpu().item())
                pred['pos1'].append(position[b,1].cpu().item())
                pred['pos2'].append(position[b,2].cpu().item())
                pred['y'].append(y)
                pred['yhat0'].append(y_hat[b,0].cpu().item())
                pred['yhat1'].append(y_hat[b,1].cpu().item())

        print('storing results')
        df = pd.DataFrame.from_dict(pred)
        if (out_dir/f'predictions_regional.feather').is_file():
            df_0 = pd.read_feather(out_dir/f'predictions_regional.feather').set_index('key')
            df = df_0.combine_first(df.set_index('key'))
            df = df.reset_index()
        df.to_feather(out_dir/f'predictions_regional.feather')

        results = ds.get_assembled_data()
        with zarr.open(str(out_dir/'maps.zarr')) as root:
            maps = root.require_group('agemaps')
            zarr.copy_all(results, maps, if_exists='replace')


# grad cam grid
@click.command()
@click.argument('checkpoint', type=click.Path(exists=True))
@click.option('--overlap', default=5)
@click.option('--patch_size', default=64)
def gradcam_grid(checkpoint, overlap=25, patch_size=64):
    patch_overlap = [overlap, overlap, overlap]
    out_dir = Path(DATA/'nako/processed/patchwise')
    out_dir.mkdir(exist_ok=True)
    batch_size = 1 

    print('loading model')
    print(f'checkpoint: {checkpoint}')
    model = AgeModel3DVolume.load_from_checkpoint(checkpoint, train_ds=None, val_ds=None)
    model.eval()
    device = torch.device('cuda')
    model = model.to(device)

    # setup
    data_path = model.hparams['dataset']['data']
    data_group = model.hparams['dataset']['group']
    info = model.hparams['dataset']['info']
    infocolumn = model.hparams['dataset']['column']
    val_set = model.hparams['dataset']['val']
    patch_size = [patch_size, patch_size, patch_size]
    val_keys = [l.strip() for l in Path(val_set).open().readlines()]

    print('loading data')
    print(f'ds: {data_path} - {data_group}')
    print(f'set: {val_set}')
    info_df = pd.read_csv(info, index_col=0, dtype={'key': 'string', infocolumn: np.float32})
    ds = GridPatchSampler(data_path,
                          val_keys,
                          patch_size,
                          patch_overlap,
                          out_channels=1,
                          out_dtype=np.float32,
                          image_group=data_group,
                          ReaderClass=DataReaderHDF5,
                          pad_args={'mode': 'constant'})
    loader = DataLoader(ds, batch_size=batch_size, num_workers=1, drop_last=False, shuffle=False)
    results = {'position': {}, 'patch': {}, 'heatmap': {}}
    for key in val_keys:
        results['position'][key] = []
        results['patch'][key] = []
        results['heatmap'][key] = []

    print('processing subjects')
    pred = {'key': [], 'pos0': [], 'pos1': [], 'pos2': [], 'y': [], 'yhat0': [], 'yhat1': []}
    for sample in loader:
        print('.', end='')
        # predict
        x = sample['data'].float()
        x = x.to(device)
        position = sample['position'].float()
        position = position.to(device)
        y_hat, heatmap = model.gradcam(x, pos=None)
        shape = np.array(x.size())

        # store raw heatmap
        key = sample['key'][0]
        results['position'][key].append(position.cpu().numpy())
        results['patch'][key].append(sample['data'][0, 0].cpu().numpy())
        results['heatmap'][key].append(heatmap.detach().cpu().numpy())

        # store zoomed heatmap
        heatmap = heatmap.detach().cpu().numpy()
        zoom = np.array(shape[2:])/np.array(heatmap.shape)
        heatmap_zoomed = scipy.ndimage.zoom(heatmap, zoom, order=0)
        sample['data'] = heatmap_zoomed[np.newaxis, np.newaxis,...]
        ds.add_processed_batch(sample)

        # store results
        for b in range(batch_size):
            key = sample['key'][b]
            y = info_df.loc[key][infocolumn]
            pred['key'].append(key)
            pred['pos0'].append(position[b,0].cpu().item())
            pred['pos1'].append(position[b,1].cpu().item())
            pred['pos2'].append(position[b,2].cpu().item())
            pred['y'].append(y)
            pred['yhat0'].append(y_hat[b,0].cpu().item())
            pred['yhat1'].append(y_hat[b,1].cpu().item())

    for key in val_keys:
        results['position'][key] = np.stack(results['position'][key], axis=0)
        results['patch'][key] = np.stack(results['patch'][key], axis=0)
        results['heatmap'][key] = np.stack(results['heatmap'][key], axis=0)

    print('storing results')
    df = pd.DataFrame.from_dict(pred)
    if (out_dir/f'predictions_grid.feather').is_file():
        df_0 = pd.read_feather(out_dir/f'predictions_grid.feather').set_index('key')
        df = df_0.combine_first(df.set_index('key'))
        df = df.reset_index()
    df.to_feather(out_dir/f'predictions_grid.feather')

    results = ds.get_assembled_data()
    store = zarr.DirectoryStore(out_dir/'maps.zarr')
    root = zarr.group(store=store)
    maps = root.require_group('gradmaps')
    zarr.copy_all(results, maps, if_exists='replace')


# patch positions
@click.command()
@click.argument('checkpoint', type=click.Path(exists=True))
def gradcam_position(checkpoint):
    out_dir = Path(DATA/'nako/processed/patchwise')
    out_dir.mkdir(exist_ok=True)
    batch_size = 1 

    print('loading model')
    print(f'checkpoint: {checkpoint}')
    model = AgeModel3DVolume.load_from_checkpoint(checkpoint, train_ds=None, val_ds=None)
    model.eval()
    device = torch.device('cuda')
    model = model.to(device)

    # setup
    patch_positions = np.load(CONFIG/'patchwise/patch_positions.npy')
    data_path = model.hparams['dataset']['data']
    data_group = model.hparams['dataset']['group']
    info = model.hparams['dataset']['info']
    infocolumn = model.hparams['dataset']['column']
    patch_size = np.array(model.hparams['dataset']['patch_size'])
    val_set = model.hparams['dataset']['val']
    val_keys = [l.strip() for l in Path(val_set).open().readlines()]
    val_transform = None

    print('loading data')
    print(f'ds: {data_path} - {data_group}')
    print(f'set: {val_set}')
    ds_test = BrainDataset(data=data_path,
                            keys=val_keys,
                            info=info,
                            group=data_group,
                            column=infocolumn,
                            preload=True,
                            transform=val_transform)
    loader = DataLoader(ds_test, batch_size=1, num_workers=1, drop_last=False, shuffle=False)

    zarr_path = out_dir/'patches.zarr'
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)
    heatmaps_mean = root.require_group('heatmaps_mean')
    heatmaps_sigma = root.require_group('heatmaps_sigma')
    images = root.require_group('images')

    print('processing data')
    # compute mean activation heatmaps
    results = {'key': [], 'p_idx':[], 'px':[], 'py':[], 'pz':[],
               'y': [], 'y_hat0': [], 'y_hat1': []}

    for sample in tqdm(loader):
        volume = sample['data'].float()
        img = []
        hmap = []

        for p_idx, position in enumerate(patch_positions):
            start_index = position - patch_size//2
            stop_index = start_index + patch_size
            x = volume[:, :, start_index[0]:stop_index[0],
                             start_index[1]:stop_index[1],
                             start_index[2]:stop_index[2]]
            x = x.to(device)
            y = sample['label'][0].float()
            key = sample['key'][0]
            y_hat, heatmap = model.gradcam(x, channel=0)

            # collect heatmap/image
            hmap.append(heatmap.cpu().numpy().astype(np.float32))
            img.append(x.cpu().numpy().astype(np.float16)[0,0])   
  
            # store prediction
            results['key'].append(key)
            results['y'].append(y.item())
            results['p_idx'].append(p_idx)
            results['px'].append(position[0])
            results['py'].append(position[1])
            results['pz'].append(position[2])
            results['y_hat0'].append(y_hat[0, 0].cpu().item())
            results['y_hat1'].append(y_hat[0, 1].cpu().item())

        # store heatmap/image to zarr
        img = np.stack(img, axis=0)
        hmap = np.stack(hmap, axis=0)
        ds = heatmaps_mean.zeros(key ,shape=hmap.shape, chunks=False, dtype=hmap.dtype, overwrite=True)
        ds[:] = hmap 
        ds = images.zeros(key ,shape=img.shape, chunks=False, dtype=img.dtype, overwrite=True)
        ds[:] = img 

    df = pd.DataFrame.from_dict(results)
    if (out_dir/f'predictions_pos.feather').is_file():
        df_0 = pd.read_feather(out_dir/f'predictions_pos.feather').set_index('key')
        df = df_0.combine_first(df.set_index('key'))
        df = df.reset_index()
    df.to_feather(out_dir/f'predictions_pos.feather')

    # compute sigma activation heatmaps
    for sample in tqdm(loader):
        volume = sample['data'].float()
        hmap = []
        for p_idx, position in enumerate(patch_positions):
            start_index = position - patch_size//2
            stop_index = start_index + patch_size
            x = volume[:, :, start_index[0]:stop_index[0],
                             start_index[1]:stop_index[1],
                             start_index[2]:stop_index[2]]
            x = x.to(device)
            y = sample['label'][0].float()
            key = sample['key'][0]
            y_hat, heatmap = model.gradcam(x, channel=1)
            # collect heatmap/image
            hmap.append(heatmap.cpu().numpy().astype(np.float32))

    # store heatmap/image to zarr
    hmap = np.stack(hmap, axis=0)
    ds = heatmaps_sigma.zeros(key ,shape=hmap.shape, chunks=False, dtype=hmap.dtype, overwrite=True)
    ds[:] = hmap 


if __name__ == '__main__':
    regional_age_map()