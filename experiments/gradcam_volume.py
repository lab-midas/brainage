import os
from pathlib import Path

import click
import torch
import zarr
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from batchgenerators.transforms.crop_and_pad_transforms import CenterCropTransform

from brainage.model.model3d import AgeModel3DVolume
from brainage.dataset.dataset3d import BrainDataset

load_dotenv()
DATA = Path(os.getenv('DATA'))

@click.command()
@click.argument('checkpoint', type=click.Path(exists=True))
def gradcam_volume(checkpoint):
    out_dir = Path(DATA/'nako/processed/volume')
    out_dir.mkdir(exist_ok=True)

    print('loading model')
    device = torch.device('cuda')
    model = AgeModel3DVolume.load_from_checkpoint(checkpoint, train_ds=None, val_ds=None)
    model.eval()
    model.to(device)

    # setup
    data_path = model.hparams['dataset']['data']
    data_group = model.hparams['dataset']['group']
    info = model.hparams['dataset']['info']
    infocolumn = model.hparams['dataset']['column']
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

    zarr_path = out_dir/'maps.zarr'
    store = zarr.DirectoryStore(zarr_path)
    root = zarr.group(store=store)
    heatmaps_mean = root.require_group('heatmaps_mean')
    heatmaps_sigma = root.require_group('heatmaps_sigma')
    images = root.require_group('images')

    print('processing data')
    # compute mean activation heatmaps
    results = {'key': [], 'y': [], 'y_hat0': [], 'y_hat1': []}
    for sample in tqdm(loader):
        x = sample['data'].float()
        x = x.to(device)
        y = sample['label'][0].float()
        key = sample['key'][0]
        y_hat, heatmap = model.gradcam(x, channel=0)

        # store heatmap/image to zarr
        hmap = heatmap.cpu().numpy().astype(np.float32)
        ds = heatmaps_mean.zeros(key ,shape=hmap.shape, chunks=False, dtype=hmap.dtype, overwrite=True)
        ds[:] = hmap 
        img = x.cpu().numpy().astype(np.float16)[0,0]
        ds = images.zeros(key ,shape=img.shape, chunks=False, dtype=img.dtype, overwrite=True)
        ds[:] = img 
        
        # store prediction
        results['key'].append(key)
        results['y'].append(y.item())
        results['y_hat0'].append(y_hat[0, 0].detach().cpu().item())
        results['y_hat1'].append(y_hat[0, 1].detach().cpu().item())
    
    # save predictions
    df = pd.DataFrame.from_dict(results)
    if (out_dir/f'predictions.feather').is_file():
        df_0 = pd.read_feather(out_dir/f'predictions.feather').set_index('key')
        df = df_0.combine_first(df.set_index('key'))
        df = df.reset_index()
    df.to_feather(out_dir/f'predictions.feather')

    # compute sigma activation heatmaps
    for sample in tqdm(loader):
        x = sample['data'].float()
        x = x.to(device)
        y = sample['label'][0].float()
        key = sample['key'][0]
        y_hat, heatmap = model.gradcam(x, channel=1)

        # store heatmap/image to zarr
        hmap = heatmap.cpu().numpy().astype(np.float32)
        ds = heatmaps_sigma.zeros(key, shape=hmap.shape, chunks=False, dtype=hmap.dtype, doverwrite=True)
        ds[:] = hmap 

if __name__ == '__main__':
    gradcam_volume()

