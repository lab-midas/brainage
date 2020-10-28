import os   
from pathlib import Path

import zarr
import dotenv
import scipy.ndimage
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from tqdm.auto import tqdm
dotenv.load_dotenv()


def average(store, key_sel, save_key=None):
    image_average = None
    hmap_average_mean = None
    N = len(key_sel)
    
    for key in tqdm(key_sel):
        with zarr.open(store=store, mode='r') as zf:
            hmap_mean = zf['heatmaps_mean'][key][:].astype(np.float32)
            img = zf['images'][key][:].astype(np.float32)

        if hmap_average_mean is None:
            hmap_average_mean = hmap_mean
        else:
            hmap_average_mean = hmap_mean/hmap_mean.max() + hmap_average_mean
    
        if image_average is None:
            image_average = img
        else:
            image_average = image_average + img
    image_average = image_average/N
    hmap_average_mean = hmap_average_mean/N

    if save_key:
        with zarr.open(store=store, mode='a') as zf:
            gr = zf.require_group('average')
            gr_img  = gr.require_group('image')
            gr_hmap_mean = gr.require_group('heatmap_mean')
            ds = gr_img.zeros(save_key, shape=image_average.shape, 
                            dtype=image_average.dtype, chunks=False,
                            overwrite=True)
            ds[:] = image_average
            ds = gr_hmap_mean.zeros(save_key, shape=hmap_average_mean.shape, 
                                    dtype=hmap_average_mean.dtype, chunks=False,
                                    overwrite=True)
            ds[:] = hmap_average_mean

    return image_average, hmap_average_mean

def zoom_heatmap(hmap, shape, order=3):
    zoom = np.array(shape)/np.array(hmap.shape)
    return scipy.ndimage.zoom(hmap, zoom, order=order)


if __name__ == '__main__':
    DATA = Path(os.getenv('DATA'))
    CONFIG = Path(os.getenv('CONFIG'))
    out_dir = DATA/'nako/processed/volume'
    cfg = OmegaConf.load(str(CONFIG/'volume/config.yaml'))
    store = zarr.DirectoryStore(str(out_dir/'maps.zarr'))
    info = pd.read_csv(cfg.dataset.info).astype({'key': str, 'age': np.float64})
    volume_predictions = pd.read_feather(DATA/'nako/processed/volume/predictions.feather').astype({'key': str})
    df = info.join(volume_predictions.set_index('key'), on='key', how='inner')

    image_average, hmap_average_mean = average(store, df['key'], save_key = 'aa')
    select = ((df['sex'] == 1))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'am')
    select = ((df['sex'] == 2))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'af')
    select = ((df['age'] >= 60) & (df['sex'] == 1))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'om')
    select = ((df['age'] <= 40) & (df['sex'] == 1))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'ym')
    select = ((df['age'] > 40) & (df['age'] < 60) & (df['sex'] == 1))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'mm')
    select = ((df['age'] >= 60) & (df['sex'] == 2))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'of')
    select = ((df['age'] <= 40) & (df['sex'] == 2))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'yf')
    select = ((df['age'] > 40) & (df['age'] < 60) & (df['sex'] == 2))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'mf')

    DATA = Path(os.getenv('DATA'))
    CONFIG = Path(os.getenv('CONFIG'))
    out_dir = DATA/'nako/processed/patchwise'
    cfg = OmegaConf.load(str(CONFIG/'volume/config.yaml'))
    store = zarr.DirectoryStore(str(out_dir/'patches.zarr'))
    info = pd.read_csv(cfg.dataset.info).astype({'key': str, 'age': np.float64})
    volume_predictions = pd.read_feather(DATA/'nako/processed/volume/predictions.feather').astype({'key': str})
    df = info.join(volume_predictions.set_index('key'), on='key', how='inner')
    position_predictions = pd.read_feather(DATA/'nako/processed/patchwise/predictions_pos.feather').astype({'key': str})
    df_pos = info.join(volume_predictions.set_index('key'), on='key', how='inner')

    image_average, hmap_average_mean = average(store, df['key'], save_key = 'aa')
    select = ((df['sex'] == 1))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'am')
    select = ((df['sex'] == 2))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'af')
    select = ((df['age'] >= 60) & (df['sex'] == 1))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'om')
    select = ((df['age'] <= 40) & (df['sex'] == 1))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'ym')
    select = ((df['age'] > 40) & (df['age'] < 60) & (df['sex'] == 1))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'mm')
    select = ((df['age'] >= 60) & (df['sex'] == 2))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'of')
    select = ((df['age'] <= 40) & (df['sex'] == 2))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'yf')
    select = ((df['age'] > 40) & (df['age'] < 60) & (df['sex'] == 2))
    image_average, hmap_average_mean = average(store, df[select]['key'], save_key = 'mf')