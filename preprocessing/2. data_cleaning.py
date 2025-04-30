"""
Script for data cleaning to `fn.stem + '_interpolated.csv'`
1. interpolate S1, HLS, LST (linear)
2. calculate indices
3. clean columns, keep used ones only
"""
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
from tqdm import tqdm

fn = Path(rf"path/to/csv/generated/from/1.merge_multi_csv")
s1_bands = ['VV', 'VH', 'angle']
l8_bands = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2']

data = pd.read_csv(fn)
data['days'] = (pd.to_datetime(data['Date'].apply(str)) - pd.to_datetime('2016-1-1')).apply(
    lambda x: x.days)  # days from 2016-1-1
data = data.groupby('ID').apply(lambda x: x.sort_values('days'))
data = data.reset_index(drop=True)
data = data.set_index('ID')

for i, (name, nt_group) in enumerate(tqdm(data.groupby('ID'))):
    if name == 'USCRN_St._Mary_1_SSW':
        print('lll')
    days = nt_group['days'].values

    # Sentinel-1
    for s1_b in s1_bands:
        valid_sn = ~np.isnan(nt_group[s1_b].values)
        if valid_sn.sum() / days.shape[0] <= 0.05:
            continue
        tmp = nt_group[s1_b].values
        tmp_interp = np.interp(days, days[valid_sn], tmp[valid_sn])
        data.loc[name, s1_b] = tmp_interp

    # Landsat-8
    nt_group[(nt_group[l8_bands[:-2]] > 1).any(axis=1) * (nt_group[l8_bands[:-2]] < 0).any(axis=1)] = np.nan
    for l8_b in l8_bands:
        valid_ls = ~np.isnan(nt_group[l8_b].values)
        if valid_ls.sum() / days.shape[0] <= 0.03:
            continue
        tmp = nt_group[l8_b].values
        tmp_interp = np.interp(days, days[valid_ls], tmp[valid_ls])
        data.loc[name, l8_b] = tmp_interp

    # LST
    valid_ls = ~np.isnan(nt_group['LST'].values)
    if valid_ls.sum() / days.shape[0] <= 0.03:
        continue
    tmp = nt_group['LST'].values
    tmp_interp = np.interp(days, days[valid_ls], tmp[valid_ls])
    data.loc[name, 'LST'] = tmp_interp

# indices
# Landsat
print('Calculate indices')
data['ndvi'] = (data.NIR - data.Red) / (data.NIR + data.Red)
data['evi'] = 2.5 * (data.NIR - data.Red) / (data.NIR + 6 * data.Red - 7.5 * data.Blue + 1)
data['ndwi'] = (data.Green - data.NIR) / (data.Green + data.NIR)
data['lswi'] = (data.NIR - data.SWIR1) / (data.NIR + data.SWIR1)
data['nsdsi'] = (data.SWIR1 - data.SWIR2) / data.SWIR1

# radar
data['cr'] = data.VV / data.VH
data['dpsvim'] = (data.VV ** 2 + data.VH ** 2) / 2 ** 5
data['pol'] = (data.VV + data.VH) / (data.VV - data.VH)
data['rvim'] = 4 * data.VH / (data.VV + data.VH)
data['vvvh'] = data.VV - data.VH

data = data.rename(columns={'sand_5': 'sand', 'clay_5': 'clay', 'bd_5': 'bd'})
data = data.reset_index()
data = data[[
    'ID', 'Latitude', 'Longitude', 'Network', 'Date', 'Year',
    'SMAP', 'VWC_5',
    'DoY', 'DoY_normalized',
    'elevation', 'slope', 'aspect', 'aspect_normalized',
    'clay', 'sand', 'bd',
    'LC_name',
    'eto', 'etr', 'pr', 'sph', 'srad', 'vpd', 'vs', 'LST',
    'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2',
    'ndvi', 'evi', 'ndwi', 'lswi', 'nsdsi',
    'VV', 'VH', 'angle',
    'cr', 'dpsvim', 'pol', 'rvim', 'vvvh'
]]

data = data.dropna()
network = gpd.read_file(r"path/to/your/network_shp")
data['Domain'] = data.merge(network, on='ID', how='left')['Domain']
data = data.drop_duplicates(subset=['ID', 'Date'])
print('Total number:', data.shape[0])

out_fn = fn.parent / (fn.stem + '_interpolated.csv')
print('Saving...')
data.to_csv(out_fn, index=None)
