"""
Script to merge all types of data downloaded from GEE, outputing f'SCAN_USCRN_HLSL30_{res}.csv'
1. combine all data types and all years
2. add DoY/aspect normalized
3. convert landcover type
4. remove stations with >80% nan gt values
5. fill missing smap using forward and backward fill
"""
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn.functional as F

resolutions = ['0.1']
years = list(map(str, range(2016, 2023)))
root = Path(r"path/to/your/gee/download/data")
gt_fn = Path(r"path/to/your/ground/truch")

gt = pd.read_csv(gt_fn)
gt['Year'] = gt['Date'].apply(lambda x: str(x)[:4])
gt = gt[gt.Network.isin(['SCAN', 'USCRN'])]

for res in resolutions:
    print(f'===================Process Res-{res}==================')
    dir = root / 'gee' / res
    for year in years:
        gt_year = gt[gt['Year'] == year]
        fns = [f.name for f in dir.glob(f'*{year}*')] + ['constant_SSM.csv']
        if year in ['2017']:
            fns.append('LC_SSM_2016.csv')
        elif year in ['2018']:
            fns.append('LC_SSM_2019.csv')
        elif year in ['2020', '2022']:
            fns.append('LC_SSM_2021.csv')

        for fn in fns:
            data = pd.read_csv(dir / fn)
            data.loc[data.Network == 'USDA_ARS', 'ID'] = data[data.Network == 'USDA_ARS'].apply(
                lambda x: 'USDA_ARS_' + x.ID, axis=1)
            data = data.dropna().reset_index(drop=True)

            if 'gridmet_SSM' in fn:
                data['Date'] = data['system:index'].apply(lambda x: int(x.split('_')[0]))
                data = data.drop(['system:index', 'Latitude', 'Longitude', 'Network', '.geo'], axis=1)
                gt_year = gt_year.merge(data, on=['ID', 'Date'], how='left')
            elif 'LC_SSM' in fn:
                data = data.drop(['system:index', 'Latitude', 'Longitude', 'Network', '.geo'], axis=1)
                gt_year = gt_year.merge(data, on='ID', how='left').rename(columns={'mode': 'LC'})
            elif 'LST_MOD11A1' in fn:
                data['Date'] = data['system:index'].apply(lambda x: int(''.join(x.split('_')[:3])))
                data = data.drop(['system:index', 'Latitude', 'Longitude', 'Network', '.geo'], axis=1)
                gt_year = gt_year.merge(data, on=['ID', 'Date'], how='left').rename(columns={'mean': 'LST'})
            elif 'S1_SSM' in fn:
                data['Date'] = data['system:index'].apply(lambda x: int(x.split('_')[4][:8]))
                data = data.drop(['system:index', 'Latitude', 'Longitude', 'Network', '.geo'], axis=1)
                gt_year = gt_year.merge(data, on=['ID', 'Date'], how='left')
            elif 'HLSL30_SSM' in fn:
                bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']
                bandnames = ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2']
                data['Date'] = data['system:index'].apply(lambda x: int(x.split('_')[1][:8]))
                data = data.drop(['system:index', 'Latitude', 'Longitude', 'Network', '.geo', 'B1', 'B9'], axis=1)
                data = data.rename(columns=dict(zip(bands, bandnames)))
                if '_1' in fn:
                    gt_year = gt_year.merge(data, on=['ID', 'Date'], how='left')
                else:
                    gt_year = gt_year.merge(data, on=['ID', 'Date'] + bandnames, how='left')
            elif 'smap_ssm' in fn:
                value_cols = [col for col in data.columns if year in col]
                smap_ssm = data[value_cols].values.reshape(-1)
                Dates = np.array([int(col.split('_')[0]) for col in data.columns if year in col])
                smap_df = pd.DataFrame(dict(ID=data.ID.values.repeat(len(value_cols)),
                                            Date=Dates[np.newaxis, :].repeat(data.shape[0], axis=0).flatten(),
                                            SMAP=smap_ssm))
                gt_year = gt_year.merge(smap_df, on=['ID', 'Date'], how='left')
            elif 'constant_SSM' in fn:
                data = data.drop(['system:index', 'Latitude', 'Longitude', 'Network', 'hillshade', '.geo'], axis=1)
                gt_year = gt_year.merge(data, on='ID', how='left')

        try:
            dataset = pd.concat([dataset, gt_year])
        except:
            print('The first year ...')
            dataset = gt_year

    dataset = dataset.rename(columns={'sand_5': 'sand', 'clay_5': 'clay', 'bd_5': 'bd'})
    # DoY
    dataset['DoY'] = pd.to_datetime(dataset['Date'], format='%Y%m%d').apply(lambda x: x.day_of_year)
    dataset['DoY_normalized'] = np.cos((dataset['DoY'] - 1) / 365 * np.pi)

    # Landcover
    lc_code_map = {21: 'Developed', 22: 'Developed', 23: 'Developed', 24: 'Developed',
                   31: 'Barren',
                   41: 'Forest', 42: 'Forest', 43: 'Forest',
                   51: 'Shrub', 52: 'Shrub',
                   71: 'Grassland', 72: 'Grassland',
                   81: 'Pasture',
                   82: 'Crop',
                   90: 'Wetland', 95: 'Wetland'}
    dataset = dataset[dataset['LC'].isin(lc_code_map.keys())].reset_index(drop=True)
    dataset['LC_name'] = dataset.LC.map(lc_code_map)

    # aspect
    dataset['aspect_normalized'] = np.cos(dataset['aspect'] / 360 * np.pi)

    # fill missing data
    # first remove stations with too many nan values
    print(len(dataset['ID'].unique()))  # 280
    dataset = dataset.loc[
        dataset.groupby('ID')['VWC_5'].filter(lambda x: len(x[pd.isnull(x)]) / len(x) < 0.2).index]
    print(len(dataset['ID'].unique()))  # 0.5 - 263 | 0.8 - 278 | 0.2 - 181

    dataset = dataset.dropna(subset=['VWC_5'])

    # forward and backward fill soil moisture 5 cm
    dataset['SMAP'] = dataset.groupby('ID')['SMAP'].apply(lambda x: x.ffill().bfill())

    print('Total number:', dataset.shape[0])
    dataset.to_csv(root.parent / f'SCAN_USCRN_HLSL30_{res}.csv')
