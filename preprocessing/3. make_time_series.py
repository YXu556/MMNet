"""
Scrip to make time series dataset (climate, LST, SMAP)
"""
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pandas as pd

days = 10
resolutions = ['0.1']  # ['0.1', '0.5', '1']
years = list(map(str, range(2016, 2023)))
names = ['eto', 'etr', 'pr', 'sph', 'srad', 'vpd', 'vs', 'LST', 'SMAP']
cols = []
for col in names:
    cols.extend(col+f'_{i}' for i in range(days, 0, -1))

root = Path(r"path/to/your/gee/download/raw/data")

for res in resolutions:
    dir = root / 'gee' / res
    sample_fn = root.parent / f'SCAN_USCRN_HLSL30_{res}_interpolated.csv'
    Allsamples = pd.read_csv(sample_fn)
    Allsamples = Allsamples.drop_duplicates()
    Allsamples = pd.concat([Allsamples, pd.DataFrame(columns=cols, index=Allsamples.index)], axis=1)

    # gridmet
    for year in years:
        fn = dir / f'gridmet_SSM_{year}.csv'
        tmp = pd.read_csv(fn)
        tmp.loc[tmp.Network == 'USDA_ARS', 'ID'] = tmp[tmp.Network == 'USDA_ARS'].apply(
            lambda x: 'USDA_ARS_' + x.ID, axis=1)
        try:
            gridmet = pd.concat([gridmet, tmp])
        except:
            gridmet = tmp

    gridmet['Date'] = gridmet['system:index'].apply(lambda x: int(x.split('_')[0]))
    gridmet = gridmet.drop(['system:index', 'Latitude', 'Longitude', 'Network', '.geo'], axis=1)

    # LST
    for year in years:
        fn = dir / f'LST_MOD11A1_{year}.csv'
        tmp = pd.read_csv(fn)

        try:
            LST = pd.concat([LST, tmp])
        except:
            LST = tmp

    LST['Date'] = LST['system:index'].apply(lambda x: int(''.join(x.split('_')[:3])))
    LST = LST.drop(['system:index', 'Latitude', 'Longitude', 'Network', '.geo'], axis=1)
    LST = gridmet[['ID', 'Date']].merge(LST, on=['ID', 'Date'], how='left')
    LST_interp = LST.groupby('ID')['mean'].apply(lambda x: x.interpolate(method='linear').interpolate(method='linear', limit_direction='backward'))

    # SMAP
    for year in years:
        fn = dir / f'smap_ssm_{year}.csv'
        tmp = pd.read_csv(fn)
        value_cols = [col for col in tmp.columns if year in col]
        smap_ssm = tmp[value_cols].values.reshape(-1)
        Dates = np.array([int(col.split('_')[0]) for col in tmp.columns if year in col])
        smap_df = pd.DataFrame(dict(ID=tmp.ID.values.repeat(len(value_cols)),
                                    Date=Dates[np.newaxis, :].repeat(tmp.shape[0], axis=0).flatten(),
                                    SMAP=smap_ssm))

        try:
            SMAP = pd.concat([SMAP, smap_df])
        except:
            SMAP = smap_df

    SMAP = gridmet[['ID', 'Date']].merge(SMAP, on=['ID', 'Date'], how='left')

    climate = pd.concat([gridmet,
                         pd.DataFrame(LST_interp.values, columns=['LST'], index=gridmet.index),
                         pd.DataFrame(SMAP.SMAP, columns=['SMAP'], index=gridmet.index)
                         ], axis=1)

    for name, group in tqdm(Allsamples.groupby('ID'), total=Allsamples.ID.unique().shape[0]):
        climate_tmp = climate[climate.ID == name].reset_index(drop=True)
        climate_tmp = pd.concat([climate_tmp, pd.DataFrame(columns=cols, index=climate_tmp.index)], axis=1)
        for i in range(1, days+1):
            climate_tmp.loc[i:, [col + f'_{i}' for col in names]] = climate_tmp[names][:-i].values
        Allsamples.loc[group.index, cols] = climate_tmp.set_index('Date').loc[group.Date][cols].values

    Allsamples = Allsamples.dropna(subset=cols)

    out_fn = sample_fn.parent / (sample_fn.stem + '_timeseries.csv')
    print('Saving...')
    Allsamples.to_csv(out_fn, index=None)


