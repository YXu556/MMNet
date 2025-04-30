import torch
import torch.nn as nn
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models import MMNet
from utils import *

from pathlib import Path
from sklearn.model_selection import KFold


out_dir = Path(r'dir/to/save/the/plot')
out_dir.mkdir(exist_ok=True, parents=True)

n_samples, d1, t, d2 = 1000, 28, 11, 8
filename = Path(r'path/to/your/SCAN_USCRN_HLSL30_0.1_ts_interpolated_timeseries.csv')

data = pd.read_csv(filename)
data = data.drop_duplicates(subset=['ID', 'Date'])
stations = np.unique(data.ID.values)
station_domain = data.set_index('ID')['Domain'].to_dict()

static_col = [
    "LC_code", 'DoY_normalized',
    "Latitude", "Longitude",
    "elevation", "slope", "aspect_normalized",
    "clay", "bd", "sand",
    'SMAP',
    'VV', 'VH', 'angle',
    'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2',
]
dynamic_col = [
    'eto', 'etr', 'pr', 'sph', 'srad', 'vpd', 'vs', 'LST',
]

d_dynamic_col = []
for col in dynamic_col:
    d_dynamic_col.extend([col + f'_{i}' for i in range(10, 0, -1)] + [col])

static_col += ['ndvi', 'evi', 'ndwi', 'lswi', 'nsdsi',
               'cr', 'dpsvim', 'pol', 'rvim', 'vvvh']
static_col_new = list(
    set(static_col) - {'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2', 'VV', 'VH',
                       'angle', })
static_col_new.sort(key=static_col.index)
static_col = static_col_new

for i, test_domain in enumerate(['W', 'M', 'E']):
    print(f'==========[{i + 1}/3] Test on {test_domain} ==========')
    best_model_path = Path(f'../checkpoints/SCAN_USCRN_HLSL30_0.1_MMNet_exp3/{test_domain}/model_best.pth')
    fold_dir = out_dir / test_domain
    fold_dir.mkdir(exist_ok=True, parents=True)

    test_stations = [k for k, v in station_domain.items() if v == test_domain]
    test_idx = np.where(np.isin(stations, test_stations))[0]
    train_idx = np.where(~np.isin(stations, test_stations))[0]

    source_data = data[data['ID'].isin(stations[train_idx])].reset_index(drop=True)
    target_data = data[data['ID'].isin(stations[test_idx])].reset_index(drop=True)

    source_data_statics = lc_encode(source_data[static_col].values, lc_code='oh')
    target_data_statics = lc_encode(target_data[static_col].values, lc_code='oh')

    mean = source_data_statics.mean(0)
    std = source_data_statics.std(0)
    std[std == 0] = 1

    source_data_dynamic = source_data[d_dynamic_col].values.reshape(source_data.shape[0], -1, 10 + 1)
    target_data_dynamic = target_data[d_dynamic_col].values.reshape(target_data.shape[0], -1, 10 + 1)

    mean_d = source_data_dynamic.mean(0).mean(-1).reshape(-1, 1)
    std_d = source_data_dynamic.transpose(0, 2, 1).reshape(-1, source_data_dynamic.shape[1]).std(0).reshape(-1, 1)

    source_y = source_data["VWC_5"]
    target_y = target_data["VWC_5"]

    X_static = target_data_statics
    X_time = target_data_dynamic
    y = target_y

    X_static = (X_static - mean) / std
    X_time = (X_time - mean_d) / std_d

    X_static_torch = torch.tensor(X_static, dtype=torch.float)
    X_time_torch = torch.tensor(X_time, dtype=torch.float).transpose(2, 1)
    y_torch = torch.tensor(y, dtype=torch.float)

    # model
    model = MMNet(d1, d2,)
    checkpoint = torch.load(best_model_path)
    state_dict = checkpoint['model_state']
    model.load_state_dict(state_dict)

    indices = random.sample(range(len(X_static_torch)), n_samples)
    X_static_torch = X_static_torch[indices]
    X_time_torch = X_time_torch[indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    X_static_torch = X_static_torch.to(device)
    X_time_torch = X_time_torch.to(device)

    explainer = shap.GradientExplainer(model, [X_static_torch, X_time_torch])
    # shap_values_static, shap_values_dynamic = explainer.shap_values([X_static_torch, X_time_torch])

    batch_size = 100
    shap_values_static = []
    shap_values_dynamic = []
    for i in tqdm(range(0, n_samples, batch_size)):
        batch_static = X_static_torch[i:i+batch_size]
        batch_time = X_time_torch[i:i+batch_size]
        shap_batch = explainer.shap_values([batch_static, batch_time])
        shap_values_static.append(shap_batch[0])
        shap_values_dynamic.append(shap_batch[1])

    shap_values_static = np.vstack(shap_values_static)
    shap_values_dynamic = np.vstack(shap_values_dynamic)

    shap_values_static_agg_lc = np.concatenate(
        (shap_values_static[:, :8].sum(1).reshape(-1, 1), shap_values_static[:, 8:]), 1)

    X_static_np = torch.concat([torch.argmax(X_static_torch[:, :8], dim=1, keepdim=True), X_static_torch[:, 8:]],
                               dim=1).cpu().numpy()
    X_dynamic_np = X_time_torch.cpu().numpy()
    np.save(fold_dir / 'static_feature.npy', X_static_np)
    np.save(fold_dir / 'dynamic_feature.npy', X_dynamic_np)
    np.save(fold_dir / 'static.npy', shap_values_static_agg_lc, )
    np.save(fold_dir / 'dynamic.npy', shap_values_dynamic, )