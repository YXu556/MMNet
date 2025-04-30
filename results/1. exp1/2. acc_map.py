import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import *


CRS = 'EPSG:32615'

CONUS = gpd.read_file(r'path/to/CONUS.shp').to_crs(CRS)
network = gpd.read_file(r"path/to/yout/networks.shp").to_crs(CRS)
network = network[network.Network.isin(['SCAN', 'USCRN'])]

models = ['mmnet', 'transformer', 'mlp']
root = Path(r"root/path/to/your/checkpoints")
out_dir = Path(r"dir/to/save/the/plot")


def display(network, model, metric):
    fig, ax = plt.subplots(figsize=(3, 2))
    CONUS.boundary.plot(ax=ax, color='gray', linewidth=0.5)
    CONUS.plot(ax=ax, color='lightgrey', alpha=0.1)
    if 'rmse' in metric:
        vmin, vmax = 0, 0.2
    elif metric == 'corr':
        vmin, vmax = 0, 1
    elif metric == 'bias':
        vmin, vmax = -0.1, 0.1
    else:
        print(metric)
        raise ValueError('metrics should in corr, rmse, ubrmse, bias')
    network.plot(ax=ax, marker='o', edgecolor='lightgrey', cmap='jet', markersize=12, alpha=0.8,
                 column=metric, vmin=vmin, vmax=vmax)  # legend=True

    plt.axis('off')
    plt.tight_layout()
    plt.savefig(out_dir / f'{metric}_{model}.png')


for model in models:
    for i, year in enumerate([2019, 2020, 2021, 2022]):
        output_dir = root / f"SCAN_USCRN_HLSL30_0.1_{model}_exp1_{'_'.join([str(year) for year in range(year - 3, year)])}"

        test_df = pd.read_csv(output_dir / 'y_test_pred.csv', index_col=0)

        if i == 0:
            result_df = test_df
        else:
            result_df = pd.concat([result_df, test_df]).reset_index(drop=True)

    overall_metrics = defaultdict(list)
    for name, group in result_df.groupby('ID'):
        if group.shape[0] <= 1: continue
        scores = accuracy(group.y_test.values.astype('float'), group.y_pred.values.astype('float'))

        overall_metrics['ID'].append(name)
        for metric, value in scores.items():
            overall_metrics[metric].append(value)

    for metric in list(overall_metrics.keys())[1:]:
        display(network.merge(pd.DataFrame(overall_metrics)), model, metric)



