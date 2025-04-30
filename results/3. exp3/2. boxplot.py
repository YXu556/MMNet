import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path
from utils import *


models = ['mmnet', 'mlp', 'transformer']
root = Path(r"root/path/to/your/checkpoints")
AllSamples = pd.read_csv(r"path/to/your/SCAN_USCRN_HLSL30_0.1_ts_interpolated_timeseries.csv")
titles = dict(zip(['corr', 'rmse', 'bias', 'ubrmse'], ['R', 'RMSE', 'Bias', 'unRMSE']))
overall_metrics = defaultdict(defaultdict)
for model in models:
    if model == 'mmnet':
        output_dir = root / f"SCAN_USCRN_HLSL30_0.1_{model}_exp3"
    else:
        output_dir = root / f"SCAN_USCRN_HLSL30_0.1_{model}_exp3_flatten"
    for i, domain in enumerate(['W', 'M', 'E']):
        test_df = pd.read_csv(output_dir / domain / f'y_test_pred.csv', index_col=0)

        if i == 0:
            result_df = test_df
        else:
            result_df = pd.concat([result_df, test_df]).reset_index(drop=True)

    model_metrics = defaultdict(list)
    for name, group in result_df.groupby('ID'):
        if group.shape[0] <= 1: continue
        scores = accuracy(group.y_test.values.astype('float'), group.y_pred.values.astype('float'))

        model_metrics['ID'].append(name)
        for metric, value in scores.items():
            model_metrics[metric].append(value)
    overall_metrics[model] = model_metrics

data = AllSamples
model_metrics = defaultdict(list)
for name, group in data.groupby('ID'):
    if group.shape[0] <= 1: continue
    scores = accuracy(group.VWC_5.values.astype('float'), group.SMAP.values.astype('float'))

    model_metrics['ID'].append(name)
    for metric, value in scores.items():
        model_metrics[metric].append(value)
overall_metrics['SMAP'] = model_metrics

fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(8, 3))
bplots = []
for i, metric in enumerate(['corr', 'rmse', 'bias', 'ubrmse']):
    ms = {}
    for model, model_metrics in overall_metrics.items():
        ms[model] = model_metrics[metric]

    bplot = ax[i].boxplot(ms.values(),
                  notch=True,
                  vert=True,
                  showfliers=False,
                  patch_artist=True,
                  medianprops={'color': 'k'})
    ax[i].set_title(titles[metric])
    if i == 0:
        ax[i].set_ylim(0.2, 1)
    elif i == 1:
        ax[i].set_ylim(0, 0.2)
        ax[i].set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])
    elif i == 2:
        ax[i].set_ylim(-0.2, 0.2)
        ax[i].set_yticks([-0.20, -0.10, 0.00, 0.10, 0.20])
    elif i == 3:
        ax[i].set_ylim(0, 0.2)
        ax[i].set_yticks([0.00, 0.05, 0.10, 0.15, 0.20])

    ax[i].grid(axis='y', linestyle='--')
    ax[i].set_xticks([], [])
    bplots.append(bplot)

# fill with colors
colors = ['pink', 'lightblue', 'lightgreen', 'lightgrey']
for bplot in bplots:
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
plt.tight_layout()
plt.show()
