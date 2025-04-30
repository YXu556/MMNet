import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from pathlib import Path
from utils import *

plot = False
acc = True
models = ['mmnet', 'transformer', 'mlp']  #
sampled_size = 50000
root = Path(r"root/path/to/your/checkpoints")
out_dir = Path(r"dir/to/save/the/plot")

for model in models:
    for i, year in enumerate([2019, 2020, 2021, 2022]):
        output_dir = root / f"SCAN_USCRN_HLSL30_0.1_{model}_exp1_{'_'.join([str(year) for year in range(year - 3, year)])}"

        val = np.load(output_dir / 'y_val_pred.npy')
        test = np.load(output_dir / 'y_test_pred.npy')

        if i == 0:
            y_val_pred = val
            y_test_pred = test
        else:
            y_val_pred = np.vstack([y_val_pred, val])
            y_test_pred = np.vstack([y_test_pred, test])

    if plot:
        y_val_pred_sampled = y_val_pred[np.random.choice(y_val_pred.shape[0], sampled_size, replace=False)].transpose(1, 0) # [0: true, 1: pred]
        y_test_pred_sampled = y_test_pred[np.random.choice(y_test_pred.shape[0], sampled_size, replace=False)].transpose(1, 0)

        # val
        ds_fn = out_dir / 'data' / f'val_{model}_{sampled_size}.npy'
        if ds_fn.exists():
            data = np.load(ds_fn)
            y_val_pred_sampled = data[:2, :]
            val_density = data[-1, :]
        else:
            val_density = gaussian_kde(y_val_pred_sampled)(y_val_pred_sampled)
            np.save(ds_fn, np.vstack([y_val_pred_sampled, val_density]))
        plt.figure(figsize=(3.5, 3.5))
        plt.scatter(y_val_pred_sampled[0], y_val_pred_sampled[1], c=val_density, s=1)
        plt.plot([0, 1], [0, 1], '--', linewidth=1, c='k')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks([0, 1], [0, 1])
        plt.yticks([0, 1], [0, 1])
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.tight_layout()
        plt.clim(0, 50)
        plt.savefig(out_dir / f'val_scatter_{model}.png')

        # test
        ds_fn = out_dir / 'data' / f'test_{model}_{sampled_size}.npy'
        if ds_fn.exists():
            data = np.load(ds_fn)
            y_test_pred_sampled = data[:2, :]
            test_density = data[-1, :]
        else:
            test_density = gaussian_kde(y_test_pred_sampled)(y_test_pred_sampled)
            np.save(ds_fn, np.vstack([y_test_pred_sampled, test_density]))

        plt.figure(figsize=(3.5, 3.5))
        plt.scatter(y_test_pred_sampled[0], y_test_pred_sampled[1], c=test_density, s=1)
        plt.plot([0, 1], [0, 1], '--', linewidth=1, c='k')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.xticks([0, 1], [0, 1])
        plt.yticks([0, 1], [0, 1])
        plt.xlabel('Measured')
        plt.ylabel('Predicted')
        plt.tight_layout()
        plt.clim(0, 50)
        plt.savefig(out_dir / f'test_scatter_{model}.png')

        # plt.show()

    if acc:
        print(model)
        # val
        val_scores = accuracy(y_val_pred[:, 0], y_val_pred[:, 1])
        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in val_scores.items()])
        print(f"Valid results : \n\n {scores_msg}")
        print("\t".join([f"{v:.4f}" for v in val_scores.values()]), "\n\n")

        scores = accuracy(y_test_pred[:, 0], y_test_pred[:, 1])
        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items()])
        print(f"Test results : \n\n {scores_msg}")
        print("\t".join([f"{v:.4f}" for v in scores.values()]), "\n\n")
