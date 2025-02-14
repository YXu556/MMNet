"""
This script is for training/evaluation of MLP
"""
import copy
import json
import random
import argparse
import sklearn.metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import KFold
from joblib import dump, load
from sklearn.ensemble import RandomForestClassifier

import torch
import torch.optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import LambdaLR

from utils import *
from datasets import *

# path to dataset
DATAPATH = Path(r'D:\DadaX\PhD\Research\0. SoilMoisture\data')


def parse_args():
    parser = argparse.ArgumentParser()

    # Setup parameters
    parser.add_argument('--dataset', default='SCAN_USCRN_HLSL30_0.1_interpolated_timeseries', help='dataset name')
    parser.add_argument('--length', default=10, type=int, help='the length of time series data')
    parser.add_argument("--use_indices", action='store_true', default=True,
                        help='use indices instead of original bands')
    parser.add_argument('--lc_code', default=None, help='how to encode LC, choose from [None, oh, bin]')
    parser.add_argument('--target', default=None, help='test dataset (use lowercase)')
    parser.add_argument('-e', '--experiment', default=1, type=int,
                        help='no. of experiment setting choice from [0: dependent|1: on-site|2. off-site|3: UDA]')
    parser.add_argument('--source', nargs='+', type=str, default=None,
                        help='year of source dataset')
    parser.add_argument("--plot", action='store_true',
                        help='whether to store the plot')
    parser.add_argument("--return_pred", action='store_true',
                        help='whether to return the pred')
    parser.add_argument("--save_pred", action='store_true',
                        help='whether to save the pred')
    parser.add_argument("--val_ratio", default=0.1, type=float,
                        help='Ratio of training data to use for validation. Default 10%.')
    parser.add_argument('--seed', default=1, type=int,
                        help='random seed')
    parser.add_argument('--output_dir', default='results/checkpoints',
                        help='logdir to store progress and models (defaults to ./results/checkpoints)')
    parser.add_argument('-s', '--suffix', default=None,
                        help='suffix to output_dir')
    parser.add_argument('-j', '--workers', type=int, default=0,
                        help='number of CPU workers to load the next batch')
    parser.add_argument('-d', '--device', type=str, default=None,
                        help='torch.Device. either "cpu" or "cuda". default will check by torch.cuda.is_available() ')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10)')
    parser.add_argument('--eval', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--overall', action='store_true',
                        help='print overall results, if exists')
    parser.add_argument('--weights', type=str, help='restore specific model to eval')
    parser.add_argument('--finetune', action='store_true', help='finetune on the target')

    # Training configuration
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('-b', '--batchsize', type=int, default=512,
                        help='batch size (number of time series processed simultaneously)')
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01,
                        help='optimizer learning rate (default 0.02)')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                        help='optimizer weight_decay (default 1e-4)')
    parser.add_argument('--model', type=str, default="MLP",
                        help='select model architecture.')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout rate.')

    args = parser.parse_args()

    # Setup folders based on method and target
    args.model = args.model.lower()  # make case invariant

    if '9' in args.dataset.split('_'):
        args.output_dir = Path(args.output_dir) / f"{'_'.join(args.dataset.split('_')[:3])}_{args.model}_coarse"
    else:
        args.output_dir = Path(
            args.output_dir) / f"{'_'.join(args.dataset.split('_')[:4])}_{args.model}_exp{args.experiment}"

    if args.suffix:
        args.output_dir = args.output_dir.parent / f"{args.output_dir.name}_{args.suffix}"

    if args.experiment in [1] and args.target:
        args.target = int(args.target)
        if args.source:
            args.source = '_'.join(args.source)
        else:
            args.source = '_'.join([str(year) for year in range(args.target - 3, args.target)])
        args.output_dir = args.output_dir.parent / (args.output_dir.name + f'_{args.source}')

    if args.eval and args.weights:
        args.output_dir = args.output_dir.parent / args.weights

    # Setup device
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    args.datapath = DATAPATH

    if args.plot:
        plot_path = Path('results/smvis')
        args.plot_dir = plot_path / args.output_dir.name
        args.plot_dir.mkdir(exist_ok=True, parents=True)

    print(args)

    return args


def main(args):
    # load supervised datasets
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # load supervised datasets
    print('Load all data')
    dataname = f'{args.dataset}.csv'
    filename = args.datapath / dataname

    data = pd.read_csv(filename)
    data = data.drop_duplicates(subset=['ID', 'Date']).reset_index(drop=True)
    if args.experiment == 2:
        stations = np.unique(data.ID.values)
        result_df = pd.DataFrame(np.stack([data.ID, data.VWC_5, np.zeros_like(data.ID)]).transpose(1, 0),
                              columns=['ID', 'y_test', 'y_pred'])

        # k-fold cross validation
        kf = KFold(random_state=args.seed, shuffle=True)
        for i, (train_idx, test_idx) in enumerate(kf.split(stations)):
            print(f'==========[{i + 1}/{kf.n_splits}] Fold {i+1} ==========')
            fold_dir = args.output_dir / f'Fold_{i+1}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            y_pred = train_supervised(data, fold_dir, args, train_idx, test_idx, return_pred=True)
            result_df.loc[data['ID'].isin(stations[test_idx]), 'y_pred'] = y_pred.flatten()

        mean, median = station_acc(result_df)
        overall = accuracy(result_df['y_test'].values.astype('float'), result_df['y_pred'].values.astype('float'))

        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in mean.items()])
        print(f"Station results (mean) : \n\n {scores_msg}")
        print("\t".join([f"{v:.4f}" for v in mean.values()]), "\n\n")

        mean['epoch'] = 'mean'
        median['epoch'] = 'median'
        overall['epoch'] = 'overall'

        log_df = pd.DataFrame([mean, median, overall]).set_index("epoch")
        fn = args.output_dir / f"testlog.csv"
        log_df.to_csv(fn)
    elif args.experiment == 3:
        stations = np.unique(data.ID.values)
        station_domain = data.set_index('ID')['Domain'].to_dict()
        result_df = pd.DataFrame(np.stack([data.ID, data.VWC_5, np.zeros_like(data.ID)]).transpose(1, 0),
                              columns=['ID', 'y_test', 'y_pred'])

        # iter from ['W', 'M', 'E']
        for i, test_domain in enumerate(['W', 'M', 'E']):
            print(f'==========[{i + 1}/3] Test on {test_domain} ==========')
            fold_dir = args.output_dir / f'{test_domain}'
            fold_dir.mkdir(parents=True, exist_ok=True)
            test_stations = [k for k, v in station_domain.items() if v == test_domain]
            test_idx = np.where(np.isin(stations, test_stations))[0]
            train_idx = np.where(~np.isin(stations, test_stations))[0]

            y_pred = train_supervised(data, fold_dir, args, train_idx, test_idx, return_pred=True)

            result_df.loc[data['ID'].isin(test_stations), 'y_pred'] = y_pred.flatten()

        mean, median = station_acc(result_df)
        overall = accuracy(result_df['y_test'].values.astype('float'), result_df['y_pred'].values.astype('float'))

        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in mean.items()])
        print(f"Station results (mean) : \n\n {scores_msg}")
        print("\t".join([f"{v:.4f}" for v in mean.values()]), "\n\n")

        mean['epoch'] = 'mean'
        median['epoch'] = 'median'
        overall['epoch'] = 'overall'

        log_df = pd.DataFrame([mean, median, overall]).set_index("epoch")
        fn = args.output_dir / f"testlog.csv"  # todo
        log_df.to_csv(fn)
    else:
        train_supervised(data, args.output_dir, args, return_pred=args.return_pred)


def train_supervised(data, output_dir, args, train_idx=None, test_idx=None, return_pred=False):
    source_dataset, target_dataset = get_plain_dataset(data, args.datapath, args, train_idx, test_idx)

    print("=> creating model '{}'".format(args.model))
    device = torch.device(args.device)
    model = get_model(args.model, args)

    if args.model in ['ridge', 'rf', 'xgboost', 'mlpsk']:
        best_model_path = output_dir / 'model_best.joblib'
        traindataset, valdataset, testdataset = get_ml_data(source_dataset, target_dataset, args)
        # if args.cluster:
        #     X_train, y_train, id_train, c_train = traindataset
        #     X_val, y_val, id_val, c_val = valdataset
        #     X_test, y_test, id_test, c_test = testdataset
        #
        #     y_val_pred = np.zeros_like(y_val)
        #     y_pred = np.zeros_like(y_test)
        #
        #     for i in range(5):
        #         print(f'---------------cluster {i + 1}------------------')
        #         if (c_test == i).sum() == 0:
        #             continue
        #         X_train_cluster, y_train_cluster = X_train[c_train == i], y_train[c_train == i]
        #         X_val_cluster, y_val_cluster = X_val[c_val == i], y_val[c_val == i]
        #         X_test_cluster, y_test_cluster = X_test[c_test == i], y_test[c_test == i]
        #         model.fit(X_train_cluster, y_train_cluster)
        #         y_val_pred_cluster = model.predict(X_val_cluster)
        #         y_val_pred[c_val == i] = y_val_pred_cluster
        #         y_pred_cluster = model.predict(X_test_cluster)
        #         y_pred[c_test == i] = y_pred_cluster
        #
        #     if args.station_acc:
        #         val_df = pd.DataFrame(np.stack([id_val, y_val, y_val_pred]).transpose(1, 0),
        #                               columns=['ID', 'y_test', 'y_pred'])
        #         mean, median = station_acc(val_df)
        #
        #         scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in mean.items()])
        #         print(f"Valid mean : \n{scores_msg}")
        #         print("\t".join([f"{v:.4f}" for v in mean.values()]), "\n\n")
        #         scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in median.items()])
        #         print(f"Valid median : \n{scores_msg}")
        #         print("\t".join([f"{v:.4f}" for v in median.values()]), "\n\n")
        #
        #         test_df = pd.DataFrame(np.stack([id_test, y_test, y_pred]).transpose(1, 0),
        #                                columns=['ID', 'y_test', 'y_pred'])
        #         mean, median = station_acc(test_df)
        #         scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in mean.items()])
        #         print(f"Test mean : \n{scores_msg}")
        #         print("\t".join([f"{v:.4f}" for v in mean.values()]), "\n\n")
        #         scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in median.items()])
        #         print(f"Test median : \n{scores_msg}")
        #         print("\t".join([f"{v:.4f}" for v in median.values()]), "\n\n")
        #     else:
        #
        #         val_scores = accuracy(y_val, y_val_pred)
        #
        #         scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in val_scores.items()])
        #         print(f"Valid results : \n{scores_msg}")
        #         print("\t".join([f"{v:.4f}" for v in val_scores.values()]), "\n\n")
        #
        #         scores = accuracy(y_test, y_pred)
        #
        #         scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items()])
        #         print(f"Test results : \n{scores_msg}")
        #         print("\t".join([f"{v:.4f}" for v in scores.values()]), "\n\n")
        #
        #     return

        X_train, y_train, id_train = traindataset
        X_val, y_val, id_val = valdataset
        X_test, y_test, id_test = testdataset

        if not args.eval:
            print(f'training {args.model}...')

            model.fit(X_train, y_train)
            dump(model, best_model_path)
        print('Restoring best model weights for testing...')
        model = load(best_model_path)

        y_val_pred = model.predict(X_val)
        y_pred = model.predict(X_test)

        # if args.station_acc:
        #     val_df = pd.DataFrame(np.stack([id_val, y_val, y_val_pred]).transpose(1, 0),
        #                           columns=['ID', 'y_test', 'y_pred'])
        #     mean, median = station_acc(val_df)
        #
        #     scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in mean.items()])
        #     print(f"Valid mean : \n{scores_msg}")
        #     print("\t".join([f"{v:.4f}" for v in mean.values()]), "\n\n")
        #     scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in median.items()])
        #     print(f"Valid median : \n{scores_msg}")
        #     print("\t".join([f"{v:.4f}" for v in median.values()]), "\n\n")
        #
        #     test_df = pd.DataFrame(np.stack([id_test, y_test, y_pred]).transpose(1, 0),
        #                            columns=['ID', 'y_test', 'y_pred'])
        #     mean, median = station_acc(test_df)
        #     scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in mean.items()])
        #     print(f"Test mean : \n{scores_msg}")
        #     print("\t".join([f"{v:.4f}" for v in mean.values()]), "\n\n")
        #     scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in median.items()])
        #     print(f"Test median : \n{scores_msg}")
        #     print("\t".join([f"{v:.4f}" for v in median.values()]), "\n\n")

        val_scores = accuracy(y_val, y_val_pred)
        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in val_scores.items()])
        print(f"Valid results : \n{scores_msg}")
        print("\t".join([f"{v:.4f}" for v in val_scores.values()]), "\n\n")

        scores = accuracy(y_test, y_pred)
        scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items()])
        print(f"Test results : \n{scores_msg}")
        print("\t".join([f"{v:.4f}" for v in scores.values()]), "\n\n")

        val_scores['epoch'] = 'train'
        scores['epoch'] = 'test'

        log_df = pd.DataFrame([val_scores, scores]).set_index("epoch")
        if args.target:
            fn = output_dir / f"testlog_{args.target}.csv"
        else:
            fn = output_dir / f"testlog.csv"
        log_df.to_csv(fn)
        return y_pred

    # load source dataset
    print("=> creating train/val dataloader")
    traindataloader, valdataloader = get_supervised_dataloader(source_dataset, args=args)
    testdataloader = get_test_dataloader(target_dataset, args)

    model.apply(weight_init)
    model.to(device)
    best_model_path = output_dir / 'model_best.pth'

    if args.weights:
        if 'exp2' in args.weights or 'exp3' in args.weights:
            ft_weight_path = str(best_model_path).replace(best_model_path.parts[2], args.weights)
        else:
            ft_weight_path = best_model_path.parents[2] / args.weights / 'model_best.pth'
        checkpoint = torch.load(ft_weight_path)
        state_dict = checkpoint['model_state']
        model.load_state_dict(state_dict)

    if not args.eval:
        criterion = torch.nn.MSELoss().to(device)
        parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=10, eta_min=1e-4
        )

        log = list()
        val_loss_min = np.Inf
        for epoch in range(args.epochs):
            train_loss = train_epoch(model, optimizer, criterion, traindataloader, device)
            val_loss, scores = evaluation(model, criterion, valdataloader, device)
            # lr_scheduler.step()

            scores_msg = ", ".join(
                [f"{k}={v:.4f}" for (k, v) in scores.items() if k not in ['class_f1', 'confusion_matrix']])
            print(f"epoch {epoch + 1}: trainloss={train_loss:.4f}, valloss={val_loss:.4f} " + scores_msg)

            scores["epoch"] = epoch + 1
            scores["trainloss"] = train_loss
            scores["testloss"] = val_loss
            log.append(scores)

            log_df = pd.DataFrame(log).set_index("epoch")
            log_df.to_csv(best_model_path.parent / "trainlog.csv")

            if val_loss < val_loss_min:
                not_improved_count = 0
                print(f'Validation loss improved from {val_loss_min:.4f} to {val_loss:.4f}!')
                val_loss_min = val_loss
                save(model, path=best_model_path, criterion=criterion)
            else:
                not_improved_count += 1
                print(f'Validation loss did not improve from {val_loss_min:.4f} for {not_improved_count} epochs')

            if not_improved_count >= 10:
                print("\nValidation performance didn\'t improve for 10 epochs. Training stops.")
                break

        if epoch == args.epochs - 1:
            print(f"\n{args.epochs} epochs training finished.")

    print('Restoring best model weights for testing...')
    if args.eval and args.weights is not None:
        best_model_path = ft_weight_path
    checkpoint = torch.load(best_model_path)
    state_dict = checkpoint['model_state']
    criterion = checkpoint['criterion']
    model.load_state_dict(state_dict)

    # test on valid
    if return_pred:
        val_loss, val_scores, y_val, y_pred_val = evaluation(model, criterion, valdataloader, device, save_res=True)
        if args.save_pred:
            np.save(best_model_path.parent / 'y_val_pred', np.hstack([y_val, y_pred_val]))
    else:
        val_loss, val_scores = evaluation(model, criterion, valdataloader, device)

    scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in val_scores.items()])
    print(f"Valid results : \n\n {scores_msg} \n\n")
    print("\t".join([f"{v:.4f}" for v in val_scores.values()]), "\n\n")

    # test on test
    if return_pred or args.save_pred:
        test_loss, scores, y_test, y_pred = evaluation(model, criterion, testdataloader, device, save_res=True)
        if args.save_pred:
            if args.experiment == 1:
                result_df = pd.DataFrame(np.hstack([data[data.Year == args.target].reset_index(drop=True)[['ID', 'Date']], y_test, y_pred]),
                                         columns=['ID', 'Date', 'y_test', 'y_pred'])
                result_df.to_csv(best_model_path.parent / f'y_test_pred_{args.target}.csv')
                np.save(best_model_path.parent / f'y_test_pred_{args.target}.npy', np.hstack([y_test, y_pred]))
            elif args.experiment in [2, 3]:
                stations = np.unique(data.ID.values)
                result_df = pd.DataFrame(np.hstack([data[data['ID'].isin(stations[test_idx])].reset_index(drop=True)[['ID', 'Date']], y_test, y_pred]),
                                         columns=['ID', 'Date', 'y_test', 'y_pred'])
                result_df.to_csv(best_model_path.parent / f'y_test_pred.csv')
                np.save(best_model_path.parent / f'y_test_pred.npy', np.hstack([y_test, y_pred]))
    else:
        test_loss, scores = evaluation(model, criterion, testdataloader, device)

    scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in scores.items()])
    print(f"Test results : \n\n {scores_msg} \n\n")
    print("\t".join([f"{v:.4f}" for v in scores.values()]), "\n\n")

    val_scores['epoch'] = 'train'
    val_scores['loss'] = val_loss
    scores['epoch'] = 'test'
    scores['loss'] = test_loss

    log_df = pd.DataFrame([val_scores, scores]).set_index("epoch")
    if args.target:
        fn = output_dir / f"testlog_{args.target}.csv"
    else:
        fn = output_dir / f"testlog.csv"
    log_df.to_csv(fn)

    if return_pred:  return y_pred


def train_epoch(model, optimizer, criterion, dataloader, device):
    losses = AverageMeter('Loss', ':.4e')

    model.train()
    with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
        for idx, (X, y) in iterator:
            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)

            # add l2 loss
            loss += sum(p.pow(2).sum() for p in model.parameters() if len(p.shape) == 2) / X.shape[
                0] * args.weight_decay
            loss /= 2

            loss.backward()
            optimizer.step()
            iterator.set_description(f"train loss={loss:.2f}")

            losses.update(loss.item(), X.size(0))

    return losses.avg


if __name__ == '__main__':
    args = parse_args()
    if args.experiment == 1 and args.target is None:
        suffix = args.output_dir.name
        for year in [2019, 2020, 2021, 2022]:
            print(f'========== {year} ==========')
            args.target = year
            args.source = '_'.join([str(year) for year in range(year - 3, year)])
            args.output_dir = args.output_dir.parent / (suffix + f'_{args.source}')
            if args.eval and args.weights:
                args.output_dir = args.output_dir.parent / args.weights
            main(args)
    else:
        main(args)
