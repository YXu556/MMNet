import numpy as np
import pandas as pd
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from collections import defaultdict
from sklearn import metrics
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from shapely.geometry import Point

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

from datasets import *
from models import *

LC_2_code = {'Barren': 2,
             'Crop': 6,
             'Developed': 5,
             'Forest': 4,
             'Grassland': 1,
             'Pasture': 7,
             'Shrub': 0,
             'Wetland': 3}


# -------------------------------------- #
#              data utils                #
# -------------------------------------- #
def get_plain_dataset(data, datapath, args, train_idx=None, test_idx=None):
    # data = data[(data.LST * 0.02 - 273.15) > 0]
    data['LC_code'] = data.LC_name.map(LC_2_code)
    # if args.source == 'random':
    #     networks = np.unique(data.ID.values)
    #     idx = np.arange(networks.shape[0])
    #     np.random.shuffle(idx)
    #     train_num = int(networks.shape[0] * 0.8)  # todo
    #     # train_num = -10
    #     train_idx, test_idx = idx[:train_num], idx[train_num:]
    #     source_data = data[data['ID'].isin(networks[train_idx])].reset_index()
    #     target_data = data[data['ID'].isin(networks[test_idx])].reset_index()
    #     # source_data = data[data.ID == 'SCAN_Alkali_Mesa'].reset_index()
    #     # target_data = data[data.ID == 'SCAN_Eastland'].reset_index()
    # elif args.source == 'year':
    #     target_year = int(args.target)
    #     source_years = range(target_year-3, target_year)
    #     source_data = data[data.year.isin(source_years)].reset_index()
    #     target_data = data[data.year==target_year].reset_index()
    # else:
    #     source_data = data[data['region_domains'].apply(lambda x: x in map(int, args.source.split('_')))].reset_index()
    #     target_data = data[data['region_domains'].apply(lambda x: x in map(int, args.target.split('_')))].reset_index()
    if train_idx is not None and test_idx is not None:
        networks = np.unique(data.ID.values)
        source_data = data[data['ID'].isin(networks[train_idx])].reset_index(drop=True)
        target_data = data[data['ID'].isin(networks[test_idx])].reset_index(drop=True)
    elif '9' in args.dataset.split('_') or args.experiment == 0:
        # idx = np.arange(data.shape[0])
        # np.random.shuffle(idx)
        # train_num = int(data.shape[0] * 0.8)
        # train_idx, test_idx = idx[:train_num], idx[train_num:]
        # source_data = data.iloc[train_idx].reset_index()
        # target_data = data.iloc[test_idx].reset_index()
        networks = np.unique(data.ID.values)
        idx = np.arange(networks.shape[0])
        np.random.shuffle(idx)
        train_num = int(networks.shape[0] * 0.8)  # todo
        # train_num = -10
        train_idx, test_idx = idx[:train_num], idx[train_num:]
        source_data = data[data['ID'].isin(networks[train_idx])].reset_index(drop=True)
        target_data = data[data['ID'].isin(networks[test_idx])].reset_index(drop=True)
    elif args.experiment == 1:
        target_year = args.target
        source_years = [int(year) for year in args.source.split('_')]
        source_data = data[data.Year.isin(source_years)].reset_index(drop=True)
        target_data = data[data.Year == target_year].reset_index(drop=True)
    elif args.experiment in [2, 3]:
        networks = np.unique(data.ID.values)
        if train_idx is None or test_idx is None: raise ValueError('train/test (station) idxs are not defined')
        source_data = data[data['ID'].isin(networks[train_idx])].reset_index(drop=True)
        target_data = data[data['ID'].isin(networks[test_idx])].reset_index(drop=True)

    xy_columns = [
        "LC_code", 'DoY_normalized',
        "Latitude", "Longitude",
        "elevation", "slope", "aspect_normalized",
        "clay", "bd", "sand",
        'eto', 'etr', 'pr', 'sph', 'srad', 'vpd', 'vs', 'LST', 'SMAP',
        'VV', 'VH', 'angle',
        'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2',
    ]

    # todo test mlp
    # if args.dataset.endswith('timeseries'):
    #     dynamic_col = [
    #         'eto', 'etr', 'pr', 'sph', 'srad', 'vpd', 'vs', 'LST',  # 'SMAP',
    #     ]
    #
    #     for col in dynamic_col:
    #         xy_columns.extend([col + f'_{i}' for i in range(args.length, 0, -1)])
    # end todo

    if args.use_indices:
        xy_columns += ['ndvi', 'evi', 'ndwi', 'lswi', 'nsdsi',
                       'cr', 'dpsvim', 'pol', 'rvim', 'vvvh']
        xy_columns_new = list(
            set(xy_columns) - {'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2', 'VV', 'VH',
                               'angle', })
        xy_columns_new.sort(key=xy_columns.index)
        xy_columns = xy_columns_new

    if '9' in args.dataset.split('_'):
        xy_columns += ["SMAP"]
    else:
        xy_columns += ['VWC_5']
    args.xy_columns = xy_columns

    source_ids = source_data.ID.values
    target_ids = target_data.ID.values

    source_data = source_data[xy_columns].values
    target_data = target_data[xy_columns].values

    if args.lc_code is not None:
        source_data = lc_encode(source_data, lc_code=args.lc_code)
        target_data = lc_encode(target_data, lc_code=args.lc_code)

    mean = source_data[:, :-1].mean(0)
    std = source_data[:, :-1].std(0)
    std[std == 0] = 1

    source_dataset = SoMoDataset(source_data, mean, std, source_ids)
    target_dataset = SoMoDataset(target_data, mean, std, target_ids)

    args.in_ch = source_dataset.X[0].shape[0]

    return source_dataset, target_dataset


def get_plain_dataset_ts(data, datapath, args, train_idx=None, test_idx=None):

    data['LC_code'] = data.LC_name.map(LC_2_code)

    if '9' in args.dataset.split('_') or args.experiment == 0:
        idx = np.arange(data.shape[0])
        np.random.shuffle(idx)
        train_num = int(data.shape[0] * 0.8)
        train_idx, test_idx = idx[:train_num], idx[train_num:]
        source_data = data.iloc[train_idx].reset_index(drop=True)
        target_data = data.iloc[test_idx].reset_index(drop=True)
    elif args.experiment == 1:
        target_year = args.target
        source_years = [int(year) for year in args.source.split('_')]
        source_data = data[data.Year.isin(source_years)].reset_index(drop=True)
        target_data = data[data.Year == target_year].reset_index(drop=True)
    elif args.experiment in [2, 3]:
        networks = np.unique(data.ID.values)
        if train_idx is None or test_idx is None: raise ValueError('train/test (station) idxs are not defined')
        source_data = data[data['ID'].isin(networks[train_idx])].reset_index(drop=True)
        target_data = data[data['ID'].isin(networks[test_idx])].reset_index(drop=True)

    static_col = [
        "LC_code", 'DoY_normalized',
        "Latitude", "Longitude",
        "elevation", "slope", "aspect_normalized",  # todo1
        "clay", "bd", "sand",
        # 'eto', 'etr', 'pr', 'sph', 'srad', 'vpd', 'vs', 'LST',
        'SMAP',
        'VV', 'VH', 'angle',
        'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2',
    ]
    dynamic_col = [
        'eto', 'etr', 'pr', 'sph', 'srad', 'vpd', 'vs', 'LST',  # 'SMAP',
    ]

    # if args.model in ['Transformer', 'transformer']:  todo
    #     static_col.remove('SMAP')
    #     dynamic_col.append('SMAP')

    d_dynamic_col = []
    for col in dynamic_col:
        d_dynamic_col.extend([col + f'_{i}' for i in range(args.length, 0, -1)] + [col])

    if '9' in args.dataset.split('_'):
        static_col.remove('SMAP')

    if args.use_indices:
        static_col += ['ndvi', 'evi', 'ndwi', 'lswi', 'nsdsi',  # todo1
                       'cr', 'dpsvim', 'pol', 'rvim', 'vvvh'
                       ]
        static_col_new = list(
            set(static_col) - {'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2', 'VV', 'VH',
                               'angle', })
        static_col_new.sort(key=static_col.index)
        static_col = static_col_new

    if args.lc_code is not None:
        source_data_statics = lc_encode(source_data[static_col].values, lc_code=args.lc_code)
        target_data_statics = lc_encode(target_data[static_col].values, lc_code=args.lc_code)
    else:
        source_data_statics = source_data[static_col].values
        target_data_statics = target_data[static_col].values

    mean = source_data_statics.mean(0)
    std = source_data_statics.std(0)
    std[std == 0] = 1

    source_data_dynamic = source_data[d_dynamic_col].values.reshape(source_data.shape[0], -1, args.length + 1)
    target_data_dynamic = target_data[d_dynamic_col].values.reshape(target_data.shape[0], -1, args.length + 1)

    # todo test transform
    # source_data_dynamic = np.concatenate([source_data_dynamic, np.repeat(source_data_statics[:, :, np.newaxis], source_data_dynamic.shape[2], axis=2)], axis=1)
    # target_data_dynamic = np.concatenate([target_data_dynamic, np.repeat(target_data_statics[:, :, np.newaxis], target_data_dynamic.shape[2], axis=2)], axis=1)
    # end todo

    mean_d = source_data_dynamic.mean(0).mean(-1)
    std_d = source_data_dynamic.transpose(0, 2, 1).reshape(-1, source_data_dynamic.shape[1]).std(0)

    np.save(args.output_dir / 'static_mean_std.npy', np.stack([mean, std]))   # todo
    np.save(args.output_dir / 'dynamic_mean_std.npy', np.stack([mean_d, std_d]))

    if '9' in args.dataset.split('_'):
        source_y = source_data["SMAP"]
        target_y = target_data["SMAP"]
    else:
        source_y = source_data["VWC_5"]
        target_y = target_data["VWC_5"]

    source_dataset = SoMoTSDataset(source_data_statics,
                                   source_data_dynamic,
                                   source_y,
                                   mean, std, mean_d, std_d)
    target_dataset = SoMoTSDataset(target_data_statics,
                                   target_data_dynamic,
                                   target_y,
                                   mean, std, mean_d, std_d)
    args.in_ch_s = source_dataset.X_s[0].shape[0]
    args.in_ch_d = source_dataset.X_d[0].shape[0]
    args.in_ch_d += args.in_ch_s  # todo
    return source_dataset, target_dataset


def lc_encode(data, lc_code=None, lc_class_num=8):
    X = data[:, :-1]
    y = data[:, -1]
    if lc_code == 'oh':
        lc_oh = F.one_hot(torch.Tensor(X[:, 0]).to(int), num_classes=lc_class_num).numpy()
        X = np.hstack([lc_oh, X[:, 1:]])
    elif lc_code == 'bin':  # todo 03b use lc_class_num 8
        lc_bin = np.array(list(map(lambda x: list(f'{int(x):03b}'), X[:, 0].tolist())), dtype='int')
        X = np.hstack([lc_bin, X[:, 1:]])
    return np.hstack([X, y.reshape(-1, 1)])


def get_ml_data(source_dataset, target_dataset, args):
    print('Load ML Dataset')
    transform = transforms.Compose([
        ToTensor(),
    ])

    X_source_np = source_dataset.X
    y_source = source_dataset.y
    X_source = list()
    for i, X in enumerate(X_source_np):
        X = (X - source_dataset.mean) / source_dataset.std
        sample = {
            'X': X,
            'y': y_source[i]
        }
        trans = transform(sample)
        X_source.append(trans['X'])
    X_source = torch.stack(X_source)

    # random sample val_dataset according to val_ratio
    num_train = X_source.shape[0]
    indices = list(range(num_train))
    num_val = int(num_train * args.val_ratio)
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[num_val:], indices[:num_val]

    X_train, y_train, id_train = X_source[train_idx], y_source[train_idx], source_dataset.ids[train_idx]
    X_val, y_val, id_val = X_source[valid_idx], y_source[valid_idx], source_dataset.ids[valid_idx]

    X_target_np = target_dataset.X
    y_target = target_dataset.y
    X_target = list()
    for i, X in enumerate(X_target_np):
        X = (X - target_dataset.mean) / target_dataset.std
        sample = {
            'X': X,
            'y': y_target[i]
        }
        trans = transform(sample)
        X_target.append(trans['X'])
    X_target = torch.stack(X_target)

    # print(f"supervised dataset:", args.source)
    print(f"size of train data: {len(source_dataset)}")
    print(f"size of test data: {len(target_dataset)}")

    # if args.cluster:
    #     source_cluster = source_dataset.cluster
    #     target_cluster = target_dataset.cluster
    #     c_train = source_cluster[train_idx]
    #     c_val = source_cluster[valid_idx]
    #     c_target = target_cluster
    #     return (X_train, y_train, id_train, c_train), (X_val, y_val, id_val, c_val), (X_target, y_target, target_dataset.ids, c_target)

    return (X_train, y_train, id_train), (X_val, y_val, id_val), (X_target, y_target, target_dataset.ids)


def get_supervised_dataloader(datasets, args, ratio=1):
    transform = transforms.Compose([
        ToTensor(),
    ])

    # random sample val_dataset according to val_ratio
    num_train = len(datasets)
    indices = list(range(num_train))
    # indices = np.random.choice(num_train, int(num_train*ratio), replace=False)  # todo
    num_val = int(num_train * args.val_ratio * ratio)  # todo
    np.random.shuffle(indices)
    train_idx, valid_idx = indices[num_val:], indices[:num_val]

    traindataset = Subset(deepcopy(datasets), train_idx)
    traindataset.update_transform(transform)
    valdataset = Subset(deepcopy(datasets), valid_idx)
    valdataset.update_transform(transform)

    train_loader = torch.utils.data.DataLoader(
        traindataset, batch_size=args.batchsize, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True, )

    val_loader = torch.utils.data.DataLoader(
        valdataset, batch_size=args.batchsize,
        num_workers=args.workers)

    # print(f"supervised dataset:", args.source)
    print(f"size of train data: {len(traindataset)} ({len(train_loader)} batches)")
    print(f"size of val data: {len(valdataset)} ({len(val_loader)} batches)")

    return train_loader, val_loader


def get_test_dataloader(targetdataset, args):
    test_transform = transforms.Compose([
        ToTensor(),
    ])

    targetdataset.update_transform(test_transform)

    testdataloader = torch.utils.data.DataLoader(
        targetdataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    print(f"test target data: {len(targetdataset)} ({len(testdataloader)} batches)")

    return testdataloader


def get_source_dataloader(sourcedataset, args):
    transform = transforms.Compose([
        ToTensor(),
    ])
    sourcedataset.update_transform(transform)

    sourcedataloader = torch.utils.data.DataLoader(
        sourcedataset, batch_size=args.batchsize,
        num_workers=args.workers, pin_memory=True,
        shuffle=True, drop_last=True, )

    # print(f"source dataset:", args.source)
    print(f"size of source data: {len(sourcedataset)} ({len(sourcedataloader)} batches)")

    return sourcedataloader


def get_map_dataloader(data, ft_weight_path, args):

    data['LC_code'] = data.LC_name.map(LC_2_code)

    xy_columns = [
        "LC_code", 'DoY_normalized',
        "Latitude", "Longitude",
        "elevation", "slope", "aspect_normalized",
        "clay", "bd", "sand",
        'eto', 'etr', 'pr', 'sph', 'srad', 'vpd', 'vs', 'LST',
        'SMAP',
        'VV', 'VH', 'angle',
        'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2',
    ]

    if args.use_indices:
        xy_columns += ['ndvi', 'evi', 'ndwi', 'lswi', 'nsdsi',
                       'cr', 'dpsvim', 'pol', 'rvim', 'vvvh']
        xy_columns_new = list(
            set(xy_columns) - {'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2', 'VV', 'VH',
                               'angle', })
        xy_columns_new.sort(key=xy_columns.index)
        xy_columns = xy_columns_new

    data_ids = data.ID.values
    if args.lc_code is not None:
        data = lc_encode(data[xy_columns].values, lc_code=args.lc_code)
    else:
        data = data[xy_columns].values
    mean, std = np.load(ft_weight_path / 'mean_std.npy')

    y = np.zeros((data.shape[0], 1))
    data = np.hstack([data, y])

    dataset = SoMoDataset(data, mean, std, data_ids, transform=transforms.Compose([ToTensor(),]))

    args.in_ch = dataset.X[0].shape[0]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # print(f"source dataset:", args.source)
    print(f"size of data: {len(dataset)} ({len(dataloader)} batches)")

    return dataloader


def get_map_dataloader_ts(data, ft_weight_path, args):

    data['LC_code'] = data.LC_name.map(LC_2_code)

    static_col = [
        "LC_code", 'DoY_normalized',
        "Latitude", "Longitude",
        "elevation", "slope", #"aspect_normalized",  # todo1
        "clay", "bd", "sand",
        # 'eto', 'etr', 'pr', 'sph', 'srad', 'vpd', 'vs', 'LST',
        'SMAP',
        'VV', 'VH', 'angle',
        'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2',
    ]
    dynamic_col = [
        'eto', 'etr', 'pr', 'sph', 'srad', 'vpd', 'vs', 'LST',  # 'SMAP',
    ]

    if args.model in ['Transformer', 'transformer']:
        static_col.remove('SMAP')
        dynamic_col.append('SMAP')

    d_dynamic_col = []
    for col in dynamic_col:
        d_dynamic_col.extend([col + f'_{i}' for i in range(args.length, 0, -1)] + [col])

    if args.use_indices:
        static_col += ['ndvi', 'evi', 'ndwi', 'lswi',# 'nsdsi',
                       'cr', 'dpsvim', 'pol', #'rvim', 'vvvh'  # todo1
                       ]
        static_col_new = list(
            set(static_col) - {'Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2', 'TIRS1', 'TIRS2', 'VV', 'VH',
                               'angle', })
        static_col_new.sort(key=static_col.index)
        static_col = static_col_new

    if args.lc_code is not None:
        data_statics = lc_encode(data[static_col].values, lc_code=args.lc_code)
    else:
        data_statics = data[static_col].values

    data_dynamic = data[d_dynamic_col].values.reshape(data.shape[0], -1, args.length + 1)

    mean, std = np.load(ft_weight_path / 'static_mean_std.npy')
    mean_d, std_d = np.load(ft_weight_path / 'dynamic_mean_std.npy')

    y = np.zeros(data.shape[0])

    dataset = SoMoTSDataset(data_statics,
                            data_dynamic,
                            y,
                            mean, std, mean_d, std_d,
                            transform=transforms.Compose([ToTensor(),]))

    args.in_ch_s = dataset.X_s[0].shape[0]
    args.in_ch_d = dataset.X_d[0].shape[0]

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True)

    # print(f"source dataset:", args.source)
    print(f"size of data: {len(dataset)} ({len(dataloader)} batches)")

    return dataloader


def cycle(iterable):  # Don't use itertools.cycle, as it repeats the same shuffle
    while True:
        for x in iterable:
            yield x


def cat_samples(Xs):
    out = torch.cat([X for X in Xs])
    return out


# -------------------------------------- #
#              model utils               #
# -------------------------------------- #
def get_model(modelname, args):
    if modelname == 'ridge':
        model = Ridge(alpha=0.1)
    elif modelname == 'rf':
        model = RandomForestRegressor(max_samples=args.batchsize)
    elif modelname == 'xgboost':
        model = XGBRegressor()
    elif modelname == 'mlpsk':
        model = MLPRegressor(
            hidden_layer_sizes=(32, 64, 128, 64, 32),
            batch_size=512,
            max_iter=100,
            learning_rate_init=0.001,
            tol=0,
            verbose=True
        )
    elif modelname == 'mlp':
        model = MLP(args.in_ch).to(args.device)
    elif modelname == 'mmnet':
        model = MMNet(args.in_ch_s, args.in_ch_d).to(args.device)
    elif modelname == 'transformer':
        model = Transformer(args.in_ch_d).to(args.device)
    else:
        raise ValueError(
            "invalid model argument. choose from 'MLP', 'RF', or 'Ridge'")

    return model


def save(model, path="model.pth", **kwargs):
    print(f"saving model to {str(path)}\n")
    model_state = model.state_dict()
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    torch.save(dict(model_state=model_state, **kwargs), path)


# -------------------------------------- #
#               eval utils               #
# -------------------------------------- #
def accuracy(y_test, y_pred):
    valid = ~np.isnan(y_pred)
    y_test = y_test[valid]
    y_pred = y_pred[valid]
    bias = (y_pred - y_test).mean()
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    ubrmse = np.sqrt(metrics.mean_squared_error(y_test - y_test.mean(), y_pred - y_pred.mean()))
    # ubrmse = np.sqrt(rmse ** 2 - (y_test.mean() - y_pred.mean()) ** 2)
    r = np.corrcoef(y_test.flatten(), y_pred.flatten())[0][1]
    return dict(
        corr=r,
        rmse=rmse,
        bias=bias,
        ubrmse=ubrmse
    )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def evaluation(model, criterion, dataloader, device, save_res=False):
    losses = AverageMeter('Loss', ':.4e')
    model.eval()
    with torch.no_grad():
        y_true_list = list()
        y_pred_list = list()
        fea_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, (X, y) in iterator:
                X = X.to(device)
                y = y.to(device)

                pred, fea = model(X, return_feats=True)
                loss = criterion(pred, y)
                iterator.set_description(f"test loss={loss:.4f}")
                losses.update(loss.item(), X.size(0))

                y_true_list.append(y)
                y_pred_list.append(pred)
                fea_list.append(fea)
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()
    fea = torch.cat(fea_list).cpu().numpy()
    scores = accuracy(y_true, y_pred)

    if save_res:
        return losses.avg, scores, y_true, y_pred
    else:
        return losses.avg, scores


def evaluation_BNN(model, criterion, dataloader, device, save_fea=False):
    losses = AverageMeter('Loss', ':.4e')
    model.eval()
    with torch.no_grad():
        y_true_list = list()
        y_pred_list = list()
        fea_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, (X, y) in iterator:
                X = X.to(device)
                y = y.to(device)

                pred, std, fea = model(X, return_feats=True)
                loss = ((pred - y) ** 2 * torch.exp(-std) * 0.5 + std * 0.5).mean()
                iterator.set_description(f"test loss={loss:.4f}")
                losses.update(loss.item(), X.size(0))

                y_true_list.append(y)
                y_pred_list.append(pred)
                fea_list.append(fea)
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()
    fea = torch.cat(fea_list).cpu().numpy()
    scores = accuracy(y_true, y_pred)

    if save_fea:
        return losses.avg, scores, fea, y_true
    else:
        return losses.avg, scores


def evaluation_uncertainty(model, criterion, dataloader, device, save_fea=False):
    losses = AverageMeter('Loss', ':.4e')
    with torch.no_grad():
        y_true_list = list()
        y_pred_list = list()
        fea_list = list()
        T = 100  # todo
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, (X, y) in iterator:
                model.eval()
                pred = []
                X = X.to(device)
                y = y.to(device)

                pred_t_ori, fea = model(X, return_feats=True)
                pred.append(pred_t_ori)
                model.apply(apply_dropout)
                for _ in range(1, T):
                    pred_t = model.decoder(fea)
                    pred.append(pred_t)

                loss = criterion(pred_t_ori, y)
                iterator.set_description(f"test loss={loss:.4f}")
                losses.update(loss.item(), X.size(0))

                y_true_list.append(y)
                y_pred_list.append(torch.cat(pred, dim=1))
                fea_list.append(fea)
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()
    uncertainty = y_pred.var(1)
    # uncertainty = (torch.tensor(y_pred).cuda()**2).mean(1) - (torch.tensor(y_pred).cuda().mean(1))**2
    fea = torch.cat(fea_list).cpu().numpy()
    scores = accuracy(y_true, y_pred[:, 0].reshape(-1, 1))
    scores['uncertainty'] = uncertainty.mean() * 1000  # todo

    if save_fea:
        return losses.avg, scores, fea, y_true
    else:
        return losses.avg, scores, uncertainty


def evaluation_BNN_uncertainty(model, criterion, dataloader, device, save_fea=False):
    losses = AverageMeter('Loss', ':.4e')
    with torch.no_grad():
        y_true_list = list()
        y_pred_list = list()
        y_std_list = list()
        fea_list = list()
        T = 100  # todo
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, (X, y) in iterator:
                model.eval()
                pred = []
                std = []
                X = X.to(device)
                y = y.to(device)

                pred_t_ori, std_t_ori, fea = model(X, return_feats=True)
                pred.append(pred_t_ori)
                std.append(std_t_ori)
                model.apply(apply_dropout)
                for _ in range(1, T):
                    pred_t = model.decoder(fea)
                    std_t = model.std_net(fea)
                    pred.append(pred_t)
                    std.append(std_t)

                loss = criterion(pred_t_ori, y)
                iterator.set_description(f"test loss={loss:.4f}")
                losses.update(loss.item(), X.size(0))

                y_true_list.append(y)
                y_pred_list.append(torch.cat(pred, dim=1))
                y_std_list.append(torch.cat(std, dim=1))
                fea_list.append(fea)
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()
    y_std = torch.cat(y_std_list).cpu().numpy()
    uncertainty = y_pred.var(1) + np.exp(y_std).mean(1)
    # uncertainty = (torch.tensor(y_pred).cuda()**2).mean(1) - (torch.tensor(y_pred).cuda().mean(1))**2
    fea = torch.cat(fea_list).cpu().numpy()
    scores = accuracy(y_true, y_pred[:, 0].reshape(-1, 1))
    scores['uncertainty'] = uncertainty.mean() * 1000  # todo

    if save_fea:
        return losses.avg, scores, fea, y_true
    else:
        return losses.avg, scores, uncertainty


def evaluation_ts(model, criterion, dataloader, device, save_res=False):
    losses = AverageMeter('Loss', ':.4e')
    model.eval()
    with torch.no_grad():
        y_true_list = list()
        y_pred_list = list()
        with tqdm(enumerate(dataloader), total=len(dataloader), leave=True) as iterator:
            for idx, (X_s, X_d, y) in iterator:
                X_s = X_s.to(device)
                X_d = X_d.to(device)
                y = y.to(device)

                pred = model(X_s, X_d)
                loss = criterion(pred, y)
                iterator.set_description(f"test loss={loss:.4f}")
                losses.update(loss.item(), X_s.size(0))

                y_true_list.append(y)
                y_pred_list.append(pred)
    y_true = torch.cat(y_true_list).cpu().numpy()
    y_pred = torch.cat(y_pred_list).cpu().numpy()
    scores = accuracy(y_true, y_pred)

    if save_res:
        return losses.avg, scores, y_true, y_pred
    else:
        return losses.avg, scores


def validation(val_loss_min, model, criterion, dataloader, device, log=None,
               epoch=None, best_model_path=None, train_loss=None):
    val_loss, val_scores = evaluation_ts(model, criterion, dataloader, device)
    val_scores['loss'] = val_loss
    val_scores.update(train_loss)
    scores_msg = ", ".join([f"{k}={v:.4f}" for (k, v) in val_scores.items()])
    print(f"\nValidation results : {scores_msg}")

    if val_loss < val_loss_min:
        print(f'Validation loss improved from {val_loss_min:.4f} to {val_loss:.4f}!\n')
        val_loss_min = val_loss
    else:
        print(f'Validation loss did not improve from {val_loss_min:.4f}.\n')

    if log is not None:
        val_scores["epoch"] = epoch + 1
        log.append(val_scores)
        log_df = pd.DataFrame(log).set_index("epoch")
        log_df.to_csv(best_model_path.parent / "trainlog.csv")

    return val_loss_min


def MMD(x, y, kernel='rbf', device='cuda'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))

    dxx = rx.t() + rx - 2. * xx  # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy  # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz  # Used for C in (1)

    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))

    if kernel == "multiscale":

        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

    if kernel == "rbf":

        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5 * dxx / a)  # a is sigma^2 in guassian kernels
            YY += torch.exp(-0.5 * dyy / a)
            XY += torch.exp(-0.5 * dxy / a)

    return torch.mean(XX + YY - 2. * XY)


def apply_dropout(m):
    if type(m) == nn.Dropout:
        m.train()


def station_acc(df):
    overall_metrics = defaultdict(list)
    for name, group in df.groupby('ID'):
        if group.shape[0] <= 1: continue
        scores = accuracy(group.y_test.values.astype('float'), group.y_pred.values.astype('float'))

        for metric, value in scores.items():
            overall_metrics[metric].append(value)

    mean, median = {}, {}
    for metric, values in overall_metrics.items():
        values = np.array(values)
        if isinstance(values[0], (str)) or np.any(np.isnan(values)):
            continue
        else:
            # print(f"{metric}: {np.median(values):.4f}")
            mean[metric] = np.mean(values)
            median[metric] = np.median(values)
    return mean, median