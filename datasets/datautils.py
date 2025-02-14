"""
This script is for data augmentation
"""
import random
import numpy as np

import torch
from torch.utils.data import ConcatDataset


# MEAN = {
#     '2019': np.array([0.15, 0.173, 0.19, 0.227, 0.286, 0.31, 0.322, 0.33, 0.266, 0.194]),
#     '2020': np.array([0.095, 0.121, 0.143, 0.179, 0.244, 0.274, 0.288, 0.3, 0.29, 0.206]),
#     '2021': np.array([0.102, 0.127, 0.147, 0.183, 0.245, 0.275, 0.289, 0.3, 0.291, 0.209]),
#     '2022': np.array([0.177, 0.202, 0.227, 0.26, 0.316, 0.345, 0.359, 0.371, 0.4, 0.323])
# }
# STD = {
#     '2019': np.array([0.232, 0.224, 0.226, 0.223, 0.201, 0.193, 0.192, 0.18, 0.119, 0.103]),
#     '2020': np.array([0.12, 0.116, 0.126, 0.125, 0.124, 0.133, 0.133, 0.132, 0.109, 0.097]),
#     '2021': np.array([0.135, 0.13, 0.137, 0.135, 0.129, 0.134, 0.135, 0.132, 0.107, 0.098]),
#     '2022': np.array([0.086, 0.085, 0.097, 0.094, 0.097, 0.11, 0.108, 0.109, 0.114, 0.11])
# }
# ---------------------------- Spectral augmentation ----------------------------
class RandomAddNoise:
    def __call__(self, sample):
        x = sample['x']
        t, c = x.shape
        for i in range(t):
            prob = np.random.rand()
            if prob < 0.15:
                prob /= 0.15
                if prob < 0.5:
                    x[i, :] += -np.abs(np.random.randn(c) * 0.5)  # np.random.uniform(low=-0.5, high=0, size=(c,))
                else:
                    x[i, :] += np.abs(np.random.randn(c) * 0.5)  # np.random.uniform(low=0, high=0.5, size=(c,))
        sample['x'] = x
        return sample


# ---------------------------- Temporal augmentation ----------------------------
class RandomTempShift:
    def __init__(self, max_shift=30, p=0.5):
        self.max_shift = max_shift
        self.p = p

    def __call__(self, sample):
        p = np.random.rand()
        doy = sample['doy']
        if p < self.p:
            # t_shifts = random.randint(-self.max_shift, self.max_shift)
            # sample['x'] = np.roll(sample['x'], t_shifts, axis=0)
            shift = np.clip(np.random.randn() * 0.3, -1,
                            1) * self.max_shift  # random.randint(-self.max_shift, self.max_shift)
            doy = doy + shift
        sample['doy'] = doy
        return sample


class RandomTempRemoval:
    def __call__(self, sample):
        x = sample['x']
        doy = sample['doy']
        mask = [1 if random.random() < 0.15 else 0 for _ in range(x.shape[0])]
        mask = np.array(mask) == 0
        sample['x'] = x[mask]
        sample['doy'] = doy[mask]

        return sample


# -------------------------- Data process ---------------------- #
class Normalize:
    def __init__(self, version='v1', scale=1):  # todo current is based on west&east
        self.mean = np.array([501.018, 1.845, 193.682, 181.218, 3.348, 14.972, 44.53, 1.403, 0.213, 0.236, 3.313,
                              -11.861, -18.917, 38.529, ])
        self.std = np.array([662.801, 2.55, 97.162, 6.367, 2.083, 10.544, 30.634, 0.145, 0.115, 0.114, 10.035,
                             2.235, 3.138, 3.122,])
        if version in ['v1']:
            self.mean = np.concatenate([self.mean, np.array([11429.468, 17932.061, 16910.526, 13956.591, 44433.591])])
            self.std = np.concatenate([self.std, np.array([2594.562, 3198.039, 3157.509, 3382.978, 3551.174])])
        elif version in ['v2']:
            self.mean = np.concatenate([self.mean, np.array([14900.073, 2887.378, 4168.633, 4134.141, 3164.785, 2031.439])])
            self.std = np.concatenate([self.std, np.array([570.282, 2551.37, 1987.7, 1517.181, 1220.707, 1057.445])])
        elif version in ['v3']:
            self.mean = np.concatenate([self.mean, np.array([11429.456, 17932.133, 16910.526, 13956.56, 44433.674,
                                                       14919.872, 2858.322, 4118.205, 4101.489, 3165.971, 2049.182])])
            self.std = np.concatenate([self.std, np.array([2594.603, 3198.036, 3157.538, 3383.028, 3551.093,
                                                     569.225, 2519.909, 1977.705, 1515.158, 1215.413, 1057.232])])
        self.scale = scale

    def __call__(self, sample):
        x, y = sample['X'], sample['y']
        x = (x - self.mean) / self.std
        sample['X'] = x
        sample['y'] = y * self.scale
        return sample


class ToTensor:
    def __call__(self, sample):
        for k, v in sample.items():
            if 'X' in k:
                sample[k] = torch.from_numpy(v).type(torch.FloatTensor)
            else:
                sample[k] = torch.from_numpy(np.array([v])).type(torch.FloatTensor)

        return sample


# -------------------------- Dataset utils ------------------------ #
class Subset(torch.utils.data.Dataset):

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.X = dataset.X[indices]
        self.y = dataset.y[indices]

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]

    def __len__(self):
        return len(self.indices)

    def update_transform(self, transform):
        self.dataset.transform = transform


# -------------------------- Dataloader utils ------------------------ #
class BalancedBatchSampler(torch.utils.data.sampler.BatchSampler):
    """
    BatchSampler - from a MNIST-like dataset, samples n_samples for each of the n_classes.
    Returns batches of size n_classes * (batch_size // n_classes)
    adapted from https://github.com/adambielski/siamese-triplet/blob/master/datasets.py
    """

    def __init__(self, labels, batch_size):
        classes = sorted(set(labels))
        print(classes)

        n_classes = len(classes)
        self._n_samples = batch_size // n_classes
        if self._n_samples == 0:
            raise ValueError(
                f"batch_size should be bigger than the number of classes, got {batch_size}"
            )

        self._class_iters = [
            InfiniteSliceIterator(np.where(labels == class_)[0], class_=class_)
            for class_ in classes
        ]

        batch_size = self._n_samples * n_classes
        self.n_dataset = len(labels)
        self._n_batches = self.n_dataset // batch_size
        if self._n_batches == 0:
            raise ValueError(
                f"Dataset is not big enough to generate batches with size {batch_size}"
            )
        print("K=", n_classes, "nk=", self._n_samples)
        print("Batch size = ", batch_size)

    def __iter__(self):
        for _ in range(self._n_batches):
            indices = []
            for class_iter in self._class_iters:
                indices.extend(class_iter.get(self._n_samples))
            np.random.shuffle(indices)
            yield indices

        for class_iter in self._class_iters:
            class_iter.reset()

    def __len__(self):
        return self._n_batches


class InfiniteSliceIterator:
    def __init__(self, array, class_):
        assert type(array) is np.ndarray
        self.array = array
        self.i = 0
        self.class_ = class_

    def reset(self):
        self.i = 0

    def get(self, n):
        len_ = len(self.array)
        # not enough element in 'array'
        if len_ < n:
            print(f"there are really few items in class {self.class_}")
            self.reset()
            np.random.shuffle(self.array)
            mul = n // len_
            rest = n - mul * len_
            return np.concatenate((np.tile(self.array, mul), self.array[:rest]))

        # not enough element in array's tail
        if len_ - self.i < n:
            self.reset()

        if self.i == 0:
            np.random.shuffle(self.array)
        i = self.i
        self.i += n
        return self.array[i: self.i]
