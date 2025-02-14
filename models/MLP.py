import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    '''The architecture of LeNet with Layers'''

    def __init__(self, input_dim, dropout=0.2):
        super(MLP, self).__init__()

        self.features_dim = 128
        encoder = [input_dim, 32, 64, 128]
        layers = []
        for i in range(len(encoder) - 1):
            layers.extend([
                nn.Linear(encoder[i], encoder[i+1]),
                nn.BatchNorm1d(encoder[i+1]),
                nn.ReLU(),
                GaussianDropout(p=dropout)
            ])
        self.encoder = nn.Sequential(*layers)

        layers = []
        decoder = [128, 64, 32, 1]
        for i in range(len(decoder) - 1):
            layers.append(nn.Linear(decoder[i], decoder[i + 1]))
            if i < (len(decoder) - 2):
                layers.extend([
                    nn.BatchNorm1d(decoder[i + 1]),
                    nn.ReLU(),
                    GaussianDropout(p=dropout)
                ])
        self.decoder = nn.Sequential(*layers)

    def forward(self, x, return_feats=False):
        f = self.encoder(x)
        y = self.decoder(f)
        return (y, f) if return_feats else y


class GaussianDropout(nn.Module):

    def __init__(self, p: float):
        """
        Multiplicative Gaussian Noise dropout with N(1, p/(1-p))
        It is NOT (1-p)/p like in the paper, because here the
        noise actually increases with p. (It can create the same
        noise as the paper, but with reversed p values)

        Source:
        Dropout: A Simple Way to Prevent Neural Networks from Overfitting
        https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

        :param p: float - determines the the standard deviation of the
        gaussian noise, where sigma = p/(1-p).
        """
        super().__init__()
        assert 0 <= p < 1
        self.t_mean = torch.ones((0,))
        self.shape = ()
        self.p = p
        self.t_std = self.compute_std()

    def compute_std(self):
        return self.p / (1 - self.p)

    def forward(self, t_hidden):
        if self.training and self.p > 0.:
            if self.t_mean.shape != t_hidden.shape:
                self.t_mean = torch.ones_like(input=t_hidden
                                              , dtype=t_hidden.dtype
                                              , device=t_hidden.device)
            elif self.t_mean.device != t_hidden.device:
                self.t_mean = self.t_mean.to(device=t_hidden.device, dtype=t_hidden.dtype)

            t_gaussian_noise = torch.normal(self.t_mean, self.t_std)
            t_hidden = t_hidden.mul(t_gaussian_noise)
        return t_hidden
