import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, input_dim_d, d_model=128, n_head=16, n_layers=1, d_inner=256,
                 activation="relu", dropout=0.2, max_len=11, T=1000, ):
        super(Transformer, self).__init__()

        self.features_dim = d_model
        self.max_len = max_len

        # dynamic
        encoder_d = [input_dim_d, 32, 64, d_model]
        layers = []
        for i in range(len(encoder_d) - 1):
            layers.append(linlayer(encoder_d[i], encoder_d[i + 1], dropout=dropout))
        self.encoder_d = nn.Sequential(*layers)

        self.inlayernorm = nn.LayerNorm(d_model)

        self.position_enc = PositionalEncoding(d_model, max_len=max_len, T=T)

        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformerencoder = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        layers = []
        decoder = [d_model, 64, 32, 1]
        for i in range(len(decoder) - 1):
            layers.append(nn.Linear(decoder[i], decoder[i + 1]))
            if i < (len(decoder) - 2):
                layers.extend([
                    nn.BatchNorm1d(decoder[i + 1]),
                    nn.ReLU(),
                    GaussianDropout(p=dropout)
                ])
        self.decoder = nn.Sequential(*layers)

    def forward(self, x_s, x_d, return_feats=False):
        # todo
        x_d = torch.cat([x_s.unsqueeze(1).expand(-1, 11, -1), x_d], dim=2)
        # dynamic
        b, s, c = x_d.shape
        x_d = x_d.permute((0, 2, 1))
        x_d = self.encoder_d(x_d)
        x_d = x_d.permute((0, 2, 1))

        x_d = self.inlayernorm(x_d)

        src_pos = torch.arange(s, dtype=torch.long).expand(b, s).cuda()  # todo
        x_d = x_d + self.position_enc(src_pos)

        x_d = x_d.transpose(0, 1)  # N x T x D -> T x N x D
        x_d = self.transformerencoder(x_d)
        f_d = x_d.transpose(0, 1)  # T x N x D -> N x T x D

        # f_d = f_d.max(1)[0]  concat f_d and SMAP
        f = f_d.mean(1)

        # decoder
        y = self.decoder(f)

        return (y, f) if return_feats else y


class linlayer(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super(linlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.do = GaussianDropout(p=dropout)

    def forward(self, input):
        input = input.permute((0, 2, 1))  # to channel last
        out = self.lin(input)
        out = out.permute((0, 2, 1))  # to channel first
        out = self.bn(out)
        out = F.relu(out)
        out = self.do(out)

        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, T: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(T) / d_model))
        pe = torch.zeros(max_len + 1, d_model)
        pe[1:, 0::2] = torch.sin(position * div_term)
        pe[1:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, doy):
        """
        Args:
            doy: Tensor, shape [batch_size, seq_len]
        """
        return self.pe[doy]


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
