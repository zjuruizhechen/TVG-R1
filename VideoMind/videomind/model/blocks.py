# Copyright (c) 2025 Ye Liu. Licensed under the BSD-3-Clause License.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from nncore.nn import Parameter


class Permute(nn.Module):

    def forward(self, x):
        return x.transpose(-1, -2)


class LearnableEmbedding(nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.weights = Parameter(1, 1, dims)

    def forward(self, x):
        return x + self.weights.expand_as(x)


class ConvPyramid(nn.Module):

    def __init__(self, dims, strides, act_cls=nn.ReLU):
        super().__init__()

        self.blocks = nn.ModuleList()
        for s in strides:
            p = int(math.log2(s))
            if p == 0:
                layers = act_cls()
            else:
                conv_cls = nn.Conv1d if p > 0 else nn.ConvTranspose1d
                layers = nn.Sequential()
                for _ in range(abs(p)):
                    module = [Permute(), conv_cls(dims, dims, 2, stride=2), Permute(), nn.LayerNorm(dims), act_cls()]
                    layers.extend(module)
            self.blocks.append(layers)

        self.strides = strides

    def forward(self, x, mask, return_mask=False):
        pymid, pymid_msk = [], []

        for s, blk in zip(self.strides, self.blocks):
            if x.size(1) < s:
                continue

            pymid.append(blk(x))

            if return_mask:
                if s > 1:
                    msk = F.max_pool1d(mask.float(), s, stride=s).long()
                elif s < 1:
                    msk = mask.repeat_interleave(int(1 / s), dim=1)
                else:
                    msk = mask
                pymid_msk.append(msk)

        return (pymid, pymid_msk) if return_mask else pymid


class Scale(nn.Module):

    def __init__(self, strides):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(len(strides)))

    def forward(self, x, i):
        return x * self.scale[i]


class ConvHead(nn.Module):

    def __init__(self, dims, out_dims, kernal_size=3, act_cls=nn.ReLU):
        super().__init__()

        # yapf:disable
        self.module = nn.Sequential(
            Permute(),
            nn.Conv1d(dims, dims, kernal_size, padding=kernal_size // 2),
            act_cls(),
            nn.Conv1d(dims, out_dims, kernal_size, padding=kernal_size // 2),
            Permute())
        # yapf:enable

    def forward(self, x):
        return self.module(x)
