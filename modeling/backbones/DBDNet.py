
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author:1111
# datetime:2019/11/21 22:48
# software: PyCharm
from torch import nn
import random

class BatchDrop(nn.Module):
    def __init__(self, h_ratio, w_ratio):
        super(BatchDrop, self).__init__()
        self.h_ratio = h_ratio
        self.w_ratio = w_ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rh = round(self.h_ratio * h)
            rw = round(self.w_ratio * w)
            sx = random.randint(0, h - rh)
            sy = random.randint(0, w - rw)
            mask = x.new_ones(x.size())
            mask[:, :, sx:sx + rh, sy:sy + rw] = 0
            x = x * mask
        return x

class BatchCrop(nn.Module):
    def __init__(self, ratio):
        super(BatchCrop, self).__init__()
        self.ratio = ratio

    def forward(self, x):
        if self.training:
            h, w = x.size()[-2:]
            rw = int(self.ratio * w)
            start = random.randint(0, h-1)
            if start + rw > h:
                select = list(range(0, start+rw-h)) + list(range(start, h))
            else:
                select = list(range(start, start+rw))
            mask = x.new_zeros(x.size())
            mask[:, :, select, :] = 1
            x = x * mask
        return x