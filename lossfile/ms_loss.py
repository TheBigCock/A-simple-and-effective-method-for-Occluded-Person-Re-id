#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author:1111
# datetime:2019/12/11 17:02
# software: PyCharm
import torch
from torch import nn


def normalize_ms(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist_ms(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist


class MultiSimilarityLoss(nn.Module):
    def __init__(self, cfg):
        super(MultiSimilarityLoss, self).__init__()
        self.thresh = 1
        self.margin = 0.1

        self.scale_pos = cfg.SOLVER.MULTI_SIMILARITY_LOSS.SCALE_POS
        self.scale_neg = cfg.SOLVER.MULTI_SIMILARITY_LOSS.SCALE_NEG

        self.m = cfg.SOLVER.MARGIN_RANK
        self.alpha = cfg.SOLVER.ALPHA

    def forward(self, feats, labels, normalize_feature=True):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        batch_size = feats.size(0)
        if normalize_feature:
            feats = normalize_ms(feats, axis=-1)
        dist_mat = euclidean_dist_ms(feats, feats)
        # torch.matmul

        epsilon = 1e-5
        loss = list()

        for i in range(batch_size):
            pos_pair_ = dist_mat[i][labels == labels[i]]
            pos_pair_ = pos_pair_[pos_pair_ < 1 - epsilon]
            neg_pair_ = dist_mat[i][labels != labels[i]]

            # pos_pair = torch.clamp(torch.add(pos_pair_, self.m - self.alpha), min=0.0)

            # neg_pair = torch.lt(neg_pair_, self.alpha)
            # neg_pair = neg_pair_[neg_pair]

            neg_pair = neg_pair_[neg_pair_ + self.margin > min(pos_pair_)]
            pos_pair = pos_pair_[pos_pair_ - self.margin < max(neg_pair_)]
            ap_pos_num = pos_pair.size(0) + 1e-5
            if len(neg_pair) < 1 or len(pos_pair) < 1:
                continue

            # weighting step
            # pos_loss = 1.0 / self.scale_pos * torch.log(
            #     1 + torch.sum(torch.exp(-self.scale_pos * (pos_pair - self.thresh))))
            # neg_loss = 1.0 / self.scale_neg * torch.log(
            #     1 + torch.sum(torch.exp(self.scale_neg * (neg_pair - self.thresh))))

            # # weighting step
            # pos_loss = 1.0 / self.scale_pos * torch.log(
            #     1 + torch.sum(torch.exp(self.scale_pos * (pos_pair + (self.m - self.alpha)))))
            neg_loss = 1.0 / self.scale_neg * torch.log(
                1 + torch.sum(torch.exp(-self.scale_neg * (neg_pair - self.alpha))))
            pos_loss = 1.0 / ap_pos_num *torch.log(
                1 + torch.sum(torch.exp(self.scale_pos * (pos_pair + (self.m - self.alpha)))))
            loss.append(pos_loss + neg_loss)

        if len(loss) == 0:
            return torch.zeros([], requires_grad=True)

        loss = sum(loss) / batch_size
        return loss











