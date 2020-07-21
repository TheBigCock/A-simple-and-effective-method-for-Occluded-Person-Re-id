# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch.nn as nn
import torch.nn.functional as F

from .triplet_loss import TripletLoss, CrossEntropyLabelSmooth
from .center_loss import CenterLoss
from .cluster_loss import ClusterLoss
from .reanked_loss import RankedLoss
from .ms_loss import MultiSimilarityLoss

def make_loss(cfg, num_classes):
    sampler = cfg.DATALOADER.SAMPLER
    if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)   # Margin of triplet loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'ranked_loss':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # Margin of triplet loss
        ranked_loss = RankedLoss(cfg.SOLVER.MARGIN_RANK, cfg.SOLVER.ALPHA, cfg.SOLVER.TVAL)  # ranked_loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'ms_loss':
        ms_loss = MultiSimilarityLoss(cfg)  # ms_loss
    elif cfg.MODEL.METRIC_LOSS_TYPE == 'triplet_cluster':
        triplet = TripletLoss(cfg.SOLVER.MARGIN)  # triplet loss
        cluster = ClusterLoss(cfg.SOLVER.CLUSTER_MARGIN, True, True, cfg.SOLVER.IMS_PER_BATCH // cfg.DATALOADER.NUM_INSTANCE, cfg.DATALOADER.NUM_INSTANCE)
    else:
        print('expected METRIC_LOSS_TYPE should be triplet'
              'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))

    if cfg.MODEL.IF_LABELSMOOTH == 'on':
        xent = CrossEntropyLabelSmooth(num_classes=num_classes)
        print("label smooth on, numclasses:", num_classes)

    if sampler == 'softmax':
        def loss_func(score, feat,  target):
            return F.cross_entropy(score, target)
    elif sampler == 'ranked_loss':
        def loss_func(score, feat,  target):
            return ranked_loss(feat, target)[0]
    elif sampler == 'ms_loss':
        def loss_func(score, feat,  target):
            return ms_loss(feat, target)[0]
    elif sampler == 'triplet':
        def loss_func(score, feat, ax, target):
            return triplet(feat, target)[0]
    elif sampler == 'softmax_triplet' or sampler == 'softmax_rank'or sampler == 'softmax_ms':
        def loss_func(score, feat, ax, target):
            if cfg.MODEL.METRIC_LOSS_TYPE == 'triplet':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + triplet(feat, target)[0]
                else:
                    return F.cross_entropy(score, target) + triplet(feat, target)[0]
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'ranked_loss':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    # part = {}
                    # # sm = nn.Softmax(dim=1)
                    # num_part = 2
                    #
                    # for i in range(num_part):
                    #     part[i] = f[i]
                    #
                    # # p_score = sm(part[0]) + sm(part[1]) + sm(part[2])
                    # # _, preds = torch.max(p_score.data, 1)
                    #
                    # loss1 = xent(part[0], target)
                    # for i in range(num_part - 1):
                    #     loss1 += xent(part[i + 1], target)

                    return xent(score, target) \
                            + cfg.SOLVER.WEIGHT * ranked_loss(feat, target) \
                            + cfg.SOLVER.ATTENTION * xent(ax, target)
                                # + loss1
                            # + cfg.SOLVER.ATTENTION * xent(ax, target) \
                            # + cfg.SOLVER.WEIGHT * ranked_loss(f, target)

                else:
                    return F.cross_entropy(score, target) \
                           + ranked_loss(feat, target)[0]
            elif cfg.MODEL.METRIC_LOSS_TYPE == 'ms_loss':
                if cfg.MODEL.IF_LABELSMOOTH == 'on':
                    return xent(score, target) + \
                           cfg.SOLVER.WEIGHT * ms_loss(feat, target)
                else:
                    return F.cross_entropy(score, target) + ms_loss(feat, target)[0]
            else:
                print('expected METRIC_LOSS_TYPE should be triplet'
                      'but got {}'.format(cfg.MODEL.METRIC_LOSS_TYPE))
    else:
        print('expected sampler should be softmax, triplet or softmax_triplet, '
              'but got {}'.format(cfg.DATALOADER.SAMPLER))
    return loss_func


