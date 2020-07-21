# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""
import logging

import os
import cv2
import torch
import torch.nn as nn
from ignite.engine import Engine
import numpy as np

from evaluationfile.reid_metric import R1_mAP, R1_mAP_reranking
from os import path

def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            output, pids, camids = batch
            output = output.to(device) if torch.cuda.device_count() >= 1 else output
            feat = model(output)

            # c_att = att.data.cpu()
            # c_att = c_att.numpy()
            # d_inputs = output.data.cpu()
            # d_inputs = d_inputs.numpy()
            # count = 0
            # in_b, in_c, in_y, in_x = output.shape
            # for item_img , item_att in zip(d_inputs, c_att):
            #
            #     v_img = ((item_img.transpose((1, 2, 0)) + 0.5 + [0.485, 0.456, 0.406]) * [0.229, 0.224, 0.225]) * 256
            #     v_img = v_img[:, :, ::-1]
            #     resize_att = cv2.resize(item_att[0], (in_x, in_y))
            #     resize_att *= 255.
            #
            #     cv2.imwrite('stock1.png', v_img)
            #     cv2.imwrite('stock2.png', resize_att)
            #     v_img = cv2.imread('stock1.png')
            #     vis_map = cv2.imread('stock2.png', 0)
            #     jet_map = cv2.applyColorMap(vis_map, cv2.COLORMAP_JET)
            #     jet_map = cv2.add(v_img, jet_map)
            #
            #     out_dir = path.join('../output')
            #     if not path.exists(out_dir):
            #         os.mkdir(out_dir)
            #     out_path = path.join(out_dir, 'attention', '{0:06d}.png'.format(count))
            #     cv2.imwrite(out_path, jet_map)
            #     out_path = path.join(out_dir, 'raw', '{0:06d}.png'.format(count))
            #     cv2.imwrite(out_path, v_img)
            #
            #     count += 1

            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def inference(
        cfg,
        model,
        val_loader,
        num_query
):
    device = cfg.MODEL.DEVICE

    logger = logging.getLogger("reid_baseline.inference")
    logger.info("Enter inferencing")
    if cfg.TEST.RE_RANKING == 'no':
        print("Create evaluator")
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50,
                                                                                 feat_norm=cfg.TEST.FEAT_NORM)},
                                                device=device
                                                )
    elif cfg.TEST.RE_RANKING == 'yes':
        print("Create evaluator for reranking")
        evaluator = create_supervised_evaluator(
            model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=50,
                                                       feat_norm=cfg.TEST.FEAT_NORM)
                            },
            device=device)
    else:
        print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))

    evaluator.run(val_loader)
    cmc, mAP = evaluator.state.metrics['r1_mAP']
    logger.info('Validation Results')
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
