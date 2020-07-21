# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys

import torch
import yaml
from torch.backends import cudnn

sys.path.append('.')
from config import cfg
from engine.trainer import do_train
from evaluationfile import WarmupMultiStepLR
from data.data_tool.collate_batch import train_collate_fn, val_collate_fn
from lossfile import make_loss
from evaluationfile import make_optimizer
from torchvision import transforms
from torch.utils.data import DataLoader
from logfile.logger import setup_logger
from modeling.baseline import Baseline
from data.data_reader import init_dataset
from data.data_tool import ImageDataset, RandomIdentitySampler, RandomErasing, Cutout

def get_data(cfg):
    normalizer = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    transform_train_list = [
        transforms.Resize(cfg.INPUT.SIZE_TRAIN),
        transforms.Pad(10),
        transforms.RandomHorizontalFlip(p=cfg.INPUT.PROB),
        transforms.RandomCrop(cfg.INPUT.SIZE_TRAIN),
        transforms.ToTensor(),
        normalizer
        ]
    if cfg.INPUT.RE_PROB > 0 :
        transform_train_list = transform_train_list + [RandomErasing(probability=cfg.INPUT.RE_PROB, mean=[0.485, 0.456, 0.406])]
    if cfg.INPUT.CUT_PROB > 0:
        transform_train_list = transform_train_list + [
            Cutout(probability=cfg.INPUT.CUT_PROB, size=64,mean=[0.0, 0.0, 0.0])]
    transform_val_list = [
        transforms.Resize(cfg.INPUT.SIZE_TEST), #Image.BICUBIC
        transforms.ToTensor(),
        normalizer
        ]
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    train_transforms = transforms.Compose(transform_train_list)  # 训练数据
    val_transforms = transforms.Compose(transform_val_list)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)

    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_dataloaders = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=train_collate_fn
        )
    else:
        train_dataloaders = DataLoader(
            train_set,
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train,
                                          cfg.SOLVER.IMS_PER_BATCH,
                                          cfg.DATALOADER.NUM_INSTANCE
                                          ),
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_dataloaders = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False,
        num_workers=cfg.DATALOADER.NUM_WORKERS,
        collate_fn=val_collate_fn
    )
    return train_dataloaders, val_dataloaders, len(dataset.query), num_classes

def train(cfg):
    # prepare dataset 训练集，验证集，验证集大小，行人类别数量
    train_loader, val_loader, num_query, num_classes = get_data(cfg)

    # prepare model
    model = Baseline(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH,
                      cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE,cfg.MODEL.BREACH)
    print('Train with the loss type is', cfg.MODEL.METRIC_LOSS_TYPE) # 损失函数为ranked_loss
    optimizer = make_optimizer(cfg, model)
    loss_func = make_loss(cfg, num_classes)

    # Add for using self trained model
    if cfg.MODEL.PRETRAIN_CHOICE == 'self':
            start_epoch = eval(cfg.MODEL.PRETRAIN_PATH.split('/')[-1].split('.')[0].split('_')[-1])
            print('Start epoch:', start_epoch)
            path_to_optimizer = cfg.MODEL.PRETRAIN_PATH.replace('model', 'optimizer')
            print('Path to the checkpoint of optimizer:', path_to_optimizer)
            model.load_state_dict(torch.load(cfg.MODEL.PRETRAIN_PATH))
            optimizer.load_state_dict(torch.load(path_to_optimizer))
            scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS,
                                          cfg.SOLVER.GAMMA,
                                          cfg.SOLVER.WARMUP_FACTOR,
                                          cfg.SOLVER.WARMUP_ITERS,
                                          cfg.SOLVER.WARMUP_METHOD,
                                          start_epoch
                                          )
    elif cfg.MODEL.PRETRAIN_CHOICE == 'imagenet':
        start_epoch = 0
        scheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA,
                                      cfg.SOLVER.WARMUP_FACTOR,
                                      cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD
                                      )
    else:
        print('Only support pretrain_choice for imagenet and self, but got {}'.format(cfg.MODEL.PRETRAIN_CHOICE))

    do_train(cfg, model, train_loader,
             val_loader,
             optimizer, scheduler,
             loss_func, num_query,
             start_epoch
             )

def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("reid_baseline", output_dir, 0)
    logger.info("Using {} GPUS".format(num_gpus))
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True
    train(cfg)


if __name__ == '__main__':
    main()
