# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from engine.inference import create_supervised_evaluator
from evaluationfile.reid_metric import R1_mAP
from lossfile.triplet_loss import CrossEntropyLabelSmooth
global ITER
ITER = 0

def create_supervised_trainer(model, optimizer, loss_fn, device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        img, target = batch
        optimizer.zero_grad()
        # criterion = nn.CrossEntropyLoss()
        # img = _parse_data(img)
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat, ax = model(img)

        # part = {}
        # # sm = nn.Softmax(dim=1)
        # num_part = 3
        #
        # for i in range(num_part):
        #     part[i] = f[i]
        #
        # # p_score = sm(part[0]) + sm(part[1]) + sm(part[2])
        # # _, preds = torch.max(p_score.data, 1)
        #
        # loss1 = criterion(part[0], target)
        # for i in range(num_part - 1):
        #     loss1 += criterion(part[i + 1], target)

        loss = loss_fn(score, feat,ax, target)
        # loss = loss1 + loss
        loss.backward()
        optimizer.step()
        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)

def do_train(cfg, model, train_loader, val_loader, optimizer,
            scheduler, loss_fn, num_query,  start_epoch):

    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")

    trainer = create_supervised_trainer(model, optimizer,
                                        loss_fn, device=device
                                        )
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query,
                                            max_rank=50,
                                            feat_norm=cfg.TEST.FEAT_NORM)},
                                            device=device
                                            )
    checkpointer = ModelCheckpoint(output_dir,
                                   cfg.MODEL.NAME,
                                   checkpoint_period, n_saved=10,
                                   require_empty=False
                                   )  # 可用于定期将模型保存到磁盘，
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer,
                              {'model': model.state_dict(),
                               'optimizer': optimizer.state_dict()}
                              )
    timer.attach(trainer, start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED,
                 step=Events.ITERATION_COMPLETED
                 )

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, "
                        "Base Lr: {:.2e}".format(engine.state.epoch, ITER,
                                                 len(train_loader),
                                                 engine.state.metrics['avg_loss'],
                                                 engine.state.metrics['avg_acc'],
                                                 scheduler.get_lr()[0]
                                                 )
                        )
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}['
                    'samples/s]'.format(engine.state.epoch,
                                        timer.value() * timer.step_count,
                                        train_loader.batch_size / timer.value())
                    )
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)


