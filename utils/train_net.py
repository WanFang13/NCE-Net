
import os
import torch
import numpy as np
import time
from utils.test_net import test
from utils.util import *


def train_epoch(
        cfg, logger, models, criterion,
        optimizers, dataloaders,
        epoch, trial, cycle
):

    models['backbone'].train()

    loss = 0
    for data in dataloaders['train']:
        inputs  = data[0].cuda()
        labels2 = data[1].cuda()
        labels  = get_one_hot_label(labels2, cfg.DATASET.NUM_CLASS)
        cfg.global_iter.update_vars()

        optimizers['backbone'].zero_grad()

        results = models['backbone'](inputs)
        target_loss = criterion(results[0], labels)

        target_loss = target_loss

        loss = target_loss

        loss.backward()
        optimizers['backbone'].step()

    # print
    if (epoch+1) % 10 == 0 or epoch == 0:
        logger.info("Trial {}, Cycle {}, Epoch {}, Loss_target {}".format(
            trial + 1, cycle + 1, epoch + 1,
            round(loss.item(), 4))
        )

#
def train(
        cfg, logger, models, criterion,
        optimizers, schedulers, dataloaders,
        num_epochs, checkpoint_dir,
        tri, cyc
):
    logger.info('>> Train a Model.')
    cfg.global_iter.reset_vars()

    for epoch in range(num_epochs):
        schedulers['backbone'].step()

        train_epoch(
            cfg,
            logger,
            models,
            criterion,
            optimizers,
            dataloaders,
            epoch, tri, cyc
        )

    logger.info('>> Finished.')






