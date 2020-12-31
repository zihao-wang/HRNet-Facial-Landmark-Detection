# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import pandas as pd
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.datasets import get_dataset
from lib.core import function
from lib.utils import utils
CUDA_VISIBLE_DEVICES = 0,

def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():

    args = parse_args()

    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'train')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    gpus = list(config.GPUS)

    dataset_type = get_dataset(config)
    train_data = dataset_type(config, is_train=True)
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY)


    val_data = dataset_type(config, is_train=False)
    val_loader = DataLoader(
        dataset=val_data,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    # config.MODEL.NUM_JOINTS = train_data.get_num_points()
    model = models.get_face_alignment_net(config)

    # copy model files
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    model = nn.DataParallel(model, device_ids=gpus).cuda()

    # loss
    criterion = torch.nn.MSELoss(size_average=True).cuda()
    optimizer = utils.get_optimizer(config, model)

    best_nme = 100
    last_epoch = config.TRAIN.BEGIN_EPOCH

    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP,
            config.TRAIN.LR_FACTOR, last_epoch-1
        )

    if config.TRAIN.RESUME:
        model_state_file = os.path.join(final_output_dir,
                                        'final.pth')
        if os.path.islink(model_state_file):
            checkpoint = torch.load(model_state_file)
            last_epoch = checkpoint['epoch']
            best_nme = checkpoint['best_nme']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint (epoch {})"
                  .format(checkpoint['epoch']))
        else:
            print("=> no checkpoint found")
    loss = []
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        losses = function.train(config, train_loader, model, criterion,
                       optimizer, epoch, writer_dict)
        loss.append(losses)
        lr_scheduler.step()

        # evaluate
        nme, predictions = function.validate(config, val_loader, model,
                                             criterion, epoch, writer_dict)

        is_best = nme < best_nme
        best_nme = min(nme, best_nme)

        logger.info('=> saving checkpoint to {}'.format(final_output_dir))
        print("best:", is_best)
        utils.save_checkpoint(
            {"state_dict": model,
             "epoch": epoch + 1,
             "best_nme": best_nme,
             "optimizer": optimizer.state_dict(),
             }, predictions, is_best, final_output_dir, 'checkpoint_{}.pth'.format(epoch))
        if is_best:
            for i in range(len(predictions)):
                afile = val_data.annotation_files[i]
                new_afile = '{}.{}.txt'.format(afile, os.path.basename(args.cfg).split('.')[0])
                with open(new_afile, 'wt') as f:
                    pts = predictions[i].cpu().numpy()
                    for j in range(len(pts)):
                        f.write("{},{}\n".format(pts[j][1]/val_data.factor[1], 
                                                 pts[j][0]/val_data.factor[0]))

    pd.DataFrame(data=loss).to_csv('loss2.csv')
    final_model_state_file = os.path.join(final_output_dir,
                                          'final_state.pth')
    logger.info('saving final model state to {}'.format(
        final_model_state_file))
    torch.save(model.module.state_dict(), final_model_state_file)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()










