# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import tqdm
import os

import torch
import numpy as np

from .evaluation import decode_preds, compute_nme

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train(config, train_loader, model, critertion, optimizer,
          epoch, writer_dict):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    nme_count = 0
    nme_batch_sum = 0

    end = time.time()
    for i, (inp, target, meta) in tqdm.tqdm(enumerate(train_loader)):
        # measure data time
        data_time.update(time.time()-end)

        # compute the output
        output = model(inp)
        target = target.cuda(non_blocking=True)
        loss = critertion(output, target)

        # debug single image prediction
        # _, num_pts, W, H = target.shape
        # os.makedirs("single_sample_debug", exist_ok=True)
        # print("debugging single image", epoch)
        # output_arr = output.cpu().detach().numpy()
        # target_arr = target.cpu().numpy()
        # img_arr = inp.cpu().numpy()
        # for i_pts in range(num_pts):
        #     fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3)
        #     ax0.imshow(output_arr[0, i_pts, :, :])
        #     ax0.set_title("output at epoch {}, for point {}, max={}, min={}".format(epoch, i_pts, output_arr.max(), output_arr.min()))
        #     ax1.imshow(img_arr[0, 1, :, :])
        #     # x, y = meta['pts'][0, i].cpu().numpy().tolist()
        #     # ax1.scatter([x], [y], 'r')
        #     ax2.imshow(target_arr[0, i_pts, :, :])
        #     ax2.set_title("target at epoch {}, for point {}, max={}, min={}".format(epoch, i_pts, target_arr.max(), target_arr.min()))
        #     fig.savefig("single_sample_debug/pts#{}@epoch{}".format(i_pts, epoch))

        # wp = [10, 11, 12, 13, 26,27,28,29,30,31]
        # for i in wp:
        #     for j in range(len(target)):
        #         loss = loss + critertion(output[j][i], target[j][i])
        # loss = critertion(output, target) + pow((target[i][10][0] - inp[i][10][0]) , 2) + pow((target[i][10][1] - inp[i][10][1]), 2)

        # NME
        score_map = output.data.cpu()
        preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])

        nme_batch = compute_nme(preds, meta)

        nme_batch_sum = nme_batch_sum + np.sum(nme_batch)
        nme_count = nme_count + preds.size(0)

        # optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), inp.size(0))

        batch_time.update(time.time()-end)
        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      speed=inp.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            logger.info(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.val, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1

        end = time.time()
    nme = nme_batch_sum / nme_count
    msg = 'Train Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f}'\
        .format(epoch, batch_time.avg, losses.avg, nme)
    logger.info(msg)
    return losses.avg


def validate(config, val_loader, model, criterion, epoch, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = AverageMeter()

    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(val_loader.dataset), num_classes, 2))

    model.eval()

    nme_count = 0
    nme_batch_sum = 0
    count_failure_1 = 0
    count_failure_3 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(val_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            target = target.cuda(non_blocking=True)

            score_map = output.data.cpu()
            # loss
            loss = criterion(output, target)

            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            nme_temp = compute_nme(preds, meta)
            # Failure Rate under different threshold
            failure_1 = (nme_temp > 1).sum()
            failure_3 = (nme_temp > 3).sum()
            count_failure_1 += failure_1
            count_failure_3 += failure_3

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            losses.update(loss.item(), inp.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_1_rate = count_failure_1 / nme_count
    failure_3_rate = count_failure_3 / nme_count

    msg = 'Test Epoch {} time:{:.4f} loss:{:.4f} nme:{:.4f} [1px]:{:.4f} ' \
          '[3px]:{:.4f}'.format(epoch, batch_time.avg, losses.avg, nme,
                                failure_1_rate, failure_3_rate)
    logger.info(msg)

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_loss', losses.avg, global_steps)
        writer.add_scalar('valid_nme', nme, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return nme, predictions


def inference(config, data_loader, model):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    num_classes = config.MODEL.NUM_JOINTS
    predictions = torch.zeros((len(data_loader.dataset), num_classes, 2))
    model.eval()
    nme_count = 0
    nme_batch_sum = 0
    count_failure_008 = 0
    count_failure_010 = 0
    end = time.time()

    with torch.no_grad():
        for i, (inp, target, meta) in enumerate(data_loader):
            data_time.update(time.time() - end)
            output = model(inp)
            score_map = output.data.cpu()
            preds = decode_preds(score_map, meta['center'], meta['scale'], [64, 64])
            # NME
            nme_temp = compute_nme(preds, meta)

            failure_008 = (nme_temp > 0.08).sum()
            failure_010 = (nme_temp > 0.10).sum()
            count_failure_008 += failure_008
            count_failure_010 += failure_010

            nme_batch_sum += np.sum(nme_temp)
            nme_count = nme_count + preds.size(0)
            for n in range(score_map.size(0)):
                predictions[meta['index'][n], :, :] = preds[n, :, :]

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

    nme = nme_batch_sum / nme_count
    failure_008_rate = count_failure_008 / nme_count
    failure_010_rate = count_failure_010 / nme_count

    msg = 'Test Results time:{:.4f} loss:{:.4f} nme:{:.4f} [008]:{:.4f} ' \
          '[010]:{:.4f}'.format(batch_time.avg, losses.avg, nme,
                                failure_008_rate, failure_010_rate)
    logger.info(msg)

    return nme, predictions



