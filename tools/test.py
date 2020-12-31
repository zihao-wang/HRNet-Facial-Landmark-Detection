# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Created by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

import os
import pprint
import argparse

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import lib.models as models
from lib.config import config, update_config
from lib.utils import utils
from lib.datasets import get_dataset
from lib.core import function
import numpy as np
import pandas as pd


def parse_args():

    parser = argparse.ArgumentParser(description='Train Face Alignment')

    parser.add_argument('--cfg', help='experiment configuration filename',
                        required=True, type=str)
    parser.add_argument('--model-file', help='model parameters', required=True, type=str)

    args = parser.parse_args()
    update_config(config, args)
    return args


def main():
    #
    args = parse_args()
    #
    logger, final_output_dir, tb_log_dir = \
        utils.create_logger(config, args.cfg, 'test')
    #
    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))
    #
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.determinstic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    #
    config.defrost()
    config.MODEL.INIT_WEIGHTS = False
    config.freeze()
    model = models.get_face_alignment_net(config)
    #
    gpus = list(config.GPUS)
    model = nn.DataParallel(model, device_ids=gpus).cuda()
    #
    # # load model
    state_dict = torch.load(args.model_file)
    if 'state_dict' in state_dict.keys():
        state_dict = state_dict['state_dict']
        model.load_state_dict(state_dict)
    else:
        model.module.load_state_dict(state_dict)
    #
    dataset_type = get_dataset(config)

    test_loader = DataLoader(
        dataset=dataset_type(config,
                             is_train=False),
        batch_size=config.TEST.BATCH_SIZE_PER_GPU*len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY
    )

    nme, predictions = function.inference(config, test_loader, model)
    torch.save(predictions, os.path.join(final_output_dir, 'predictions.pth'))
    target = test_loader.dataset.load_all_pts()
    pred = 16 * predictions
    l = len(pred)
    res = 0.0
    res_tmp = [ 0.0 for i in range(config.MODEL.NUM_JOINTS)]


    res_tmp = np.array(res_tmp)
    res_temp_x = target - pred
    res_temp_x = res_temp_x[:, :, 0]
    res_temp_y = target - pred
    res_temp_y = res_temp_y[:, :, 1]

    # csv_file_test_x = pd.DataFrame(np.transpose(np.array(pred[:, :, 0])), columns=test_loader.dataset.annotation_files)
    # csv_file_test_y = pd.DataFrame(np.transpose(np.array(pred[:, :, 1])), columns=test_loader.dataset.annotation_files)
    # csv_file_target_x = pd.DataFrame(np.transpose(np.array(target[:, :, 0])), columns=test_loader.dataset.annotation_files)
    # csv_file_target_y = pd.DataFrame(np.transpose(np.array(target[:, :, 1])), columns=test_loader.dataset.annotation_files)

    for i in range(l):
        trans = np.sqrt(pow(target[i][0][0] - target[i][1][0], 2) + pow(target[i][0][1] - target[i][1][1], 2)) / 30.0
        res_temp_x[i] = res_temp_x[i] / trans
        res_temp_y[i] = res_temp_y[i] / trans
        for j in range(len(target[i])):
            dist = np.sqrt(np.power((target[i][j][0] - pred[i][j][0]), 2) + np.power((target[i][j][1] - pred[i][j][1]), 2)) / trans
            res += dist
            res_tmp[j] += dist
    res_t = np.sqrt(res_temp_x * res_temp_x + res_temp_y * res_temp_y)
    # pd.DataFrame(data=res_temp_x.data.value).to_csv('res_x')
    # pd.DataFrame(data=res_temp_y.data.value).to_csv('res_y')
    # pd.DataFrame(data=res_t.data.value).to_csv('res_t')
    res_tmp /= np.float(len(pred))
    print(res_tmp)
    print(np.mean(res_tmp))
    res /= (len(pred) * len(pred[0]))
    print(res)

    # csv_file_target_x.to_csv('target_data_x.csv')
    # csv_file_target_y.to_csv('target_data_y.csv')
    # csv_file_test_x.to_csv('test_data_x.csv')
    # csv_file_test_y.to_csv('test_data_y.csv')

if __name__ == '__main__':
    main()

