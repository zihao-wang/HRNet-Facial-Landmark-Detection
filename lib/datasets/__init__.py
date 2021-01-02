# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng(tianhengcheng@gmail.com)
# ------------------------------------------------------------------------------

from .dental import Dental

__all__ = ['Dental', 'get_dataset']

def get_dataset(config):

    if config.DATASET.DATASET == 'Dental':
        return Dental
    else:
        raise NotImplementedError()

