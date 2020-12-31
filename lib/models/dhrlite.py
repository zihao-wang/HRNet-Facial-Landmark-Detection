from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hrnet import blocks_dict, HighResolutionModule


class DentalHighResolutionLiteNet(nn.Module):
    def __init__(self, config, **kwargs):
        pass