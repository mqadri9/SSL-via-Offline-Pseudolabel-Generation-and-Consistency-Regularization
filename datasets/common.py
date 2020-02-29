import numpy as np
import os, sys
import config as config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms


class CondifenceMeasure():
    def __init__(self):
        pass
    
    def confidence_measure_1(self, reconstructions):
        pass