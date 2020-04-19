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
import scipy.stats

def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


class CondifenceMeasure():
    def __init__(self):
        pass
   
    def confidence_measure_1(self, reconstructions, label=None):
        #print(reconstructions.shape)
        #print("===================================================")
        out = softmax(reconstructions, axis=1)
        #print(out.shape)
        variances = np.min(np.var(out, axis=0))
        #print("variances")
        #print(variances)
        #print(reconstructions)
        
        pseudolabel = np.mean(reconstructions, axis=0)
        #sys.exit()
        #print(pseudolabel)
        #pseudolabel = reconstructions[0]
        #print(pseudolabel.shape)
        #print("pseudolabel")
        #print(np.argmax(pseudolabel))
        #print("label")
        #print(label)
        if variances < 0.01:
            take = True
        else:
            take = False
        #sys.exit()
        return take, variances, np.argmax(pseudolabel), pseudolabel