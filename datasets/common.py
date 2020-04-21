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
    
    def sharpen(self, pseudolabel, T=2):
        sign = np.sign(pseudolabel)
        num = np.power(np.abs(pseudolabel), 1/T)
        denom = np.sum(num)
        
        pseudolabel = sign*num/denom
        #print(pseudolabel)
        
        return pseudolabel
    
       
    def confidence_measure_1(self, reconstructions, prev_thresh, label=None):
        pseudolabel = np.mean(reconstructions, axis=0)
        
        pseudolabel = self.sharpen(pseudolabel)
        #print(reconstructions.shape)
        #print("===================================================")
        #out = softmax(reconstructions, axis=1)
        #print(out.shape)
        #variances = np.min(np.var(reconstructions, axis=0))
        #variances = np.max(np.var(reconstructions, axis=0))
        #variances = np.var(reconstructions, axis=0)
        #variances = scipy.stats.entropy(variances)
        
        #print(reconstructions)
        maxes = np.argmax(reconstructions, axis=1)
        #print("=======================================")
        #print(maxes)
        counts = np.bincount(maxes)
        #print(counts)
        count = np.argmax(counts)
        #print(count)
        mask = maxes == count
        #print(mask)
        #mean = np.mean(np.max(reconstructions[mask, :], axis=1))
        variances = np.var(np.max(reconstructions[mask, :], axis=1))#/mean
        #print(reconstructions[mask, :])
        skip = False
        if len(mask) < 10:
            take = False
            skip = True
        else:
            if variances >= prev_thresh:
                take = True
            else:
                take = False
        return take, variances, np.argmax(pseudolabel), pseudolabel, skip
