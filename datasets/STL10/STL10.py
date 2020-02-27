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

class SpecLoader():
    
    def __init__(self):
        pass
    
    def prepare_data_single(self, path_to_dataset, cfg):
        print('==> Preparing data..')
        transform_train=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        transform_test = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        self.trainset = torchvision.datasets.STL10(root='{}/data'.format(path_to_dataset), split="train", download=True, transform=transform_train)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
        
        self.testset = torchvision.datasets.STL10(root='{}/data'.format(path_to_dataset), split="test", download=True, transform=transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
                
        self.criterion = nn.CrossEntropyLoss()
        
    def gen_pseudolabels(self):
        pass