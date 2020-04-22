import numpy as np
import os, sys
import config as config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import pickle as pk
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
from common import ConfidenceMeasure, DatasetLoader, DatasetLoader2, MiniDatasetLoader, MiniAugmentedDatasetLoader
from common import SPLoss, CustomCrossEntropyLoss, GenPseudolabel
from torch.utils.data import DataLoader
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

DATASET_NAME = "STL10"
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')


class SpecLoader():
    
    def __init__(self, path_to_dataset, cfg):
        self.path_to_dataset = path_to_dataset
        self.cfg = cfg
        self.transform_train=transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.transform_data_distill=transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            #transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
                
        self.transform_test = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.transformation = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
           
    def prepare_data_single(self):
        print('==> Preparing data..')

        self.testset = torchvision.datasets.STL10(root='{}/data'.format(self.path_to_dataset), 
                                                  split="test", 
                                                  download=True, 
                                                  transform=self.transform_test)
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg.batch_size, 
                                                      shuffle=False, 
                                                      num_workers=self.cfg.num_workers)
        
        self.ds = torchvision.datasets.STL10(root='{}/data'.format(self.path_to_dataset), 
                                             split="train+unlabeled",  
                                             download=True)

        cache_file = '{}_stats_{}.pkl'.format(DATASET_NAME, self.cfg.stats_samples_num)
        cache_file = os.path.join(self.path_to_dataset, "data", cache_file)
        # Check if we already have a pkl file saved for this split. If not regenerate the data
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self.data = pk.load(fid)
            print('Loaded from {}, {} samples are loaded'.format(cache_file, len(self.data)))
        else:    
            self.data = []
            stat_portion = np.random.choice(4999, self.cfg.stats_samples_num, replace=False)
            for index, d in enumerate(self.ds):
                element = {
                        "input": d[0],
                        "label": d[1], 
                        "index": index,
                        "cont_label": np.zeros((len(classes),))
                     }
                if d[1] == -1:
                    element["labelled"] = "False"
                elif index in stat_portion:
                    element["labelled"] = "stats"
                else:
                    element["labelled"] = "True"
                self.data.append(element)        
            with open(cache_file, 'wb') as fid:
                pk.dump(self.data, fid, pk.HIGHEST_PROTOCOL)
            print('{} samples read wrote {}'.format(len(self.data), cache_file))
          
        print("Number of training samples (labelled + unlabelled) {}".format(len(self.data)))
        data2 = [x for x in self.data if x['labelled']=="True"]
        self.trainset_loader = DatasetLoader(data2, self.cfg, self.path_to_dataset, self.transform_train)

        self.batch_generator = DataLoader(self.trainset_loader, 
                                          batch_size=self.cfg.batch_size, 
                                          shuffle=True, 
                                          num_workers=self.cfg.num_workers, 
                                          pin_memory=True)
                
        print("batch_size = {}".format(self.cfg.batch_size))
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.criterion = CustomCrossEntropyLoss()
        self.cross_entropy_test = nn.CrossEntropyLoss()   
        
    def gen_pseudolabels(self, model, data_orig, rt_lp, prev_thresh=0):
        
        print("generated pseudolabels")
        
        self.testset = torchvision.datasets.STL10(root='{}/data'.format(self.path_to_dataset), 
                                                  split="test",  
                                                  download=True, 
                                                  transform=self.transformation)
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg.batch_size, 
                                                      shuffle=False, 
                                                      num_workers=self.cfg.num_workers)

        confidence_measure = self.cfg.confidenceMeasure
        confidence_measure = eval("ConfidenceMeasure().{}".format(confidence_measure))
        print("=================")
        print(confidence_measure)
        PseudolabelGen = GenPseudolabel(self.cfg, self.transform_train, self.transform_data_distill)          
        data = PseudolabelGen.gen_pseudolabels(model, data_orig, rt_lp, prev_thresh, confidence_measure)
        
        self.trainset_loader = DatasetLoader2(data, self.cfg, self.path_to_dataset, self.transform_train, multiplicative=self.cfg.multiplicative)

        self.batch_generator = DataLoader(self.trainset_loader, 
                                          batch_size=self.cfg.batch_size, 
                                          shuffle=True, 
                                          num_workers=self.cfg.num_workers, 
                                          pin_memory=True)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.criterion = SPLoss()   
        self.cross_entropy_test = nn.CrossEntropyLoss()
        return prev_thresh

    def gen_pseudolabels2(self, model, data_orig, rt_lp):
        
        print("generated pseudolabels")
        
        self.testset = torchvision.datasets.STL10(root='{}/data'.format(self.path_to_dataset), split="test", download=True, transform=self.transformation)
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg.batch_size, 
                                                      shuffle=False, 
                                                      num_workers=self.cfg.num_workers)    
        
        PseudolabelGen = GenPseudolabel(self.cfg, self.transform_train, None)  
        data = PseudolabelGen.gen_pseudolabels2(model, data_orig, rt_lp)
        
        self.trainset_loader = DatasetLoader2(data, self.cfg, self.path_to_dataset, self.transform_train, already_transformed=True)
        
        self.batch_generator = DataLoader(self.trainset_loader, 
                                          batch_size=self.cfg.batch_size, 
                                          shuffle=True, 
                                          num_workers=self.cfg.num_workers, 
                                          pin_memory=True)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.criterion = SPLoss()   
        self.cross_entropy_test = nn.CrossEntropyLoss()

    
    
    
    
    
    
    
    
    