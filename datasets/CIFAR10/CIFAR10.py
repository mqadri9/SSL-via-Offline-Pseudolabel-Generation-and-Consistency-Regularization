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
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATASET_NAME = "CIFAR10"
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
class SpecLoader():
    
    def __init__(self, path_to_dataset, cfg):
        self.path_to_dataset = path_to_dataset
        self.cfg = cfg
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_train = transforms.Compose([
            #transforms.RandomRotation(40),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
               
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_data_distill = transforms.Compose([
            #TODO: Make more informed decision about new size
            #transforms.RandomRotation(40),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    def prepare_data_single(self):
        print('==> Preparing data..')
        
        self.testset = torchvision.datasets.CIFAR10(root='{}/data'.format(self.path_to_dataset), 
                                                    train=False, 
                                                    download=True, 
                                                    transform=self.transformation)
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg.batch_size, 
                                                      shuffle=False, 
                                                      num_workers=self.cfg.num_workers)
         
        self.trainset = torchvision.datasets.CIFAR10(root='{}/data'.format(self.path_to_dataset), 
                                                     train=True, 
                                                     download=True)
        
        training_split_percentage = int(self.cfg.training_split_percentage)
        total_num_samples = len(self.trainset)
        labelled_upper_bound = int(training_split_percentage*total_num_samples / 100)
        cache_file = '{}_split_{}_percent_labelled.pkl'.format(DATASET_NAME, training_split_percentage)
        cache_file = os.path.join(self.path_to_dataset, "data", cache_file)
        # Check if we already have a pkl file saved for this split. If not regenerate the data
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self.data = pk.load(fid)
            print('Loaded from {}, {} samples are loaded'.format(cache_file, len(self.data)))
        else:
            self.data = []
            stat_portion = np.random.choice(labelled_upper_bound, self.cfg.stats_samples_num, replace=False)
            for index, d in enumerate(self.trainset):
                if index < labelled_upper_bound: 
                    element = {
                        "input": d[0],
                        "label": d[1], 
                        "labelled": "True",
                        "index": index,
                        "cont_label": np.zeros((len(classes),))
                    }
                    if index in stat_portion:
                        element["labelled"] = "stats"
                else:
                    element = {
                        "input": d[0],
                        "label": d[1], 
                        "labelled": "False",
                        "index": index,
                        "cont_label": np.zeros((len(classes),))
                    }
                self.data.append(element)

            self.data = sorted(self.data, key = lambda i: i['labelled'], reverse=True)
            with open(cache_file, 'wb') as fid:
                pk.dump(self.data, fid, pk.HIGHEST_PROTOCOL)
            print('{} samples read wrote {}'.format(len(self.data), cache_file))
        
        data2 = [x for x in self.data if x['labelled']]
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
        
        self.testset = torchvision.datasets.CIFAR10(root='{}/data'.format(self.path_to_dataset), train=False, download=True, transform=self.transformation)
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg.batch_size, 
                                                      shuffle=False, 
                                                      num_workers=self.cfg.num_workers)

        confidence_measure = self.cfg.confidenceMeasure
        confidence_measure = eval("CondifenceMeasure().{}".format(confidence_measure))
        PseudolabelGen = GenPseudolabel(self.cfg, self.transform_train, self.transform_data_distill)          
        data = PseudolabelGen.gen_pseudolabels(model, data_orig, rt_lp, prev_thresh, confidence_measure)
          
        self.trainset_loader = DatasetLoader(data, self.cfg, self.path_to_dataset, self.transform_train, multiplicative=self.cfg.multiplicative)

        self.batch_generator = DataLoader(self.trainset_loader, 
                                          batch_size=self.cfg.batch_size, 
                                          shuffle=True, 
                                          num_workers=self.cfg.num_workers, 
                                          pin_memory=True)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.criterion = SPLoss()   
        self.cross_entropy_test = nn.CrossEntropyLoss()
    

    def gen_pseudolabels2(self, model, data_orig, rt_lp):
        
        print("generated pseudolabels")
        
        self.testset = torchvision.datasets.CIFAR10(root='{}/data'.format(self.path_to_dataset), train=False, download=True, transform=self.transformation)
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg.batch_size, 
                                                      shuffle=False, 
                                                      num_workers=self.cfg.num_workers)

        PseudolabelGen = GenPseudolabel(self.cfg, self.transform_train)  
        data = PseudolabelGen.gen_pseudolabels2(model, data_orig, rt_lp)        
        
        self.trainset_loader = DatasetLoader(data, self.cfg, self.path_to_dataset, self.transform_train, already_transformed=True)
        
        self.batch_generator = DataLoader(self.trainset_loader, 
                                          batch_size=self.cfg.batch_size, 
                                          shuffle=True, 
                                          num_workers=self.cfg.num_workers, 
                                          pin_memory=True)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.criterion = SPLoss()   
        self.cross_entropy_test = nn.CrossEntropyLoss()
