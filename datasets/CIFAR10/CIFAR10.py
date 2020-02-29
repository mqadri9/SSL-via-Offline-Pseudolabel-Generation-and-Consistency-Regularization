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
from common import CondifenceMeasure
from torch.utils.data import DataLoader
import random

DATASET_NAME = "CIFAR10"
class DatasetLoader(Dataset):
    def __init__(self, dataset, cfg, path_to_dataset, transformation):
        print("Inside custom DatasetLoader")
        self.total_num_examples = 0
        self.cfg = cfg
        self.transforms = transformation
        training_split_percentage = self.cfg.training_split_percentage
        cache_file = '{}_split_{}_percent_labelled.pkl'.format(DATASET_NAME, training_split_percentage)
        cache_file = os.path.join(path_to_dataset, "data", cache_file)
        # Check if we already have a pkl file saved for this split. If not regenerate the data
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self.data = pk.load(fid)
            print('Loaded from {}, {} samples are loaded'.format(cache_file, len(self.data)))
        else:
            self.data = []
            total_num_samples = len(dataset)
            for index, d in enumerate(dataset):
                if index < int(training_split_percentage*total_num_samples / 100): 
                    element = {
                        "input": d[0],
                        "label": d[1], 
                        "labelled": True,
                        "index": index
                    }
                else:
                    element = {
                        "input": d[0],
                        "label": d[1], 
                        "labelled": False,
                        "index": index
                    }
                self.data.append(element)
            self.data = sorted(self.data, key = lambda i: i['labelled'], reverse=True)
            with open(cache_file, 'wb') as fid:
                pk.dump(self.data, fid, pk.HIGHEST_PROTOCOL)
            print('{} samples read wrote {}'.format(len(self.data), cache_file))
            
        condition_labelled = {"labelled": True}
        self.data_labelled = list(filter(lambda item: all((item[k]==v for (k,v) in condition_labelled.items())), self.data))
        condition_unlabelled = {"labelled": False}
        self.data_unlabelled = list(filter(lambda item: all((item[k]==v for (k,v) in condition_unlabelled.items())), self.data))
        self.num_unlabelled = len(self.data_unlabelled)
        self.num_labelled = len(self.data_labelled)
        
    def __getitem__(self, index):
        labelled_selection_prob = self.cfg.labelled_selection_prob/100
        if self.cfg.training_split_percentage == 100:
            idx = random.randint(0, self.num_labelled-1)
            data =  self.data_labelled[idx]
        elif random.random() < labelled_selection_prob:
            idx = random.randint(0,self.num_labelled-1)
            data = self.data_labelled[idx]
        else:
            idx = random.randint(0,self.num_unlabelled-1)
            data = self.data_unlabelled[idx]
        
        element = {
            "input": self.transforms(data["input"]),
            "label": data["label"], 
            "labelled": data["labelled"],
            "index": data["index"]
        }
        
        return element
    
    def __len__(self):
        return len(self.data)


class SpecLoader():
    
    def __init__(self, path_to_dataset, cfg):
        self.path_to_dataset = path_to_dataset
        self.cfg = cfg
    
    def generate_split_for_teacher(self):
        assert self.train_set, "No train set object has been assigned yet. Call prepare_data_single first"
        training_split_percentage = self.cfg.training_split_percentage
     
    
    def prepare_data_single(self):
        print('==> Preparing data..')
        self.transform_test = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_train = transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
               
        transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        self.testset = torchvision.datasets.CIFAR10(root='{}/data'.format(self.path_to_dataset), 
                                                    train=False, 
                                                    download=True, 
                                                    transform=transformation)
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg.batch_size, 
                                                      shuffle=False, 
                                                      num_workers=self.cfg.num_workers)
         
        self.trainset = torchvision.datasets.CIFAR10(root='{}/data'.format(self.path_to_dataset), 
                                                     train=True, 
                                                     download=True, 
                                                     transform=transformation)
        self.batch_generator = DataLoader(self.trainset, 
                                          batch_size=self.cfg.batch_size, 
                                          shuffle=True, 
                                          num_workers=self.cfg.num_workers, 
                                          pin_memory=True)
        print("batch_size = {}".format(self.cfg.batch_size))
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.criterion = nn.CrossEntropyLoss()

    def gen_pseudolabels(self, model):
        
        print("generated pseudolabels")
        
        self.trainset = torchvision.datasets.CIFAR10(root='{}/data'.format(self.path_to_dataset), train=True, download=True)
        self.trainloader = torch.utils.data.DataLoader(self.trainset, 
                                                       batch_size=self.cfg.batch_size, 
                                                       shuffle=True, 
                                                       num_workers=self.cfg.num_workers)
        
        self.testset = torchvision.datasets.CIFAR10(root='{}/data'.format(self.path_to_dataset), train=False, download=True)
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg.batch_size, 
                                                      shuffle=False, 
                                                      num_workers=self.cfg.num_workers)
        
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.criterion = nn.CrossEntropyLoss()   
    
    
    
    
    
    