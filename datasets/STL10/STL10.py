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

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

DATASET_NAME = "STL10"
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
classes = ('airplane', 'bird', 'car', 'cat', 'deer', 'dog', 'horse', 'monkey', 'ship', 'truck')

class DatasetLoader(Dataset):
    def __init__(self, data, cfg, path_to_dataset, transformation, already_transformed=False, multiplicative=1):
        print("Inside custom DatasetLoader")
        self.total_num_examples = 0
        self.cfg = cfg
        self.multiplicative = multiplicative
        self.transforms = transformation
        self.already_transformed = already_transformed
        self.data = data
        condition_labelled = {"labelled": "True"}
        self.data_labelled = list(filter(lambda item: all((item[k]==v for (k,v) in condition_labelled.items())), self.data))
        print(self.data_labelled[0]["input"])
        condition_unlabelled = {"labelled": "False"}
        self.data_unlabelled = list(filter(lambda item: all((item[k]==v for (k,v) in condition_unlabelled.items())), self.data))
        self.num_unlabelled = len(self.data_unlabelled)
        self.num_labelled = len(self.data_labelled)
        print("Number of labelled examples {}".format(self.num_labelled))
        print("Number of unlabelled examples {}".format(self.num_unlabelled))
        
    def __getitem__(self, index):
        labelled_selection_prob = self.cfg.labelled_selection_prob
        if self.cfg.training_split_percentage == 100:
            idx = random.randint(0, self.num_labelled-1)
            data =  self.data_labelled[idx]
        elif random.random() < labelled_selection_prob:
            idx = random.randint(0,self.num_labelled-1)
            data = self.data_labelled[idx]
        else:
            idx = random.randint(0,self.num_unlabelled-1)
            data = self.data_unlabelled[idx]
        
        if not self.already_transformed:
            print(data["input"])
            print(type(data["input"]))
            input = self.transforms(data["input"])
        else:
            input = data['input']
        element = {
            "input": input,
            "label": data["label"], 
            "labelled": data["labelled"],
            "index": data["index"],
            "cont_label": data["cont_label"]
        }        
        return element
    
    def __len__(self):
        if self.cfg.labelled_selection_prob == 100:
            return self.num_labelled
        return len(self.data)*self.multiplicative

class CustomCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(CustomCrossEntropyLoss, self).__init__()

    def forward(self, gt, gt_cont, predictions, labelled, cfg):
        cross_entropy = nn.CrossEntropyLoss()
        loss = cross_entropy(predictions, gt)
        return loss    


class SpecLoader():
    
    def __init__(self, path_to_dataset, cfg):
        self.path_to_dataset = path_to_dataset
        self.cfg = cfg
        self.transform_train=transforms.Compose([
            #transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            transforms.RandomRotation(30)
        ])
        
        self.transform_test = transforms.Compose([
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
            stat_portion = np.random.choice(4999, self.cfg.stats_samples_num)
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
        
    def gen_pseudolabels(self):
        pass