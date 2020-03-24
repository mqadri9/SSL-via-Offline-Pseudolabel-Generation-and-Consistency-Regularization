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
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
class DatasetLoader(Dataset):
    def __init__(self, data, cfg, path_to_dataset, transformation, already_transformed=False):
        print("Inside custom DatasetLoader")
        self.total_num_examples = 0
        self.cfg = cfg
        self.transforms = transformation
        self.already_transformed = already_transformed
        self.data = data
        condition_labelled = {"labelled": True}
        self.data_labelled = list(filter(lambda item: all((item[k]==v for (k,v) in condition_labelled.items())), self.data))
        condition_unlabelled = {"labelled": False}
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
        return len(self.data)

class MiniDatasetLoader(Dataset):
    def __init__(self, data, transformation):
        self.data = data
        self.transformation = transformation
    
    def __getitem__(self, index):
        data =  self.data[index]
        element = {
            "input": self.transformation(data["input"]),
            "label": data["label"], 
            "labelled": data["labelled"],
            "index": data["index"]
        }
        return element

    def __len__(self):
        return len(self.data)

class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, gt, gt_cont, predictions, labelled, cfg):
        #print(labelled)
        pred_labelled = predictions[labelled]
        gt_labelled = gt[labelled]
        #print(pred_labelled.shape)
        #print(gt_labelled.shape)
        pred_unlabelled = predictions[~labelled]
        gt_unlabelled = gt_cont[~labelled]        
        #print(pred_unlabelled.shape)
        #print(gt_unlabelled.shape)        
        cross_entropy = nn.CrossEntropyLoss()
        labelled_loss = cross_entropy(pred_labelled, gt_labelled)
        num_unlabelled = gt_unlabelled.shape[0]
        unlabelled_loss = torch.abs(pred_unlabelled - gt_unlabelled).sum()/num_unlabelled
        
        #print("labelled loss {}".format(labelled_loss))
        #print("unlabelled loss {}".format(unlabelled_loss))
        lam = cfg.balancing_factor
        loss = labelled_loss + lam*unlabelled_loss
        #print("totol_loss: {}".format(loss))
        return loss

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
        self.transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        self.transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
               
        self.transformation = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
            
    def generate_split_for_teacher(self):
        assert self.train_set, "No train set object has been assigned yet. Call prepare_data_single first"
        training_split_percentage = self.cfg.training_split_percentage
     
    
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
        
        training_split_percentage = self.cfg.training_split_percentage
        cache_file = '{}_split_{}_percent_labelled.pkl'.format(DATASET_NAME, training_split_percentage)
        cache_file = os.path.join(self.path_to_dataset, "data", cache_file)
        # Check if we already have a pkl file saved for this split. If not regenerate the data
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                self.data = pk.load(fid)
            print('Loaded from {}, {} samples are loaded'.format(cache_file, len(self.data)))
        else:
            self.data = []
            total_num_samples = len(self.trainset)
            for index, d in enumerate(self.trainset):
                if index < int(training_split_percentage*total_num_samples / 100): 
                    element = {
                        "input": d[0],
                        "label": d[1], 
                        "labelled": True,
                        "index": index,
                        "cont_label": np.zeros((len(classes),))
                    }
                else:
                    element = {
                        "input": d[0],
                        "label": d[1], 
                        "labelled": False,
                        "index": index,
                        "cont_label": np.zeros((len(classes),))
                    }
                self.data.append(element)

            self.data = sorted(self.data, key = lambda i: i['labelled'], reverse=True)
            with open(cache_file, 'wb') as fid:
                pk.dump(self.data, fid, pk.HIGHEST_PROTOCOL)
            print('{} samples read wrote {}'.format(len(self.data), cache_file))
        
        self.trainset_loader = DatasetLoader(self.data, self.cfg, self.path_to_dataset, self.transform_train)

        self.batch_generator = DataLoader(self.trainset_loader, 
                                          batch_size=self.cfg.batch_size, 
                                          shuffle=True, 
                                          num_workers=self.cfg.num_workers, 
                                          pin_memory=True)
                
        print("batch_size = {}".format(self.cfg.batch_size))
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.criterion = CustomCrossEntropyLoss()
        self.cross_entropy_test = nn.CrossEntropyLoss()

    def gen_pseudolabels(self, model, data_orig, rt_lp):
        
        print("generated pseudolabels")
        
        self.testset = torchvision.datasets.CIFAR10(root='{}/data'.format(self.path_to_dataset), train=False, download=True, transform=self.transformation)
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg.batch_size, 
                                                      shuffle=False, 
                                                      num_workers=self.cfg.num_workers)

        model.eval()
        data = []
        i = 0
        print("Generating pseudolabels for retrain iteration {}..".format(rt_lp))
        miniloader = MiniDatasetLoader(data_orig, self.transform_train)
        minibatchsize = 32
        mini_batch_generator = DataLoader(miniloader, 
                                          batch_size=minibatchsize, 
                                          shuffle=False, 
                                          num_workers=2, 
                                          pin_memory=True)
        data = []
        for batch_idx, batch in enumerate(mini_batch_generator):
            if batch_idx % 1000 == 0:
                print("processing batch {}".format(batch_idx))
            #if batch_idx == 200:
            #    break
            #for each samples in batch:
            #    cuda_tensor = []
            #    for i in range(num augmentations)
            #          cuda_tensor.append(augment(sample))
            #    with torch.no_grad():
            #        outputs = model(cuda_tensor)
            #    takeornottake(outputs)
            
            inputs = batch["input"]
            label = batch['label']
            labelled = batch['labelled']
            index = batch['index']
            inputs_cuda = inputs.to(device)
            inputs = inputs.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            labelled = labelled.detach().cpu().numpy()
            index = index.detach().cpu().numpy()
            with torch.no_grad():
                outputs = model(inputs_cuda)
            del inputs_cuda
            outputs = outputs.detach().cpu().numpy()
            for i in range(inputs.shape[0]):
                tmp = {
                    'labelled':labelled[i],
                    'index': index[i],
                    'input': inputs[i]
                }
                tmp['label'] = label[i]
                tmp['cont_label'] = outputs[i]
                data.append(tmp)
                #===============================================================
                # print(type(tmp["input"]))
                # print(type(tmp["label"]))
                # print(type(tmp["labelled"]))
                # print(type(tmp["index"]))
                # sys.exit()
                #===============================================================
                  
        self.trainset_loader = DatasetLoader(data, self.cfg, self.path_to_dataset, self.transform_train, already_transformed=True)
        
        self.batch_generator = DataLoader(self.trainset_loader, 
                                          batch_size=self.cfg.batch_size, 
                                          shuffle=True, 
                                          num_workers=self.cfg.num_workers, 
                                          pin_memory=True)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.criterion = SPLoss()   
        self.cross_entropy_test = nn.CrossEntropyLoss()
    
