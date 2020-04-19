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
#os.environ["CUDA_VISIBLE_DEVICES"]="1"

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
            #print(data["input"])
            #print(type(data["input"]))
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

class DatasetLoader2(Dataset):
    def __init__(self, data, cfg, path_to_dataset, transformation, already_transformed=False, multiplicative=1):
        print("Inside custom DatasetLoader")
        self.total_num_examples = 0
        self.cfg = cfg
        self.multiplicative = multiplicative
        self.transforms = transformation
        self.already_transformed = already_transformed
        self.data = data
        condition_labelled = {"labelled": True}
        self.data_labelled = list(filter(lambda item: all((item[k]==v for (k,v) in condition_labelled.items())), self.data))
        print(self.data_labelled[0]["input"])
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
            #print(data["input"])
            #print(type(data["input"]))
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


class MiniAugmentedDatasetLoader(Dataset):
    def __init__(self, data, transformation, n_augmentations):
        self.data = data
        self.transformation = transformation
        self.n_augments = n_augmentations

    def __getitem__(self, index):
        true_index = int(index/self.n_augments)
        data = self.data[true_index]
        element = {
            "input": self.transformation(data["input"]),
            "label": data["label"], 
            "labelled": data["labelled"],
            "index": data["index"]
        }
        return element

    def __len__(self):
        return self.n_augments*len(self.data)


class SPLoss(nn.Module):
    def __init__(self):
        super(SPLoss, self).__init__()

    def forward(self, gt, gt_cont, predictions, labelled, cfg):
        #labelled = torch.where(labelled=="True", labelled)
        pred_labelled = predictions[labelled]
        gt_labelled = gt[labelled]
        pred_unlabelled = predictions[~labelled]
        gt_unlabelled = gt_cont[~labelled]        
        num_labelled = gt_labelled.shape[0]
        if num_labelled == 0:
            labelled_loss = 0
        else:
            cross_entropy = nn.CrossEntropyLoss()
            labelled_loss = cross_entropy(pred_labelled, gt_labelled)
        num_unlabelled = gt_unlabelled.shape[0]
        if num_unlabelled == 0:
            unlabelled_loss = 0
        else:
            unlabelled_loss = torch.pow(pred_unlabelled - gt_unlabelled, 2).mean()
        lam = cfg.balancing_factor
        loss = labelled_loss + lam*unlabelled_loss
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
        self.transform_train=transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.transform_data_distill=transforms.Compose([
            transforms.RandomCrop(96, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(30),
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
        
    def gen_pseudolabels(self, model, data_orig, rt_lp):
        
        print("generated pseudolabels")
        
        self.testset = torchvision.datasets.STL10(root='{}/data'.format(self.path_to_dataset), split="test",  download=True, transform=self.transformation)
        self.testloader = torch.utils.data.DataLoader(self.testset, 
                                                      batch_size=self.cfg.batch_size, 
                                                      shuffle=False, 
                                                      num_workers=self.cfg.num_workers)

        model.eval()
        data = []
        i = 0
        print("Generating pseudolabels for retrain iteration {}..".format(rt_lp))
        
        n_augments = 10

        miniloader = MiniAugmentedDatasetLoader(data_orig, self.transform_data_distill, n_augments)
        #miniloader = MiniDatasetLoader(data_orig, self.transform_train)
        
        # TODO: Fix batch size so that we don't split up augmentations from the same sample
        minibatchsize = 150
        mini_batch_generator = DataLoader(miniloader, 
                                          batch_size=minibatchsize, 
                                          shuffle=False, 
                                          num_workers=2, 
                                          pin_memory=True)
        data = []
        sum_variance_error = 0
        sum_variance_correct = 0
        variances_correct = []
        variances_incorrect = []
        num_correct = 0
        num_incorrect = 0
        output_arr = []
        target_arr = []
        for batch_idx, batch in enumerate(mini_batch_generator):
            if batch_idx % 1000 == 0:
                print("processing batch {}".format(batch_idx))
            inputs = batch["input"]
            label = batch['label']
            labelled = batch['labelled']
            index = batch['index']
            inputs_cuda = inputs.to(device)
            inputs = inputs.detach().cpu().numpy()
            label = label.detach().cpu().numpy()
            #labelled = labelled.detach().cpu().numpy()
            labelled = np.array(labelled)
            index = index.detach().cpu().numpy()
            with torch.no_grad():
                outputs = model(inputs_cuda)
            del inputs_cuda
            outputs = outputs.detach().cpu().numpy()
            for true_i in range(0, minibatchsize, n_augments):
                take = True
                orig_index = index[true_i]
                sample = data_orig[orig_index]
                assert sample['index'] == orig_index
                assert sample['labelled'] == labelled[true_i]
                assert sample['label'] == label[true_i]
                tmp = {
                    'index': sample['index'],
                    'input': sample['input']
                }
                tmp['label'] = sample['label']
                if sample['labelled'] == "True":
                    tmp['labelled'] = True
                    tmp['cont_label'] = torch.from_numpy(np.zeros(10)).type(torch.FloatTensor)
                elif sample["labelled"] == "stats":
                    conf_meas = CondifenceMeasure()
                    output_arr.append(outputs[true_i:true_i+n_augments])
                    target_arr.append(tmp['label'])
                    take, variances, one_hot_pseudolabel, pseudolabel = conf_meas.confidence_measure_1(outputs[true_i:true_i+n_augments], 
                                                                                  label=tmp['label'])
                    tmp['cont_label'] = torch.from_numpy(np.zeros(10)).type(torch.FloatTensor)
                    if one_hot_pseudolabel != tmp['label']:
                        num_incorrect += 1
                        variances_incorrect.append(variances)
                        sum_variance_error += variances
                        #print("{} | {}".format(variances, 0))
                    else:
                        num_correct += 1
                        variances_correct.append(variances)
                        sum_variance_correct += variances
                        #print("{} | {}".format(variances, 1))
                                           
                else:
                    tmp['labelled'] = False
                    conf_meas = CondifenceMeasure()
                    output_arr.append(outputs[true_i:true_i+n_augments])
                    target_arr.append(tmp['label'])
                    take, variances, one_hot_pseudolabel, pseudolabel = conf_meas.confidence_measure_1(outputs[true_i:true_i+n_augments], 
                                                                                  label=tmp['label'])
                    tmp['cont_label'] = torch.from_numpy(pseudolabel).type(torch.FloatTensor)

                if take:
                    data.append(tmp)
            #if batch_idx > 1000:
            #    break
        
                
        variances_incorrect = np.array(variances_incorrect)
        variances_correct = np.array(variances_correct)
        print("average correct variance")
        print(np.mean(variances_correct))
        print(np.std(variances_correct))
        print("average incorrect variance")
        print(np.mean(variances_incorrect))
        print(np.std(variances_incorrect))
        print("Statistics")
        print(np.array(data).shape)
        print(num_correct)
        print(num_incorrect)
        np.save("output_arr_{}".format(rt_lp), output_arr)
        np.save("target_arr{}".format(rt_lp), target_arr)
        #for d in data:
        #    print("============================")
        #    for k in d:
        #        print("k {}   || type(k) {}".format(k, type(d[k])))
                
        #=======================================================================
        # self.trainset_loader = DatasetLoader(data, self.cfg, self.path_to_dataset, self.transform_train, already_transformed=True)
        # 
        # self.batch_generator = DataLoader(self.trainset_loader, 
        #                                   batch_size=self.cfg.batch_size, 
        #                                   shuffle=True, 
        #                                   num_workers=self.cfg.num_workers, 
        #                                   pin_memory=True)
        #=======================================================================
        #data = data[0:4999]
          
        self.trainset_loader = DatasetLoader2(data, self.cfg, self.path_to_dataset, self.transform_train, multiplicative=self.cfg.multiplicative)

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
        
        self.testset = torchvision.datasets.STL10(root='{}/data'.format(self.path_to_dataset), split="test", download=True, transform=self.transformation)
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
                                          num_workers=self.cfg.num_workers, 
                                          pin_memory=True)
        data = []
        print("Loaded mini_batch_generator")
        for batch_idx, batch in enumerate(mini_batch_generator):
            if batch_idx % 1000 == 0:
                print("processing batch {}".format(batch_idx))
            #if batch_idx == 200:
            #    break
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
        
        self.trainset_loader = DatasetLoader2(data, self.cfg, self.path_to_dataset, self.transform_train, already_transformed=True)
        
        self.batch_generator = DataLoader(self.trainset_loader, 
                                          batch_size=self.cfg.batch_size, 
                                          shuffle=True, 
                                          num_workers=self.cfg.num_workers, 
                                          pin_memory=True)
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        
        self.criterion = SPLoss()   
        self.cross_entropy_test = nn.CrossEntropyLoss()

    
    
    
    
    
    
    
    
    