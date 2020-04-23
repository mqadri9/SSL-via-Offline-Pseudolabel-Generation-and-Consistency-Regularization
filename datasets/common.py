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
from torch.utils.data import DataLoader
import random
import scipy.stats
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
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

class GenPseudolabel():
    def __init__(self, cfg, transform_train, transform_data_distill):
        super(GenPseudolabel, self).__init__()
        self.cfg = cfg
        self.transform_train = transform_train
        self.transform_data_distill = transform_data_distill
    
    def get_svm_func(self, stat_portion):
        print("inside_svm_fun. Length of stat_portion: {}".format(len(stat_portion)))
        fun = None
        return fun
    
    def gen_pseudolabels(self, model, data_orig, rt_lp, prev_thresh, confidence_measure):
        model.eval()
        data = []
        i = 0
        print("Generating pseudolabels for retrain iteration {}..".format(rt_lp))
        
        n_augments = 10
        miniloader = MiniAugmentedDatasetLoader(data_orig, self.transform_data_distill, n_augments)
        #miniloader = MiniDatasetLoader(data_orig, self.transform_train)
        
        # TODO: Fix batch size so that we don't split up augmentations from the same sample
        minibatchsize = 50
        mini_batch_generator = DataLoader(miniloader, 
                                          batch_size=minibatchsize, 
                                          shuffle=False, 
                                          num_workers=2, 
                                          pin_memory=True)
        
        stat_portion = [x for x in data_orig if x["labelled"]=="stats"]
        if len(stat_portion)>0:
            fun = self.get_svm_func(stat_portion)
            params = {"fun":fun}
        else:
            params = {}
        
        
        data = []
        all_variances = []
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
                        output_arr.append(outputs[true_i:true_i+n_augments])
                        target_arr.append(tmp['label'])
                        take, variances, one_hot_pseudolabel, pseudolabel, skip = confidence_measure(outputs[true_i:true_i+n_augments], prev_thresh=prev_thresh,
                                                                                                     label=tmp['label'], params=params)
                        tmp['cont_label'] = torch.from_numpy(np.zeros(10)).type(torch.FloatTensor)
                        tmp['labelled'] = "stats"
                        take = False
                        if not skip:
                            if one_hot_pseudolabel != tmp['label']:
                                num_incorrect += 1
                                variances_incorrect.append(variances)
                                sum_variance_error += variances
                            else:
                                num_correct += 1
                                variances_correct.append(variances)
                                sum_variance_correct += variances   
                    else:
                        tmp['labelled'] = False
                        #m = np.mean(variances_correct)
                        #s = np.std(variances_correct)
                        #prev_thresh = m + (3*s)/(rt_lp+1)
                        prev_thresh=0
                        take, variances, one_hot_pseudolabel, pseudolabel, skip = confidence_measure(outputs[true_i:true_i+n_augments], prev_thresh=prev_thresh,
                                                                                                                 label=tmp['label'], params=params)
                        all_variances.append(variances)
                        tmp['cont_label'] = torch.from_numpy(pseudolabel).type(torch.FloatTensor)

                    if take:
                        data.append(tmp)
            #if batch_idx > 400:
            #    break
        
        print("maximum variance")
        print(np.mean(np.array(all_variances)))
        print(np.min(np.array(all_variances)))
        print(np.std(np.array(all_variances)))
        print(np.max(np.array(all_variances)))
        prev_thresh = np.mean(np.array(all_variances)) + 3*np.std(np.array(all_variances))
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
        np.save("variances_correct{}".format(rt_lp), variances_correct)
        np.save("variances_incorrect{}".format(rt_lp), variances_incorrect)        
        return data
    
    def gen_pseudolabels2(self, model, data_orig, rt_lp):
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
        return data

class ConfidenceMeasure():
    def __init__(self):
        pass
    
    def sharpen(self, pseudolabel, T=2):
        sign = np.sign(pseudolabel)
        num = np.power(np.abs(pseudolabel), 1/T)
        denom = np.sum(num)
        
        pseudolabel = sign*num/denom
        
        return pseudolabel
    
       
    def confidence_measure_1(self, reconstructions, prev_thresh, label=None, params={}):
        variances = np.min(np.var(reconstructions, axis=0))
        pseudolabel = np.mean(reconstructions, axis=0)
        if variances < 0.01:
            take = True
        else:
            take = False
        skip = False
        return take, variances, np.argmax(pseudolabel), pseudolabel, skip

    def confidence_measure_2(self, reconstructions, prev_thresh, label=None, params={}):
        pseudolabel = np.mean(reconstructions, axis=0)
        
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
    
    
    
    
