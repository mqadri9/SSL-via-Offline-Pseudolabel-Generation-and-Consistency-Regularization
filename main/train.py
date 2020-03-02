import config as config
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
import numpy as np
import os, sys
import copy as cp
import argparse
import importlib
from utils import progress_bar

import parser

#===============================================================================
parser = argparse.ArgumentParser(description='semi-supervised Training')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--path', '-p', help='path to model where to resume')
args = parser.parse_args()
 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
 
cfg = config.Config
architecture, type = cfg.net_arch.rsplit(".", 1)
arch = importlib.import_module(architecture)
network = getattr(arch, type)
 
path_to_dataset = os.path.join(cfg.dataset_dir, cfg.dataset)
dataset = importlib.import_module(cfg.dataset)
 
def delete_network(net):
    del net

def get_optimizer(optimizer_name, net):
    if optimizer_name == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr, momentum=0.9, weight_decay=5e-4) 
    else:
        print("Error! Unknown optimizer name: ", optimizer_name)
        assert 0

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=cfg.lr_dec_epoch, gamma=cfg.lr_dec_factor)
    return optimizer, scheduler
    
def create_network():
    net = network()
    net = net.to(device)
    optimizer, scheduler = get_optimizer("sgd", net)
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
    return net, optimizer, scheduler

def load_teacher():
    net_teacher, _, _ = create_network()
    checkpoint = torch.load('{}/checkpoint_teacher/ckpt_loop_0_perc_{}.pth'.format(path_to_dataset,
                                                                                    cfg.training_split_percentage))
    net_teacher.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    optimizer = checkpoint['optimizer']
    scheduler = checkpoint['scheduler']
    return net_teacher, best_acc, start_epoch, optimizer, scheduler

def get_latest_starting_model_params(mypath):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(mypath) if f.startswith("ckpt_loop_")]
    for file in onlyfiles:
        arr = file.split("_")
    

def load_student_as_new_teacher(index):
    net_teacher, optimizer = create_network()
    checkpoint = torch.load('{}/checkpoint_student/ckpt_loop_{}.pth'.format(path_to_dataset, index))
    net_teacher.load_state_dict(checkpoint['net'])
    return net_teacher

def train(specLoader, net, optimizer, scheduler, fun="teacher", rt_lp=1, start_epoch=0, best_acc=0):
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    print("Start epoch {}".format(start_epoch))
    for epoch in range(start_epoch, cfg.num_epochs):
        print('\nEpoch: %d' % epoch)
        scheduler.step() 
        train_loss = 0
        correct = 0
        total = 0
        iter = 0
        net.train()
        for batch_idx, data in enumerate(specLoader.batch_generator):
            inputs = data["input"]
            targets = data['label']
            cont_targets = data['cont_label']
            #print(inputs.shape)
            #print(targets.shape)
            #print(cont_targets.shape)
            inputs, targets, cont_targets = inputs.to(device), targets.to(device), cont_targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = specLoader.criterion(targets, cont_targets, outputs, data['labelled'], cfg)
            loss.backward()
            optimizer.step()
     
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            iter +=1
            progress_bar(batch_idx, len(specLoader.batch_generator), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %f'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, scheduler.get_lr()[0]))
        
        test_loss = 0
        correct = 0
        total = 0                       
        net.eval()
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(specLoader.testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = specLoader.cross_entropy_test(outputs, targets)
     
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
     
                progress_bar(batch_idx, len(specLoader.testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d) | lr: %f'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total, scheduler.get_lr()[0]))
     
        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
                'optimizer': optimizer,
                'scheduler': scheduler
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')                
            torch.save(state, '{}/checkpoint_{}/ckpt_loop_{}_perc_{}.pth'.format(path_to_dataset, fun, rt_lp, cfg.training_split_percentage))
            best_acc = acc
                 
        ## NEED TO RECREATE PSEUDOLABELS FROM NEW STUDENT MODEL

if __name__ == "__main__":
    specLoader = dataset.SpecLoader(path_to_dataset, cfg)
    specLoader.prepare_data_single()
    data = specLoader.data
    if cfg.load_latest_teacher:
        net_teacher, best_acc, start_epoch, optimizer, scheduler = load_teacher()
        if cfg.train_teacher and args.resume:
            train(specLoader,
                  net_teacher, 
                  optimizer,
                  scheduler,
                  fun="teacher", 
                  rt_lp=0, 
                  start_epoch=start_epoch, 
                  best_acc=best_acc)
    else:
        net_teacher, optimizer, scheduler = create_network()
        train(specLoader, net_teacher, optimizer, scheduler, fun="teacher", rt_lp=0, start_epoch=0, best_acc=0)
       
    print("Loaded teacher network")
    max_retrain_loop = cfg.max_retrain_loop
    if cfg.train_teacher:
        max_retrain_loop = 0
    for rt_lp in range(max_retrain_loop):
        print("training student loop {}".format(rt_lp))
         
        specLoader = dataset.SpecLoader(path_to_dataset, cfg)
        specLoader.gen_pseudolabels(net_teacher, data, rt_lp)
        net_student, optimizer, scheduler = create_network()
         
        train(specLoader, net_student, optimizer, scheduler, fun="student", rt_lp=rt_lp, start_epoch=0, best_acc=0)
         
        net_teacher = load_student_as_new_teacher(rt_lp)
        
        
        
        
