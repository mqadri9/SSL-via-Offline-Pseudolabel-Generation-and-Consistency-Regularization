import os
import sys
import numpy as np

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


class Config:
    trainset = ["CIFAR10"]
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, '..')
    model_dir = os.path.join(root_dir, 'models')
    dataset_dir = os.path.join(root_dir, 'datasets')
    
    net_arch = "resnet.ResNet50"
    dataset = "CIFAR10"
    lr = 1e-3
    
    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))        
    
cfg = Config()

add_pypath(os.path.join(cfg.dataset_dir))
add_pypath(os.path.join(cfg.model_dir))
for i in range(len(cfg.trainset)):
    add_pypath(os.path.join(cfg.dataset_dir, cfg.trainset[i]))


