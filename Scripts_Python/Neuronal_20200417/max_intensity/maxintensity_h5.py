from __future__ import print_function
import argparse
import os
from math import log10
import numpy as np
import sys
import os
import random
from glob import glob
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as standard_transforms

import h5py
from skimage import io, exposure, img_as_uint, img_as_float
import imageio
args = {
    'num_class': 1,
    'ignore_label': 255,
    'num_gpus': 2,
    'start_epoch': 1,
    'num_epoch': 100,
    'batch_size': 20,
    'lr': 0.001,
    'lr_decay': 0.9,
    'dice': 0,
    'weight_decay': 1e-3,
    'momentum': 0.9,
    'snapshot': '',
    'snapshot2': '',
    'opt': 'adam',
    'beta1': 0.5,
    'input_nc': 32,
    'output_nc': 1,
    'dataset':'DEEP-TFM-norm-rm-black',
    'ckpt':'checkpoint'
}


class HDF5Dataset(Dataset):
    def __init__(self,img_dir, isTrain=True):
        self.isTrain = isTrain
        if isTrain: 
            fold_dir = "29711_ID.txt"   
        else: 
            fold_dir = "test.txt"

        ids = open(fold_dir, 'r')

        self.index_list = []
        
        for line in ids:
            self.index_list.append(line[0:-1])
        self.img_dir = img_dir

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, index):
        id_ = int(self.index_list[index]) 
        with h5py.File(self.img_dir, 'r') as db:
             _img = db['input'][id_] 

             _target = db['gt'][id_] 
        _img = _img.astype('int')
        _target = _target.astype('int')
        return _img, _target

img_dir = '/n/holyscratch01/wadduwage_lab/uom_bme/dataset_static_2020/cells_tr_data_6sls_17-Apr-2020.h5'

dataset_ = HDF5Dataset(img_dir=img_dir, isTrain=True)
training_data_loader = DataLoader(dataset=dataset_, batch_size=args['batch_size'], shuffle=True, num_workers=0, drop_last=True)

max_im = 0
max_gt = 0
for iteration, batch in enumerate(training_data_loader, 1):
   real_a_cpu, real_b_cpu = batch[0], batch[1]
   max_a = np.max(real_a_cpu.cpu().detach().numpy())
   if max_a > max_im:
       max_im = max_a
   max_b = np.max(real_b_cpu.cpu().detach().numpy())
   if max_b > max_gt:
       max_gt = max_b
print(max_im)
print(max_gt)
