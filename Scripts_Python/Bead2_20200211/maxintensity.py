from __future__ import print_function
import argparse
import os
from math import log10
import numpy as np
import sys
import os
import random
from glob import glob
from skimage import io
from PIL import Image
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as standard_transforms

args = {
    'num_class': 1,
    'ignore_label': 255,
    'num_gpus': 2,
    'start_epoch': 1,
    'num_epoch': 200,
    'batch_size': 50,
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
    'dataset':'DEEP-TFM-dynamic',
    'ckpt':'checkpoint'
}


class Dataset(Dataset):
    def __init__(self, img_dir, transform=None):

        idx = 0
        file_img = open(img_dir, 'r')
        self.img_anno_pairs = {}
        for line in file_img:
            self.img_anno_pairs[idx] = line[0:-1]
            idx = idx + 1      
        
    def __len__(self):
        return len(self.img_anno_pairs)

    def __getitem__(self, index):
        _img = np.zeros((128,128,32))
        _target = np.zeros((1,128,128))
        f_path = '/n/holyscratch01/wadduwage_lab/uom_bme/dataset_static_2020/20200211_synthBeads_2/tr_data_6sls/'
        
        j=0
        for i in range(0,32):            
            _img[:, :, j] = io.imread(f_path + self.img_anno_pairs[index]+'_'+str(i+1)+'.png') 
            j = j+1
        _target[0,:,:] = io.imread(f_path + self.img_anno_pairs[index] + '_gt.png')

        if np.min(np.array(_target)) == np.max(np.array(_target)):
            print('Yes')
            k=0
            for i in range(0,32):
                 _img[:, :, k] = io.imread(f_path + self.img_anno_pairs[index+1]+'_'+str(i+1)+'.png')
                 k = k+1
            _target[0,:,:] = io.imread(f_path + self.img_anno_pairs[index+1] + '_gt.png')
        _img = np.transpose(_img,(2,0,1))

        
        return _img, _target

img_dir = 'outfile_1000.txt'

dataset_ = Dataset(img_dir=img_dir, transform=None)
training_data_loader = DataLoader(dataset=dataset_, batch_size=args['batch_size'], shuffle=True, num_workers=0, drop_last=True)

max_im = 0
max_gt = 0
for iteration, batch in enumerate(training_data_loader, 1):

   # forward
   real_a_cpu, real_b_cpu = batch[0], batch[1]
   max_a = np.max(real_a_cpu.cpu().detach().numpy())
   if max_a > max_im:
       max_im = max_a

   max_b = np.max(real_b_cpu.cpu().detach().numpy())
   if max_b > max_gt:
       max_gt = max_b

print(max_im)
print(max_gt)
