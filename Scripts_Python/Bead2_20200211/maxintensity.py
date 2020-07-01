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
#from networks import define_G, define_D, GANLoss, print_network
#from data import get_training_set, get_test_set
#import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as standard_transforms
#from model import UNet
#from Discriminator import Discriminator

#from model import UNet
#from torchvision.models import resnet18

#os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
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
        #self.img_anno_pairs = glob(img_dir)
        transforms_ = [standard_transforms.ToTensor()
                       #,standard_transforms.Normalize((26.704), (49.92))]
                       #standard_transforms.Normalize((25.704, 23.208), (49.92, 41.98))
                        ]

        self.transform = standard_transforms.Compose(transforms_)

        transforms_target  = [standard_transforms.ToTensor()
                       #,standard_transforms.Normalize((23.74), (45.17))
                           ]

        self.transform_target = standard_transforms.Compose(transforms_target)
        # self.transform_img = standard_transforms.Compose(
        #     standard_transforms.ToTensor(), standard_transforms.Normalize((26.704, 23.208), (49.92, 41.98)))
        # self.transform_target = standard_transforms.Compose(
        #     standard_transforms.ToTensor(), standard_transforms.Normalize((23.74), (45.17)))
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
        #print(self.img_anno_pairs[index])
        j=0
        for i in range(0,32):            
            _img[:, :, j] = io.imread(f_path + self.img_anno_pairs[index]+'_'+str(i+1)+'.png') 
          #_img[:, :, j+1] = Image.open(f_path + self.img_anno_pairs[index] + '_' + 'pattern'+str(i+1) + '.png')
            j = j+1
            #print(np.max(_img))
            #_img[:, :, 1] = Image.open(self.img_anno_pairs[index] + '_flair.png')
        _target[0,:,:] = io.imread(f_path + self.img_anno_pairs[index] + '_gt.png')
        #_target = np.array(_target)
        #print('target',np.max(_target))
        #_img = self.transform(np.array(_img, dtype=np.uint8))
        #_target = self.transform(np.array(_target, dtype=np.uint8))
        if np.min(np.array(_target)) == np.max(np.array(_target)):
            print('Yes')
            k=0
            for i in range(0,32):
                 _img[:, :, k] = io.imread(f_path + self.img_anno_pairs[index+1]+'_'+str(i+1)+'.png')
                 k = k+1
            _target[0,:,:] = io.imread(f_path + self.img_anno_pairs[index+1] + '_gt.png')
        #_img = self.transform(np.array(_img, dtype=np.uint16))
        #_target = self.transform(np.array(_target, dtype=np.uint16))
        _img = np.transpose(_img,(2,0,1))

        #_img = self.transform(np.array(_img)[None, :, :])
        #_target = self.transform(np.array(_target)[None, :, :])
        #print(np.unique(np.array(_target)))
        #print('img_size',_img.shape)
        #print(np.unique(np.array(_img)))
        #_img = torch.from_numpy(np.array(_img)[None, :, :]).float()
        #_target = torch.from_numpy(np.array(_target)[None, :, :]).float()
        #cv2.imwrite('test.png',_imgn)
        
        return _img, _target

img_dir = 'outfile_1000.txt'

dataset_ = Dataset(img_dir=img_dir, transform=None)
training_data_loader = DataLoader(dataset=dataset_, batch_size=args['batch_size'], shuffle=True, num_workers=0, drop_last=True)

max_im = 0
max_gt = 0
for iteration, batch in enumerate(training_data_loader, 1):
   #print(iteration) 
   # forward
   real_a_cpu, real_b_cpu = batch[0], batch[1]
   max_a = np.max(real_a_cpu.cpu().detach().numpy())
   if max_a > max_im:
       max_im = max_a

   max_b = np.max(real_b_cpu.cpu().detach().numpy())
   if max_b > max_gt:
       max_gt = max_b
#print(iteration)
print(max_im)
print(max_gt)
