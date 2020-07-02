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
from model import UNet
from Discriminator import Discriminator
import h5py
from skimage import io, exposure, img_as_uint, img_as_float
import imageio

from skimage import img_as_ubyte

args = {
    'num_class': 1,
    'ignore_label': 255,
    'num_gpus': 2,
    'start_epoch': 1,
    'num_epoch': 300,
    'batch_size': 100,
    'lr': 0.0005,
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
    'dataset':'DEEP-TFM-lr-0.0005-without-norm',
    'ckpt':'checkpoint'
}
max_im = 1
max_gt = 1
#max_gt = 741
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real
        return self.loss(input, target_tensor.cuda())

class HDF5Dataset(Dataset):
    def __init__(self,img_dir, isTrain=True):
        self.isTrain = isTrain
        if isTrain: 
            fold_dir = "train.txt"   
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
        _img = np.dtype('>u2') 
        _target = np.dtype('>u2') 
        id_ = int(self.index_list[index])
        with h5py.File(self.img_dir, 'r') as db:
             _img = db['input'][id_] 
             _target = db['gt'][id_] 
        if np.max(_target) == 0:
             with h5py.File(self.img_dir, 'r') as db:
                 _img = db['input'][id_+1]
                 _target = db['gt'][id_+1]
        _img = torch.from_numpy(np.divide(_img,max_im)).float()
        _target = torch.from_numpy(np.divide(_target,max_gt)).float()
        
        return _img, _target

class XSigmoidLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_t, y_prime_t):
        ey_t = y_t - y_prime_t
        return torch.mean(2 * ey_t / (1 + torch.exp(-ey_t)) - ey_t)


img_dir = '/n/holyscratch01/wadduwage_lab/uom_bme/ForwardModel_matlab/_cnn_synthTrData/03-Jun-2020/cells_tr_data_6sls_03-Jun-2020.h5'
dataset_ = HDF5Dataset(img_dir=img_dir, isTrain=True)
training_data_loader = DataLoader(dataset=dataset_, batch_size=args['batch_size'], shuffle=True, num_workers=0, drop_last=True)

dataset_test = HDF5Dataset(img_dir=img_dir, isTrain=False)
testing_data_loader = DataLoader(dataset=dataset_test, batch_size=args['batch_size'], shuffle=True, num_workers=0, drop_last=True)


netG = UNet(n_classes=args['output_nc']).cuda()
netG = torch.nn.parallel.DataParallel(netG, device_ids=range(args['num_gpus']))
netD = Discriminator().cuda()
netD = torch.nn.parallel.DataParallel(netD, device_ids=range(args['num_gpus']))

criterionGAN = GANLoss().cuda()
criterionL1 = nn.L1Loss().cuda()
criterionMSE = nn.MSELoss().cuda()
criterionxsig = XSigmoidLoss().cuda()
# setup optimizer

optimizerD = optim.Adam(netD.parameters(), lr=args['lr'], betas=(args['beta1'], 0.999))

real_a = torch.FloatTensor(args['batch_size'], args['input_nc'], 128, 128).cuda()
real_b = torch.FloatTensor(args['batch_size'], args['output_nc'], 128, 128).cuda()


real_a = Variable(real_a)
real_b = Variable(real_b)

resume_epoch = 35
def test(args, model, device, test_loader, k_fold, class_weights):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, weight = class_weights).item()  # sum up batch loss


    test_loss /= len(test_loader.dataset)
    return test_loss, 100. * correct / len(test_loader.dataset) , report

def train(epoch):
    for iteration, batch in enumerate(training_data_loader, 1):
        
        # forward
        real_a_cpu, real_b_cpu = batch[0], batch[1]
       	real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
       	real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)

        fake_b = netG(real_a)
        #print(fake_b.size())
        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################

        optimizerD.zero_grad()
        
        # train with fake
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake, False)
        # train with real

        real_ab = torch.cat((real_a, real_b), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True)
        
        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5
            
        loss_d.backward()       
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()
        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True)

         # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * 10

        loss_g = loss_g_gan + loss_g_l1

        loss_g.backward()

        optimizerG.step()
        
        if iteration % 200 == 0: 
            print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f}".format(
                epoch, iteration, len(training_data_loader), loss_d.item(), loss_g.item()))
    netG.eval()
    test_loss = 0
    for iteration, batch in enumerate(testing_data_loader, 1):
        real_a_cpu, real_b_cpu = batch[0], batch[1]
       	real_a.resize_(real_a_cpu.size()).copy_(real_a_cpu)
       	real_b.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        fake_b = netG(real_a)
        test_loss += criterionL1(fake_b, real_b).item()
    print(len(testing_data_loader.dataset))
    test_loss /= len(testing_data_loader.dataset)
    print('epoch[{}]: Loss_test: {:.4f}'.format(epoch,test_loss))
           
def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", args['dataset'])):
        os.mkdir(os.path.join("checkpoint", args['dataset']))
    net_g_model_out_path = "checkpoint/{}/netG_model_epoch_{}.pth.tar".format(args['dataset'], epoch)
    net_d_model_out_path = "checkpoint/{}/netD_model_epoch_{}.pth.tar".format(args['dataset'], epoch)
    torch.save(netG, net_g_model_out_path)
    torch.save(netD, net_d_model_out_path)
    print("Checkpoint saved to {}".format("checkpoint" + args['dataset']))

for epoch in range(1, args['num_epoch'] + 1):
    train(epoch)
    checkpoint(epoch)