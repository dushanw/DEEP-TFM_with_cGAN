from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset, WeightedRandomSampler
from torch.optim.lr_scheduler import StepLR
import os
import random
import numpy as np
import pandas as pd
from skimage import io
from sklearn.preprocessing import MinMaxScaler  

from model import UNet
import h5py
" Go to Line number 163 "
from sklearn.metrics import classification_report

print("Random Seed: ", 13)
random.seed(13)
torch.manual_seed(13)
All_Accuracy = []
All_Epoch = []
#max_im = 3364
#max_gt = 741
max_im = 1
max_gt = 1
model_path = "ckpt/train_deep_tfm_loss_mse/"

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

class Dataset(Dataset):
    def __init__(self, img_dir):
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
        f_path = '/n/holyscratch01/wadduwage_lab/uom_bme/dataset_static_2020/20200105_synthBeads_1/tr_data_1sls/'
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

        _img = torch.from_numpy(np.divide(_img,max_im)).float()
        _target = torch.from_numpy(np.divide(_target,max_gt)).float()

        return _img,_target

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)

        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    train_loss /= len(train_loader.dataset)
    

    return train_loss

def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.mse_loss(output, target).item()  
    test_loss /= len(test_loader.dataset)

    return test_loss


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=100, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=67, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=80, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.3, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=13, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--output_nc', type=int, default=1, metavar='N',
                        help='output channels')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 2, 'pin_memory': True} if use_cuda else {}
    

    #Dataset making

    img_dir = '/n/holyscratch01/wadduwage_lab/uom_bme/dataset_static_2020/20200105_synthBeads_1/train.txt'
    img_dir_test = '/n/holyscratch01/wadduwage_lab/uom_bme/dataset_static_2020/20200105_synthBeads_1/test.txt'
    train_dataset = Dataset(img_dir=img_dir)
    test_dataset = Dataset(img_dir=img_dir_test)



    # Data Loading  
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)


    model = UNet(n_classes=args.output_nc).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)


    Best_ACC = 0
    Best_Epoch = 1


    for epoch in range(1, args.epochs + 1):
        tloss= train(args, model, device, train_loader, optimizer, epoch)
        vloss= test(args, model, device, test_loader)
        scheduler.step()
        try:
            os.makedirs(model_path)
        except OSError:
            pass
        torch.save(model.state_dict(), model_path +"/fcn_deep_" + str(epoch) + ".pth")

if __name__ == '__main__':
    main()
