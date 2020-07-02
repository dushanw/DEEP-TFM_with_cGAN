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

from sklearn.preprocessing import MinMaxScaler  

from model import UNet
import h5py
" Go to Line number 163 "
from sklearn.metrics import classification_report
import skimage.io as io
print("Random Seed: ", 13)
random.seed(13)
torch.manual_seed(13)
All_Accuracy = []
All_Epoch = []
max_im = 1
max_gt = 1
#max_im = 1
#max_gt = 1
model_path = "ckpt/deep_tfm/"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
        f_path = '/n/holyscratch01/wadduwage_lab/uom_bme/dataset_static_2020/11-Feb-2020/tr_data_5sls/'
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

def train(args, model, device, train_loader, optimizer, epoch, criterion):
    model.train()
    train_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output,target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
    
    train_loss /= len(train_loader.dataset)
    

    return train_loss

def test(args, model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output,target).item()

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
    
    train_dir = 'train.txt'
    test_dir = 'test.txt'
    train_dataset = Dataset(img_dir=train_dir)
    test_dataset = Dataset(img_dir=test_dir)



    # Data Loading # 
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)

    model = UNet(n_classes=args.output_nc).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-6)
    scheduler = StepLR(optimizer, step_size=20, gamma=args.gamma)
    criterion = torch.nn.SmoothL1Loss()
    Best_ACC = 0
    Best_Epoch = 1


    for epoch in range(1, args.epochs + 1):
        tloss= train(args, model, device, train_loader, optimizer, epoch,criterion)
        vloss= test(args, model, device, test_loader,criterion)
        print("epoch:%.1f" %epoch, "Train_loss:%.4f" % tloss, "Val_loss:%.4f" % vloss)
        scheduler.step()
        try:
            os.makedirs(model_path)
        except OSError:
            pass
        torch.save(model.state_dict(), model_path +"/fcn_deep_" + str(epoch) + ".pth")

if __name__ == '__main__':

    main()
