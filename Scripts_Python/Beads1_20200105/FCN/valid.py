import argparse
import os
import numpy as np
from PIL import Image
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from skimage import img_as_ubyte
import torch.nn as nn
from skimage.io import imread, imsave
from skimage import io
from glob import glob
from skimage import io, exposure, img_as_uint, img_as_float
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
from model import UNet


# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--batchSize', type=int, default=4, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=200, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=16, help='input image channels')
parser.add_argument('--output_nc', type=int, default=1, help='output image channels')
parser.add_argument('--ngf', type=int, default=64, help='generator filters in first conv layer')
parser.add_argument('--ndf', type=int, default=64, help='discriminator filters in first conv layer')
parser.add_argument('--lr', type=float, default=0.0002, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--dataset', default=True, help='DEEP-TFM-l1loss')
parser.add_argument('--model', type=str, default='checkpoint/DEEP-TFM-l1loss/netG_model_epoch_50.pth.tar', help='model file to use')
parser.add_argument('--cuda', default=True, help='use cuda')
opt = parser.parse_args(args=[])
max_im = 1
max_gt = 1
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

criterionMSE = nn.MSELoss() #.to(device)


transform = transforms.Compose(transform_list)

img_dir = open('/n/holyscratch01/wadduwage_lab/uom_bme/dataset_static_2020/20200105_synthBeads_1/test.txt','r')

avg_mse = 0
avg_psnr = 0

for epochs in range(55,56):
    my_model = 'ckpt/train_deep_tfm_loss_mse/fcn_deep_' + str(epochs) + '.pth'
    netG = UNet(n_classes=args.output_nc)
    netG.load_state_dict(torch.load(my_model))
    netG.eval()
    p = 0
    f_path = '/n/holyscratch01/wadduwage_lab/uom_bme/dataset_static_2020/20200105_synthBeads_1/tr_data_1sls/'    
    for line in img_dir:
        print(line)
        GT_ = io.imread(f_path + str(line[0:-1]) + '_gt.png')
        modalities = np.zeros((32,128,128))
        for i in range(0,32):
             modalities[i,:,:] = io.imread(f_path + str(line[0:-1]) +'_'+str(i+1) +'.png')  
        depth = modalities.shape[2]
        predicted_im = np.zeros((128,128,1))
        if np.min(np.array(GT_))==np.max(np.array(GT_)):
             print('Yes')
        GT = torch.from_numpy(np.divide(GT_,max_gt))
        img = torch.from_numpy(np.divide(modalities,max_im)[None, :, :]).float()
        netG = netG.cuda()
        input = img.cuda()
        out = netG(input)
        out = out.cpu()
        out_img = out.data[0]
        out_img = np.squeeze(out_img)
        GT = np.squeeze(GT)
        predict_path= 'Predicted_mse/epoch_' + str(epochs) +'/'
        if not os.path.exists(predict_path):
            os.makedirs(predict_path)
        imsave(predict_path + '/' + str(line[0:-1]) + '_pred.png',out_img)
        imsave(predict_path + '/' + str(line[0:-1]) + '_gt.png',(GT))
print('mse=',torch.div(avg_mse,p))
print(avg_mse)
print(avg_psnr)
print(p)
img_dir.close()
