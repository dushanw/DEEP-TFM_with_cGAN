import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn



class Discriminator(nn.Module):
    def __init__(self, hidden_dim=1024):
        super(Discriminator, self).__init__()
        self.hidden_dim = hidden_dim

        
        self.conv1 = nn.Conv2d(33, 16, 4, 2, 1)
        self.conv2 = nn.Conv2d(16, 32, 4, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 4, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 4, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 4, 2, 1)
        self.conv6 = nn.Conv2d(256, 512, 4, 2, 1)
        self.re    = nn.LeakyReLU(0.2, True)
        self.bn1   = nn.BatchNorm2d(16)
        self.bn2   = nn.BatchNorm2d(32)
        self.bn3   = nn.BatchNorm2d(64)
        self.bn4   = nn.BatchNorm2d(128)
        self.bn5   = nn.BatchNorm2d(256)
        self.bn7   = nn.BatchNorm2d(512)
        self.bn6   = nn.BatchNorm2d(1)

        self.dconv = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.dconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dconv2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dconv3 = nn.ConvTranspose2d(64, 32, 4, 2, 1)
        self.dconv4 = nn.ConvTranspose2d(32, 16, 4, 2, 1)
        self.dconv5 = nn.ConvTranspose2d(16, 1, 4, 2, 1)
            
        # self.scse1 = SCSEBlock(channel=256)
        # self.scse2 = SCSEBlock(channel=128)
        # self.scse3 = SCSEBlock(channel=64)
        # self.scse4 = SCSEBlock(channel=32)
        # self.scse5 = SCSEBlock(channel=16)
        
    def forward(self, x):

        e1 = self.re(self.bn1(self.conv1(x)))
        e2 = self.re(self.bn2(self.conv2(e1)))
        e3 = self.re(self.bn3(self.conv3(e2)))
        e4 = self.re(self.bn4(self.conv4(e3)))
        e5 = self.re(self.bn5(self.conv5(e4)))
        e6 = self.re(self.bn7(self.conv6(e5)))

        
        #print('e6',e6.size)
        
        d6 = self.dconv(e6)
        d6 = self.re(self.bn5(d6))
        #d6 = d6 + self.scse1(d6) 
        d5 = self.dconv1(d6)
        #d5 = F.interpolate(d5, size=e4.size()[2:], mode='nearest') + e4
        #d5 = d5 + e4
        d5 = self.re(self.bn4(d5))
        #d5 = d5 + self.scse2(d5) 
        d4 = self.dconv2(d5) 
        #d4 = F.interpolate(d4, size=e3.size()[2:], mode='nearest') + e3
        d4 = self.re(self.bn3(d4))
        #d4 = d4 + self.scse3(d4) 
        d3 = self.dconv3(d4) 
        #d3 = F.interpolate(d3, size=e2.size()[2:], mode='nearest') + e2
        d3 = self.re(self.bn2(d3))
        #d3 = d3 + self.scse4(d3) 
        d2 = self.dconv4(d3) 
        #d2 = F.interpolate(d2, size=e1.size()[2:], mode='nearest') + e1
        d2 = self.re(self.bn1(d2))
        #d2 = d2 + self.scse5(d2) 
        d1 = self.dconv5(d2)
        
        #print('d1',d1.size())
        #d1 = F.interpolate(d1, size=x.size()[2:], mode='nearest') + x
        out = self.re(self.bn6(d1))
        #out = F.interpolate(out, size=(138,186,186), mode='trilinear')
        #print('out',out.size())
        #hidden = self.encode(image)
        #out = self.decode(hidden)
        return out#, hidden.view(image.size(0), -1)
