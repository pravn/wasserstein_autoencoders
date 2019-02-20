from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Encoder, self).__init__()

        self.enc_mu = nn.Sequential(
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, 2 * ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2 * ngf),
            nn.ReLU(True),
            nn.Conv2d(2 * ngf, 4 * ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(4 * ngf),
            nn.ReLU(True),
            nn.Conv2d(4 * ngf, 8 * ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(8 * ngf),
            nn.ReLU(True),
            nn.Conv2d(8*ngf,nz,4, 1, 0, bias=False),
            nn.BatchNorm2d(nz),
            nn.ReLU(True))


    def forward(self, x):
        mu = self.enc_mu(x)
        return mu
    

class Decoder(nn.Module):
    def __init__(self, nc, nz, ngf):
        super(Decoder, self).__init__()

        self.convT1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(ngf * 8)
        # state size. (ngf*8) x 4 x 4

        self.convT2 = nn.ConvTranspose2d(ngf * 8 , ngf * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(ngf * 4)
        # state size. (ngf*4) x 8 x 8

        self.convT3 = nn.ConvTranspose2d(ngf * 4 , ngf * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(ngf * 2)
        # state size. (ngf*2) x 16 x 16

        self.convT4 = nn.ConvTranspose2d(ngf * 2 , ngf, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x 32 x 32

        self.convT5 = nn.ConvTranspose2d(ngf , ngf, 4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(ngf)
        # state size. (ngf) x 64 x 64

        self.convT6 = nn.ConvTranspose2d(ngf , nc, 3, 1, 1, bias=False)
        # state size. (nc) x 64 x 64

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.convT1(x)), 0.2, True)
        out = F.leaky_relu(self.bn2(self.convT2(out)), 0.2, True)
        out = F.leaky_relu(self.bn3(self.convT3(out)), 0.2, True)
        out = F.leaky_relu(self.bn4(self.convT4(out)), 0.2, True)
        out = F.leaky_relu(self.bn5(self.convT5(out)),0.2, True)
        out = F.tanh(self.convT6(out))
        return out



class WAE(nn.Module):
    def __init__(self, nc=3, nz=64, ngf=64):
        super(WAE, self).__init__()
        
        self.encoder = Encoder(nc, nz, ngf)
        self.decoder = Decoder(nc, nz, ngf)


    def forward(self, x):
        z = self.encoder(x)
        x = self.decoder(z)

        return x, z.squeeze()


class Discriminator(nn.Module):
    def __init__(self, nz=64, ndf=512):
        super(Discriminator, self).__init__()

        layers = []

        layers += [nn.Sequential(nn.Linear(nz, ndf),
                                 nn.ReLU(True))]

        for i in range(3):
            layers += [nn.Sequential(nn.Linear(ndf, ndf),
                                     nn.ReLU(True))]

        layers += [nn.Linear(ndf, 1)]
        layers += [nn.Sigmoid()]

        self.layers = nn.Sequential(*layers)

    def forward(self, z):
        out = self.layers(z)
        return out
        
