import argparse
import torch
import os
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch import autograd
#from torchvision.datasets import MNIST
#from torchvision.transforms import transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from torch.nn import functional as F


def plot_loss(loss_array,name):
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(loss_array)
    plt.savefig('loss_'+name)


def weights_init_G(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.002)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_D(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def run_trainer(train_loader, net_WAE, netD, args):

    LAMBDA = args.LAMBDA
    batch_size = args.batchSize

    optimizer_WAE = optim.Adam(net_WAE.parameters(), lr=args.lr,betas=(args.beta1,0.999))
    optimizerD = optim.Adam(netD.parameters(), lr=args.lr,betas=(args.beta1,0.999))

    WAE_scheduler = StepLR(optimizer_WAE, step_size=1000, gamma=0.8)
    D_scheduler = StepLR(optimizerD, step_size=1000, gamma=0.8)

    real_label = 1
    fake_label = 0

    device = torch.device("cuda:0" if args.cuda else "cpu")

    if args.restart == '':
        net_WAE.apply(weights_init_G)
        netD.apply(weights_init_D)


    else:
        netD = torch.load('./D_model.pt')
        net_WAE = torch.load('./WAE_model.pt')
        
    criterion_MSE = nn.MSELoss()
    criterion_cross_entropy = nn.BCELoss()
    
    if args.cuda:
        criterion_MSE = criterion_MSE.cuda()
        criterion_cross_entropy = criterion_cross_entropy.cuda()

    for epoch in range(1000):

        data_iter = iter(train_loader)
        i = 0

        for i, (images, labels) in enumerate(train_loader):
            images = Variable(images)
            images = images.cuda()
            label = torch.full((batch_size,), real_label, device=device)

            #train netD
            for p in netD.parameters():
                p.requires_grad = True

            for p in net_WAE.parameters():
                p.requires_grad = False

            netD.zero_grad()

            #train fake noise
            x, z = net_WAE(images)
            label.fill_(fake_label)

            D_fake = netD(z)
            errD_fake = criterion_cross_entropy(D_fake, label)
            errD_fake.backward(retain_graph=True)
            
            #train real - this is gaussian noise
            if args.cuda:
                noise = torch.cuda.FloatTensor(z.size()).normal_(0,1)
            else:
                noise = torch.FloatTensor(z.size()).normal_(0,1)

            noise = Variable(noise)
            label.fill_(real_label)
            D_real = netD(noise)
            errD_real = criterion_cross_entropy(D_real, label)
            
            errD_real.backward()
            
            optimizerD.step()
            #D_scheduler.step()

            #train netE, netG
            for p in netD.parameters():
                p.requires_grad = False

            for p in net_WAE.parameters():
                p.requires_grad = True

            net_WAE.zero_grad()

            #reconstruction term 
            recon, z = net_WAE(images)
            
            recon_loss = criterion_MSE(recon, images)
            
            label.fill_(real_label)
            adv_term = netD(z)
            adv_loss = criterion_cross_entropy(adv_term, label)
            adv_loss = -args.LAMBDA * adv_loss

            recon_loss.backward(retain_graph=True)
            adv_loss.backward()
            optimizer_WAE.step()
            
            if  i % 100 == 0 :
                print('saving images for batch', i)
                save_image(recon.squeeze().data.cpu().detach(), './fake.png')
                save_image(images.data.cpu().detach(), './real.png')

            if i % 100 == 0:
                torch.save(net_WAE, './WAE_model.pt')
                torch.save(netD, './D_model.pt')
                
                print('%d [%d/%d] Loss_D (real/fake) [%.4f/%.4f] Loss WAE (recon/adv) [%.4f/%.4f]'%
                      (epoch, i, len(train_loader), errD_real, errD_fake, 
                       recon_loss, adv_loss))




     
            
