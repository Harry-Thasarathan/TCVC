from __future__ import print_function
import argparse
import os
import numpy as np
from math import log10
from os.path import join
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from src.models import define_G, define_D, print_network
from src.util import Progbar, stitch_images, postprocess, load
from src.data import get_training_set, get_test_set, create_iterator
from src.dataset import DatasetFromFolder
from src.loss import AdversarialLoss, StyleLoss, PerceptualLoss
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms


os.environ['CUDA_VISIBLE_DEVICES'] = '0'


# Training settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--root', required=True, help='facades')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--logfile', required=False, help='trainlogs.dat')
parser.add_argument('--checkpoint_path_G', required=False, help='load pre-trained Generator?')
parser.add_argument('--checkpoint_path_D', required=False, help='load pre-trained Discriminator?')
parser.add_argument('--batchSize', type=int, default=16, help='training batch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--nEpochs', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--input_nc', type=int, default=1, help='input image channels')
parser.add_argument('--output_nc', type=int, default=3, help='output image channels')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.002')
parser.add_argument('--beta1', type=float, default=0, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--L1lamb', type=int, default=10, help='weight on L1 term in objective')
parser.add_argument('--Stylelamb', type=int, default=1000, help='weight on Style term in objective')
parser.add_argument('--Contentlamb', type=int, default=1, help='weight on Content term in objective')
parser.add_argument('--Adversariallamb', type=int, default=0.1, help='weight on Adv term in objective')
opt = parser.parse_args()

print(opt)

if opt.cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

cudnn.benchmark = True

torch.manual_seed(opt.seed)
if opt.cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')
root_path = opt.root
train_set = get_training_set(join(root_path , opt.dataset))
test_set = get_test_set(join(root_path , opt.dataset))

training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

sample_iterator = create_iterator(6, test_set)

print('===> Building model')
netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, False, [0])
netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, False, [0])

if opt.checkpoint_path_G and opt.checkpoint_path_D:
    load(opt.checkpoint_path_G, opt.checkpoint_path_D, netG, netD)

criterionGAN = AdversarialLoss()
criterionSTYLE = StyleLoss()
criterionCONTENT = PerceptualLoss()
criterionL1 = nn.L1Loss()
criterionMSE = nn.MSELoss()

# setup optimizer
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr * 0.1, betas=(opt.beta1, 0.999))

print('---------- Networks initialized -------------')
print_network(netG)
print_network(netD)
print('-----------------------------------------------')

real_a = torch.FloatTensor(opt.batchSize, opt.input_nc, 256, 256)
real_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)
prev_b = torch.FloatTensor(opt.batchSize, opt.output_nc, 256, 256)

if opt.cuda:
    netD = netD.cuda()
    netG = netG.cuda()
    criterionGAN = criterionGAN.cuda()
    criterionL1 = criterionL1.cuda()
    critertionSTYLE = criterionSTYLE.cuda()
    criterionCONTENT = criterionCONTENT.cuda()
    criterionMSE = criterionMSE.cuda()
    real_a = real_a.cuda()
    real_b = real_b.cuda()
    prev_b = prev_b.cuda()

real_a = Variable(real_a)
real_b = Variable(real_b)
prev_b = Variable(prev_b)

def train(epoch):

    for iteration, batch in enumerate(training_data_loader, 1):
        # forward
        real_a_cpu, real_b_cpu, prev_b_cpu = batch[0], batch[1], batch[2]
        real_a.data.resize_(real_a_cpu.size()).copy_(real_a_cpu)
        real_b.data.resize_(real_b_cpu.size()).copy_(real_b_cpu)
        prev_b.data.resize_(prev_b_cpu.size()).copy_(prev_b_cpu)

        input_joined = torch.cat((real_a,prev_b),1)

        fake_b = netG(input_joined)

        ############################
        # (1) Update D network: maximize log(D(x,y)) + log(1 - D(x,G(x)))
        ###########################

        optimizerD.zero_grad()

        # train with fake
        fake_ab = torch.cat((real_a, prev_b, fake_b), 1)
        pred_fake = netD.forward(fake_ab.detach())
        loss_d_fake = criterionGAN(pred_fake,False,True)

        # train with real
        real_ab = torch.cat((real_a, prev_b, real_b), 1)
        pred_real = netD.forward(real_ab)
        loss_d_real = criterionGAN(pred_real, True, True) 


        # Combined loss
        loss_d = (loss_d_fake + loss_d_real) * 0.5

        loss_d.backward()

        #Discriminator parameters update every 12 iterations 
        if (iteration == 1 or iteration % 12 == 0):
            optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(x,G(x))) + L1(y,G(x))
        ##########################
        optimizerG.zero_grad()

        # First, G(A) should fake the discriminator
        fake_ab = torch.cat((real_a, prev_b,fake_b), 1)
        pred_fake = netD.forward(fake_ab)
        loss_g_gan = criterionGAN(pred_fake, True, False)

        # Second, G(A) = B
        loss_g_l1 = criterionL1(fake_b, real_b) * opt.L1lamb
        loss_g = loss_g_gan + loss_g_l1

        loss_g_style = criterionSTYLE(fake_b,real_b) * opt.Stylelamb
        loss_g = loss_g + loss_g_style

        loss_g_content = criterionCONTENT(fake_b,real_b) * opt.Contentlamb
        loss_g = loss_g + loss_g_content

        loss_g.backward()

        optimizerG.step()

        if (iteration % 25 == 0):
            logs = [("epoc", epoch),("iter", iteration),("Loss_G", loss_g.item()),("Loss_D", loss_d.item()), ("Loss_G_adv",loss_g_gan.item()),("Loss_G_L1",loss_g_l1.item()),("Loss_G_style",loss_g_style.item()),("Loss_G_content",loss_g_content.item()),("Loss_D_Real",loss_d_real.item()),("Loss_D_Fake",loss_d_fake.item())]
            log_train_data(logs)

        if(iteration % 250 == 0):
            sample(iteration)


        print("===> Epoch[{}]({}/{}): Loss_D: {:.4f} Loss_G: {:.4f} LossD_Fake: {:.4f} LossD_Real: {:.4f}  LossG_Adv: {:.4f} LossG_L1: {:.4f} LossG_Style {:.4f} LossG_Content {:.4f}".format(
           epoch, iteration, len(training_data_loader), loss_d, loss_g, loss_d_fake, loss_d_real, loss_g_gan, loss_g_l1, loss_g_style, loss_g_content))

def sample(iteration):
    with torch.no_grad():

        input,target,prev_frame = next(sample_iterator)
        
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            prev_frame = prev_frame.cuda()

        pred_input = torch.cat((input,prev_frame),1)
        prediction = netG(pred_input)
        prediction = postprocess(prediction)
        input = postprocess(input)
        target = postprocess(target)

    img = stitch_images(input, target, prediction)
    samples_dir = root_path + "/samples"

    if not os.path.exists(samples_dir):
        os.makedirs(samples_dir)

    sample = opt.dataset + "_" + str(epoch) + "_" + str(iteration).zfill(5) + ".png"
    print('\nsaving sample ' + sample + ' - learning rate: ' + str(opt.lr))
    img.save(os.path.join(samples_dir, sample))

def log_train_data(loginfo):
    log_dir = root_path + "/logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_file = log_dir + "/" + opt.logfile
    with open(log_file, 'a') as f:
        f.write('%s\n' % ' '.join([str(item[1]) for item in loginfo]))


def checkpoint(epoch):
    if not os.path.exists("checkpoint"):
        os.mkdir("checkpoint")
    if not os.path.exists(os.path.join("checkpoint", opt.dataset)):
        os.mkdir(os.path.join("checkpoint", opt.dataset))
    net_g_model_out_path = "checkpoint/{}/netG_weights_epoch_{}.pth".format(opt.dataset, epoch)
    net_d_model_out_path = "checkpoint/{}/netD_weights_epoch_{}.pth".format(opt.dataset, epoch)
    
    torch.save({'generator': netG.state_dict()}, net_g_model_out_path)
    torch.save({'discriminator': netD.state_dict()}, net_d_model_out_path)
    
    print("Checkpoint saved to {}".format("checkpoint" + opt.dataset))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    checkpoint(epoch)


def run():
    torch.multiprocessing.freeze_support()
    print('loop')

if __name__ == '__main__':
    run()
