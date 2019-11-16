from __future__ import print_function
from PIL import Image
from skimage import io, feature, color, util
from src.models import InpaintGenerator
import argparse
import os
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, SequentialSampler
import torchvision.transforms as transforms
from torchvision.utils import save_image
from src.data import get_training_set, get_test_set, get_val_set, create_iterator
from src.dataset import DatasetFromFolder
from src.util import is_image_file, load_img,save_img

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--root', required=True, help='root path')
parser.add_argument('--dataset', required=True, help='facades')
parser.add_argument('--model', type=str, default='checkpoint/facades/netG_model_epoch_200.pth', help='model file to use')
parser.add_argument('--cuda', action='store_true', help='use cuda')
opt = parser.parse_args()
print(opt)

root_path = opt.root
val_set = get_val_set(os.path.join(root_path , opt.dataset))

seq_sampler = SequentialSampler(val_set)

val_data_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False,sampler = seq_sampler)

checkpoint = torch.load(opt.model)
netG = InpaintGenerator()
netG.load_state_dict(checkpoint['generator'])
netG.cuda()

transform_list = [transforms.ToTensor(),
                  transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]

transform = transforms.Compose(transform_list)

counter = 0

with torch.no_grad():
    for batch in val_data_loader:
        input, target, prev_frame = Variable(batch[0], volatile=True), Variable(batch[1], volatile=True), Variable(batch[2], volatile=True)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
            prev_frame = prev_frame.cuda()
        if counter != 0:
            prev_frame = tmp
            print("success")
        pred_input = torch.cat((input,prev_frame),1)
        out = netG(pred_input)
        tmp = out
        
        if not os.path.exists(os.path.join("result", opt.dataset)):
            os.makedirs(os.path.join("result", opt.dataset))
       
        image_name = opt.dataset + "_" + str(counter).zfill(5) + ".jpg"
        save_image(out,"result/{}/{}".format(opt.dataset, image_name))
        print("saving:"+image_name)
        
        counter += 1