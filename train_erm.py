import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

from data import get_data_erm
from train_utils import train_erm

import timm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer_name = "runs/pacs_erm_50gn_mar25_64_train.pt"
tr_writer = SummaryWriter(writer_name + "_train")
val_writer = SummaryWriter(writer_name + "_val")
test_writer = SummaryWriter(writer_name + "_test")

train_loader, val_loader, test_loader = get_data_erm("art_painting", im_size=64)

net = timm.create_model('resnet50_gn', pretrained=True)
net.fc = nn.Linear(2048, 7)
#net = torchvision.models.resnet34(pretrained=True)
#net.fc = nn.Linear(512, 7)
net = net.to(device)

#for i in range(1, 5):
#    net.layer3[i] = nn.Identity()

epochs = 100

#opt = optim.Adam(net.parameters(), lr=0.0001, weight_decay=1e-4)
opt = optim.SGD(net.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6)

train_erm(net, opt, epochs, train_loader, val_loader, test_loader, scheduler=scheduler, writers=(tr_writer, val_writer, test_writer))
