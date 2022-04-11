import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import timm
import argparse

from data import get_data_ebm
from ebm import train_ebm, random_fill_buffer, fill_buffer_from_data, init_conditionals, im_size, n_classes
from wideresnet import WRN

parser = argparse.ArgumentParser("Energy Based Models")

parser.add_argument("--model", choices=["resnet50_gn", "wrn", "bit50"], default="resnet50_gn")

parser.add_argument("--replay_buffer_size", type=int, default=1344)
parser.add_argument("--reinit_freq", type=float, default=0.1)

parser.add_argument("--optimizer", choices=["adam", "sgd"], default="sgd")
parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
parser.add_argument("--lr", type=float, default=0.005)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--n_epochs", type=int, default=300)

parser.add_argument("--n_steps", type=int, default=30)
parser.add_argument("--step_size", type=float, default=1.0)
parser.add_argument("--noise_std", type=float, default=1e-2)

parser.add_argument("--noise_init", choices=["mixture", "data"], default="mixture")

parser.add_argument("--writer_name", type=str, default="a")

args = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer_name = "runs/" + args.writer_name
tr_writer = SummaryWriter(writer_name + "_train")
val_writer = SummaryWriter(writer_name + "_val")
test_writer = SummaryWriter(writer_name + "_test")

train_loader, train_loader_unlabeled, train_loader_buffer, val_loader, test_loader = get_data_ebm("art_painting", im_size=im_size)

if args.model == "resnet50_gn":
    net = timm.create_model('resnet50_gn', pretrained=True)
    net.fc = nn.Linear(2048, n_classes)
elif args.model == "bit50":
    net = timm.create_model('resnetv2_50x1_bitm', pretrained=True, num_classes=n_classes)
elif args.model == "wrn":
    net = WRN(im_sz=im_size, depth=22, norm='layer')
net = net.to(device)

replay_buffer = torch.zeros(args.replay_buffer_size, 3, im_size, im_size).to(device)

if args.noise_init == "mixture":
    init_conditionals(train_loader)
    random_fill_buffer(replay_buffer)
elif args.noise_init == "data":
    fill_buffer_from_data(replay_buffer, train_loader_buffer)

replay_buffer = replay_buffer.to('cpu')

n_epochs = args.n_epochs
if args.optimizer == "adam":
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.wd)
elif args.optimizer == "sgd":
    optimizer = optim.SGD(net.parameters(), lr=args.lr, weight_decay=args.wd)
    
if args.scheduler == "cosine":
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, n_epochs, eta_min=1e-6)
elif args.scheduler == "none":
    scheduler = None

train_ebm(net, optimizer, scheduler, n_epochs, replay_buffer, train_loader, train_loader_unlabeled, train_loader_buffer,
          val_loader, test_loader, args, writers=(tr_writer, val_writer, test_writer))
