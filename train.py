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
import argparse
import timm
from wideresnet import WRN
import resnet

from data import get_data_erm, get_data_distil
from train_utils import train_erm, train_distil

parser = argparse.ArgumentParser("DG with ERM and Distillation")
parser.add_argument("--type", choices=["erm", "distil"], default="erm")
parser.add_argument("--model", choices=["resnet50", "resnet34", "resnet18", "bit50", "wrn28_10"], default="resnet18")
parser.add_argument("--teacher", choices=["resnet50", "resnet34", "resnet18", "bit50", "wrn28_10"], default="resnet18")
parser.add_argument("--teacher_ckpt", type=str, default="a")
parser.add_argument("--optimizer", choices=["adam", "sgd"], default="sgd")
parser.add_argument("--scheduler", choices=["none", "cosine"], default="cosine")
parser.add_argument("--lr", type=float, default=0.01)
parser.add_argument("--wd", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--n_epochs", type=int, default=300)
parser.add_argument("--writer_name", type=str, default="a")
parser.add_argument("--target_domain", type=str, default="art_painting")
parser.add_argument("--dataset", type=str, choices=["pacs", "cifar10-c"], default="pacs")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

writer_name = "runs/" + args.writer_name
tr_writer = SummaryWriter(writer_name + "_train")
val_writer = SummaryWriter(writer_name + "_val")
test_writer = SummaryWriter(writer_name + "_test")
writers=(tr_writer, val_writer, test_writer)
if args.writer_name == "a":
    writers = None
    
im_size = 227 if args.dataset == "pacs" else 32

def get_model(name, args):
    pretrain = True if args.dataset == "pacs" else False
    n_classes = 7 if args.dataset == "pacs" else 10
    if name == "resnet18":
        if args.dataset == "pacs":
            net = torchvision.models.resnet18(pretrained=pretrain)
            net.fc = nn.Linear(512, n_classes)
        else:
            net = resnet.ResNet18()
    elif name == "resnet34":
        if args.dataset == "pacs":
            net = torchvision.models.resnet34(pretrained=pretrain)
            net.fc = nn.Linear(512, n_classes)
        else:
            net = resnet.ResNet34()
    elif name == "resnet50":
        net = torchvision.models.resnet50(pretrained=pretrain)
        net.fc = nn.Linear(2048, n_classes)
    elif name == "bit50":
        net = timm.create_model('resnetv2_50x1_bitm', pretrained=pretrain, num_classes=n_classes)
    elif name == "wrn28_10":
        net = WRN(depth=28, width=10, n_classes=n_classes, im_sz=im_size)
    return net

def get_optimizer_and_scheduler(params, args):
    if args.optimizer == "adam":
        optimizer = optim.Adam(params, lr=args.lr, weight_decay=args.wd)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(params, lr=args.lr, weight_decay=args.wd, momentum=args.momentum)

    if args.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs, eta_min=1e-6)
    elif args.scheduler == "none":
        scheduler = None
        
    return optimizer, scheduler

if args.type == "erm":
    train_loader, val_loader, test_loader = get_data_erm(args.dataset, args.target_domain, im_size=im_size)
    
    net = get_model(args.model, args).to(device)
    
    optimizer, scheduler = get_optimizer_and_scheduler(net.parameters(), args)

    train_erm(net, optimizer, args.n_epochs, train_loader, val_loader, test_loader, scheduler=scheduler, 
              writers=(tr_writer, val_writer, test_writer))

elif args.type == "distil":
    train_loader, val_loader, test_loader = get_data_distil(args.dataset, args.target_domain, im_size=im_size)
    
    teacher_net = get_model(args.teacher, args).to(device)
    teacher_net.load_state_dict(torch.load(args.teacher_ckpt))
    
    student_net = get_model(args.model, args).to(device)
    
    optimizer, scheduler = get_optimizer_and_scheduler(student_net.parameters(), args)
    
    train_distil(student_net, teacher_net, optimizer, scheduler, args.n_epochs, train_loader, val_loader, test_loader, 
                 T=5, mixup_alpha=1.0, writers=writers)
