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

from data import get_data_distil
from train_utils import train_distil

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

tr_writer = SummaryWriter('runs/pacs_distil_pt_mr4_dropped_train')
val_writer = SummaryWriter('runs/pacs_distil_pt_mr4_dropped_val')
test_writer = SummaryWriter('runs/pacs_distil_pt_mr4_dropped_test')

train_loader, val_loader, test_loader = get_data_distil("art_painting")

teacher_net = torchvision.models.resnet34(pretrained=True)
teacher_net.fc = nn.Linear(512, 7)
teacher_net = teacher_net.to(device)
#teacher_net.load_state_dict(torch.load("models/pacs_erm34_feb22_34_full_train.pt"))
teacher_net.load_state_dict(torch.load("models/pacs_erm34_feb22_34_01_09_train.pt"))


student_net = torchvision.models.resnet34(pretrained=True)
student_net.fc = nn.Linear(512, 7)
student_net = student_net.to(device)

for i in range(1, 5):
    student_net.layer3[i] = nn.Identity()

#for layer in [student_net.layer1, student_net.layer2, student_net.layer3, student_net.layer4]:
#    layer[1] = torch.nn.Identity()
#    layer[2] = torch.nn.Identity()

epochs = 500

opt = optim.SGD(student_net.parameters(), lr=0.02, momentum=0.9, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=1e-6)

train_distil(student_net, teacher_net, opt, scheduler, epochs, train_loader, val_loader, test_loader, 
             T=5, mixup_alpha=1.0, writers=(tr_writer, val_writer, test_writer))
