import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
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
from ebm import init_random, sample_from_model

parser = argparse.ArgumentParser("Sample from JEM")

parser.add_argument("--buffer_size", type=int, default=1344)
parser.add_argument("--reinit_freq", type=float, default=0.1)
parser.add_argument("--n_steps", type=int, default=30)
parser.add_argument("--step_size", type=float, default=1.0)
parser.add_argument("--noise_std", type=float, default=1e-2)
parser.add_argument("--sample_steps", type=int, default=300)
parser.add_argument("--batch_size", type=int, default=100)
parser.add_argument("--save_dir", type=str, default="a")
parser.add_argument("--model_ckpt", type=str, default="a")
args = parser.parse_args()


device = "cuda"
net = timm.create_model('resnet50_gn', pretrained=True)
net.fc = nn.Linear(2048, 7)
net.load_state_dict(torch.load(args.model_ckpt))
net = net.to(device)


train_loader, _, _, _, _ = get_data_ebm("pacs", "art_painting", im_size=64)
init_conditionals(train_loader)


replay_buffer = torch.zeros(args.buffer_size, 3, im_size, im_size).to(device)
random_fill_buffer(replay_buffer)

for i in range(args.sample_steps):
    print(i)
    inds = torch.randint(0, len(replay_buffer), (args.batch_size,))
    buffer_samples = replay_buffer[inds]
    
    random_samples = init_random(args.batch_size).to(device)
    
    choose_random = (torch.rand(args.batch_size) < args.reinit_freq).float()[:, None, None, None].to(device)
    start_x = choose_random * random_samples + (1 - choose_random) * buffer_samples
    
    sample = sample_from_model(net, start_x, args.n_steps, args.step_size, args.noise_std)
    replay_buffer[inds] = sample
    
for i in range(len(replay_buffer)):
    img = torch.clip((replay_buffer[i] + 1) / 2, 0.0, 1.0)
    save_image(img, args.save_dir + f'/{i}.png')