import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.normal import Normal

from train_utils import test

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
im_size = 64
n_classes = 7

################################################################################

def logsumexp_grad_1(net, x0):
    net.eval()
    x = x0.clone().detach().requires_grad_(True)
    out = torch.logsumexp(net(x), dim=1).sum()
    out.backward()
    return x.grad.detach()

def sample_from_model2(net, start_x, n_steps, step_size, noise_std):
    net.eval()
    x_t = start_x
    for t in range(n_steps):
        x_t += step_size * logsumexp_grad(net, x_t) + noise_std * torch.randn_like(x_t)
        x_t = x_t.detach()
    net.train()
    return x_t

def sample_from_model_aa(net, start_x, n_steps, step_size, noise_std):
    net.eval()
    start_x = torch.atanh(torch.clip(start_x, -0.9999, 0.9999))
    x_k = start_x.clone().detach().requires_grad_(True)
    for t in range(n_steps):
        f_prime = torch.autograd.grad(torch.logsumexp(net(torch.tanh(x_k)), dim=1).sum(), [x_k], retain_graph=False)[0]
        x_k.data += step_size * f_prime + noise_std * torch.randn_like(x_k)
    net.train()
    return torch.tanh(x_k).detach()

def total_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

#def init_random_0(bs):
#    classes = np.random.randint(n_classes, size=bs)
#    return means[classes].view(bs, 3, im_size, im_size) + torch.randn(bs, 3, im_size, im_size) * 0.1
        
################################################################################

conditionals = []
means = torch.zeros([n_classes, 3 * im_size * im_size]).to(device)
covs = torch.zeros([n_classes, 3 * im_size * im_size, 3 * im_size * im_size]).to(device)

def init_conditionals(data_loader):
    global conditionals 
    global means
    global covs 
    
    cnts = torch.tensor([0.0] * 7).to(device)
    
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        means.index_add_(0, y, x.view(x.shape[0], -1))
        cnts.index_add_(0, y, torch.ones(x.shape[0]).to(device))
    means /= cnts[:, None]
    
    #for i in range(n_classes):
    #    covs[i] = torch.eye(3 * im_size * im_size) * 0.1
    
    idx = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        if idx % 10 == 0:
            print(idx)
        idx += 1
        for i in range(len(y)):
            covs[y[i]] += (x[i].view(1, -1) - means[y[i]].view(1, -1)).T @ (x[i].view(1, -1) - means[y[i]].view(1, -1))
    covs /= cnts[:, None, None]
    
    covs = covs.to('cpu')
    means = means.to('cpu')
    
    for i in range(n_classes):
        dist = MultivariateNormal(means[i].to(device), 
                                  covariance_matrix=covs[i].to(device) + 1e-4 * torch.eye(3 * im_size * im_size).to(device))
        conditionals.append(dist)
        
def init_random(bs):
    global conditionals
    size = [3, im_size, im_size]
    new = torch.zeros(bs, 3, im_size, im_size)
    for i in range(bs):
        index = np.random.randint(n_classes)
        dist = conditionals[index]
        new[i] = dist.sample().view(size)
    return new
    
def random_fill_buffer(replay_buffer):
    assert replay_buffer.shape[0] % n_classes == 0
    len_batch = replay_buffer.shape[0] // n_classes
    size = [3, im_size, im_size]
    
    for i in range(n_classes):
        replay_buffer[i * len_batch:(i + 1) * len_batch] = conditionals[i].sample_n(len_batch).view(len_batch, *size)
        

def fill_buffer_from_data(replay_buffer, data_loader):
    i = 0
    for batch in data_loader:
        replay_buffer[i:i+len(batch[0])] = batch[0]
        i += len(batch[0])
        if i >= len(replay_buffer):
            break

def sample_from_model(net, start_x, n_steps, step_size, noise_std):
    net.eval()
    x_k = start_x.clone().detach().requires_grad_(True)
    #grad_norms = 0
    for t in range(n_steps):
        f_prime = torch.autograd.grad(torch.logsumexp(net(x_k), dim=1).sum(), [x_k], retain_graph=False)[0]
        x_k.data += step_size * f_prime + noise_std * torch.randn_like(x_k)
        #grad_norms += f_prime.data.norm(2)
    #print(f"grad norm: {grad_norms / n_steps}")
    net.train()
    return x_k.detach()

def ebm_loss(net, x, sample):
    return -torch.logsumexp(net(x), dim=1).mean() + torch.logsumexp(net(sample), dim=1).mean()

def train_epoch_ebm(net, optimizer, replay_buffer, train_loader_xy, train_loader_x, train_loader_buffer,
                    args, writers, epoch):
    loss_log = []
    net.train()
    
    itr_unlabeled = iter(train_loader_x)
    itr_buffer = iter(train_loader_buffer)
    
    for i, (x_labeled, y) in zip(range(len(train_loader_xy)), train_loader_xy):
        print(i)
        x_labeled = x_labeled.to(device)
        y = y.to(device)
        x_unlabeled = next(itr_unlabeled)[0].to(device)
        x_buffer = next(itr_buffer)[0].to(device)
        
        inds = torch.randint(0, len(replay_buffer), (len(x_unlabeled),))
        buffer_samples = replay_buffer[inds].to(device)
        
        if args.noise_init == "mixture":
            random_samples = init_random(len(x_unlabeled)).to(device)
        elif args.noise_init == "data":
            random_samples = x_buffer

        choose_random = (torch.rand(len(x_unlabeled)) < args.reinit_freq).float()[:, None, None, None].to(device)
        start_x = choose_random * random_samples + (1 - choose_random) * buffer_samples
        
        sample = sample_from_model(net, start_x, args.n_steps, args.step_size, args.noise_std)
        replay_buffer[inds] = sample.to('cpu')
        
        
        pred_y = net(x_labeled)
        ce_loss = F.cross_entropy(pred_y, y)
        gen_loss = ebm_loss(net, x_unlabeled, sample)
        loss = ce_loss + gen_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        loss = loss.data.cpu().item()
        loss_log.append(loss)
        
        _, predicted = torch.max(pred_y.data, 1)
        total = y.size(0)
        correct = (predicted == y).sum().item()
        
        print(f"cross entropy: {ce_loss.data.cpu().item()}, ebm loss: {gen_loss.data.cpu().item()}")
        print("accuracy:", correct / total * 100.0)
        
        
        if writers:
            step = epoch * len(train_loader_xy) + i
            
            if step % 5 == 0:
                writers[0].add_image('batch_sample', 
                                     torch.clip((torchvision.utils.make_grid(x_unlabeled[:8]) + 1)/2, 0.0, 1.0), step)
                writers[0].add_image('ebm_sample', 
                                     torch.clip((torchvision.utils.make_grid(sample[:8]) + 1)/2, 0.0, 1.0), step)
                writers[0].add_image('zzz_sample', 
                                     torch.clip((torchvision.utils.make_grid(random_samples[:8]) + 1)/2, 0.0, 1.0), step)
        
            writers[0].add_scalar('batch cross-entropy', ce_loss.data.cpu().item(), step)
            
            writers[0].add_scalar('batch ebm loss', gen_loss.data.cpu().item(), step)
            
            writers[0].add_scalar('batch accuracy', correct / total * 100.0, step)
        
    return loss_log

def train_ebm(net, optimizer, scheduler, n_epochs, replay_buffer, train_loader_xy, train_loader_x, train_loader_buffer,
              val_loader, test_loader, args, writers, save_path=None):
    for epoch in range(n_epochs):
        train_loss = train_epoch_ebm(net, optimizer, replay_buffer, train_loader_xy, train_loader_x, train_loader_buffer,
                                     args, writers, epoch)
        val_loss, val_acc = test(net, val_loader)
        test_loss, test_acc = test(net, test_loader)
        
        print(f"Epoch {epoch}, val loss: {val_loss}, val acc: {val_acc}, test loss: {test_loss}, test acc: {test_acc}")
        print()
        
        if writers is not None:
            tr_writer, val_writer, test_writer = writers
            val_writer.add_scalar('cross-entropy loss', val_loss, epoch)
            test_writer.add_scalar('cross-entropy loss', test_loss, epoch)
            
            val_writer.add_scalar('accuracy', val_acc, epoch)
            test_writer.add_scalar('accuracy', test_acc, epoch)
        
        if scheduler is not None:
            scheduler.step() 
             
    if save_path is None and writers is not None:
        wr_dir = writers[0].get_logdir()
        save_path = 'models/' + wr_dir[wr_dir.rfind('/')+1:] + ".pt"
        print(save_path)
    if save_path is not None:
        torch.save(net.state_dict(), save_path)
        
        


