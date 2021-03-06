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
    for t in range(n_steps):
        f_prime = torch.autograd.grad(torch.logsumexp(net(x_k), dim=1).sum(), [x_k], retain_graph=False)[0]
        x_k.data += step_size * f_prime + noise_std * torch.randn_like(x_k)
    net.train()
    return x_k.detach()

def ebm_loss(net, x, sample):
    return -torch.logsumexp(net(x), dim=1).mean() + torch.logsumexp(net(sample), dim=1).mean()

def train_epoch_ebm(net, optimizer, replay_buffer, train_loader_xy, train_loader_x, train_loader_buffer,
                    args, writers, epoch, teacher_net=None):
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
        elif args.noise_init == "uniform":
            random_samples = (torch.rand(*x_unlabeled.shape) * 2 - 1).to(device)

        choose_random = (torch.rand(len(x_unlabeled)) < args.reinit_freq).float()[:, None, None, None].to(device)
        start_x = choose_random * random_samples + (1 - choose_random) * buffer_samples
        
        sample = sample_from_model(net, start_x, args.n_steps, args.step_size, args.noise_std)
        replay_buffer[inds] = sample.to('cpu')
        
        
        pred_y = net(x_labeled)
        
        if teacher_net is not None:
            T = 5.0
            with torch.no_grad():
                teacher_output = teacher_net(x_labeled)
            ce_loss = F.kl_div(F.log_softmax(pred_y / T, dim=1), 
                               F.softmax(teacher_output / T, dim=1), 
                               reduction='batchmean') * T * T
        else:
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
              val_loader, test_loader, args, writers, save_path=None, teacher_net=None):
    for epoch in range(n_epochs):
        train_loss = train_epoch_ebm(net, optimizer, replay_buffer, train_loader_xy, train_loader_x, train_loader_buffer,
                                     args, writers, epoch, teacher_net=teacher_net)
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
        
        


