import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import numpy as np

from data import batch_size

criterion = nn.CrossEntropyLoss()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def test(model, loader):
    loss_log = []
    model.eval()
    
    correct = 0
    total = 0
    
    for batch_num, batch in enumerate(loader):   
        data = batch[0].to(device)
        target = batch[1].to(device)
        
        with torch.no_grad():
            output = model(data)
            
        loss = F.cross_entropy(output, target)        
        loss = loss.data.cpu().item()
        loss_log.append(loss)
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
    return np.mean(loss_log), 100 * correct / total

##########################################################################################

def train_epoch_erm(model, optimizer, train_loader):
    loss_log = []
    model.train()
    for _, (data, target) in zip(range(len(train_loader)), train_loader):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        output = model(data)        
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss = loss.data.cpu().item()
        loss_log.append(loss)
    return loss_log   
    
def train_erm(model, opt, n_epochs, train_loader, val_loader, test_loader, scheduler=None, writers=None, save_path=None):
    for epoch in range(n_epochs):
        train_loss = train_epoch_erm(model, opt, train_loader)
        val_loss, val_acc = test(model, val_loader)
        test_loss, test_acc = test(model, test_loader)
        
        print(f"Epoch {epoch}, train loss: {np.mean(train_loss)}, val loss: {val_loss}, val acc: {val_acc}")
        print(f"test loss: {test_loss}, test acc: {test_acc}")
        if ((epoch + 1) % 5 == 0):
            tr_loss, tr_acc = test(model, train_loader)
            print(f"real train loss: {tr_loss}, train acc: {tr_acc}")
        print()
        
        if writers is not None:
            tr_writer, val_writer, test_writer = writers
            tr_writer.add_scalar('cross-entropy loss', np.mean(train_loss), epoch)
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
        torch.save(model.state_dict(), save_path)
    
##########################################################################################

def mixup(x, y, alpha):
    lmbd = np.random.beta(alpha, alpha)

    index = torch.randperm(x.size()[0])

    mixed = lmbd * x + (1 - lmbd) * x[index, :]
    y1, y2 = y, y[index]
    return mixed, y1, y2, lmbd
    
def train_epoch_distil(student, teacher, optimizer, train_loader, T, alpha, mixup_alpha, writers):
    loss_log = []
    student.train()
    teacher.eval()
    
    for _, (data, target) in zip(range(len(train_loader)), train_loader):
        data = data.to(device)
        target = target.to(device)
        
        data, target1, target2, lmbd = mixup(data, target, mixup_alpha)
        
        if writers is not None:
            writers[0].add_image('mixup_images', (torchvision.utils.make_grid(data[:4]) + 1)/2)
        
        with torch.no_grad():
            teacher_output = teacher(data)
        
        optimizer.zero_grad()
        
        student_output = student(data)        
        cls_loss = lmbd * F.cross_entropy(student_output, target1) + (1 - lmbd) * F.cross_entropy(student_output, target2)

        dist_loss = F.kl_div(F.log_softmax(student_output / T, dim=1), 
                             F.softmax(teacher_output / T, dim=1), 
                             reduction='batchmean') * T * T
        
        loss = alpha * cls_loss + (1 - alpha) * dist_loss
        
        loss.backward()
        optimizer.step()
        loss = loss.data.cpu().item()
        loss_log.append(loss)
        
    return loss_log   

def train_distil(student, teacher, opt, scheduler, n_epochs, train_loader, val_loader, test_loader, 
                 T=10, alpha=0.0, mixup_alpha=0.0, writers=None, save_path=None):
    for epoch in range(n_epochs):
        train_loss = train_epoch_distil(student, teacher, opt, train_loader, T, alpha, mixup_alpha, writers)
        
        val_loss, val_acc = test(student, val_loader)
        test_loss, test_acc = test(student, test_loader)
        
        print(f"Epoch {epoch}, train distillation loss: {np.mean(train_loss)}")
        print(f"val ce loss: {val_loss}, val acc: {val_acc}")
        print(f"test ce loss: {test_loss}, test acc: {test_acc}")
        
        print()
        
        if writers is not None:
            tr_writer, val_writer, test_writer = writers
            tr_writer.add_scalar('distillation loss', np.mean(train_loss), epoch)
            
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
        torch.save(student.state_dict(), save_path)
    
##########################################################################################
    
def train_epoch_distil_noisy(student, teacher, optimizer, train_loader, T, alpha):
    loss_log = []
    student.train()
    
    for _, (data, target, teacher_output) in zip(range(len(train_loader)), train_loader):
        data = data.to(device)
        teacher_output = teacher_output.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        student_output = student(data)        
        cls_loss = F.cross_entropy(student_output, target)

        dist_loss = F.kl_div(F.log_softmax(student_output / T, dim=1), 
                                    F.softmax(teacher_output / T, dim=1), 
                                    reduction='batchmean') * T * T
        
        loss = alpha * cls_loss + (1 - alpha) * dist_loss
        
        loss.backward()
        optimizer.step()
        loss = loss.data.cpu().item()
        loss_log.append(loss)
        
    return loss_log   


def train_distil_noisy(student, teacher, opt, scheduler, n_epochs, train_loader, val_loader, test_loader, 
                 T=10, alpha=0.0, writers=None, save_path=None):
    for epoch in range(n_epochs):
        train_loss = train_epoch_distil_noisy(student, teacher, opt, train_loader, T=T, alpha=alpha)
        
        val_loss, val_acc = test(student, val_loader)
        test_loss, test_acc = test(student, test_loader)
        
        print(f"Epoch {epoch}, train distillation loss: {np.mean(train_loss)}")
        print(f"val ce loss: {val_loss}, val acc: {val_acc}")
        print(f"test ce loss: {test_loss}, test acc: {test_acc}")
        
        print()
        
        if writers is not None:
            tr_writer, val_writer, test_writer = writers
            tr_writer.add_scalar('distillation loss', np.mean(train_loss), epoch)
            
            val_writer.add_scalar('cross-entropy loss', val_loss, epoch)
            test_writer.add_scalar('cross-entropy loss', test_loss, epoch)
            
            val_writer.add_scalar('accuracy', val_acc, epoch)
            test_writer.add_scalar('accuracy', test_acc, epoch)
        
        scheduler.step()
        
    if save_path is None and writers is not None:
        wr_dir = writers[0].get_logdir()
        save_path = 'models/' + wr_dir[wr_dir.rfind('/')+1:] + ".pt"
        print(save_path)
    if save_path is not None:
        torch.save(student.state_dict(), save_path)
