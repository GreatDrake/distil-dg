import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
from PIL import Image
import os
import random
from sklearn.model_selection import train_test_split

batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class CIFAR10C(torchvision.datasets.VisionDataset):
    def __init__(self, root :str, name :str,
                 transform=None, target_transform=None):
        super(CIFAR10C, self).__init__(
            root, transform=transform,
            target_transform=target_transform
        )
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')
        
        self.data = np.load(data_path)
        self.targets = np.load(target_path)
        
    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)
            
        return img, targets
    
    def __len__(self):
        return len(self.data)

def get_datasets_pacs(test_domain, transform_train, im_size=227, seed=54):
    transform_test = transforms.Compose(
        [transforms.Resize(im_size),
         transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    )

    datasets = dict()
    names = ["art_painting", "cartoon", "photo", "sketch"]
    test_name = test_domain

    for name in names:
        transform = transform_train if name != test_name else transform_test
        datasets[name] = torchvision.datasets.ImageFolder(root="kfold/"+name, transform=transform)

    train_dataset = torch.utils.data.ConcatDataset([datasets[name] for name in names if name != test_name])
    
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    train_idx, valid_idx = train_test_split(np.arange(len(train_dataset)), test_size=0.1, shuffle=True, random_state=seed)

    trainset = torch.utils.data.Subset(train_dataset, train_idx)
    valset = torch.utils.data.Subset(train_dataset, valid_idx)
    testset = datasets[test_name]
    
    return trainset, valset, testset

def get_datasets_cifar(transform_train, im_size=227, seed=54):
    transform_test = transforms.Compose(
        [transforms.Resize(im_size),
         transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    )
    
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    
    trainset = torchvision.datasets.CIFAR10(root='./data_cifar', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root='./data_cifar', train=False, download=True, transform=transform_test)
    
    target_domains = []
    
    for name in [s[:s.find(".npy")] for s in os.listdir("CIFAR-10-C") if s != "labels.npy"]:
        #if "noise" not in name:
        #    continue
        #print(name)
        ds = CIFAR10C(root="CIFAR-10-C", name=name, transform=transform_test)
        np.random.seed(hash(name) % 123456789)
        indices = np.random.choice(list(range(len(ds))), size=5000, replace=False)
        target_domains.append(Subset(ds, indices))
    
    testset = torch.utils.data.ConcatDataset(target_domains)
    
    return trainset, valset, testset


def get_dataloaders(ds_name, test_domain, transform_train, im_size=227, seed=54):
    if ds_name == "pacs":
        trainset, valset, testset = get_datasets_pacs(test_domain, transform_train, im_size=im_size, seed=seed)
    if ds_name == "cifar10-c":
        trainset, valset, testset = get_datasets_cifar(transform_train, im_size=im_size, seed=seed)
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    
    return train_loader, val_loader, test_loader



def get_data_erm(ds_name, test_domain, im_size=227, seed=54):
    print(im_size)
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
         #transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + 0.0 * torch.randn_like(x)]
    )
    
    transform_base = transforms.Compose(
        [transforms.Resize(im_size),
         transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    )
    
    train_loader, _, _ = get_dataloaders(ds_name, test_domain, transform_train, im_size=im_size, seed=seed)
    _, val_loader, test_loader = get_dataloaders(ds_name, test_domain, transform_base, im_size=im_size, seed=seed)

    return train_loader, val_loader, test_loader

def get_data_ebm(ds_name, test_domain, im_size=64, seed=54):
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(im_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + 0.05 * torch.randn_like(x)]
    )
    
    train_loader, val_loader, test_loader = get_dataloaders(ds_name, test_domain, transform_train, im_size=im_size, seed=seed)
    train_loader_unlabeled, _, _ = get_dataloaders(ds_name, test_domain, transform_train, im_size=im_size, seed=seed)
    
    transform_buffer = transforms.Compose([transform_train, lambda x: x + 0.15 * torch.randn_like(x)])
    train_loader_buffer, _, _ = get_dataloaders(ds_name, test_domain, transform_buffer, im_size=im_size, seed=seed)
    
    return train_loader, train_loader_unlabeled, train_loader_buffer, val_loader, test_loader


def get_data_distil(ds_name, test_domain):
    transform_train = transforms.Compose(
        [transforms.ColorJitter(brightness=(0.6, 1.5), contrast=(0.6, 1.5), saturation=(0.6, 1.5), hue=(-0.2, 0.2)),
         transforms.RandomResizedCrop(227),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
         lambda x: x + 0.0 * torch.randn_like(x)]
    )
    
    transform_base = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    )
    
    train_loader, _, _ = get_dataloaders(ds_name, test_domain, transform_train)
    _, val_loader, test_loader = get_dataloaders(ds_name, test_domain, transform_base)

    return train_loader, val_loader, test_loader

class NoisyDistilDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, teacher, transform):
        self.dataset = []
        self.transform = transform
         
        loader = torch.utils.data.DataLoader(base_dataset, batch_size=128, shuffle=False, num_workers=4)
        teacher.eval()
        
        for batch, labels in loader:
            batch = batch.to(device)
            with torch.no_grad():
                teacher_output = teacher(batch)
            
            for j in range(len(batch)):
                self.dataset.append((batch[j].to('cpu'), teacher_output[j].to('cpu'), labels[j]))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        batch, teacher_out, label = self.dataset[idx]
        
        return self.transform(batch), label, teacher_out


def get_data_distil_noisy(test_domain, teacher):
    transform_base = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((.5, .5, .5), (.5, .5, .5))]
    )
    
    trainset, valset, testset = get_datasets(test_domain, transform_base)
    
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(227, scale=(0.75, 1.0), ratio=(0.85, 1.15)),
         transforms.RandomHorizontalFlip()]
    )
    
    trainset = NoisyDistilDataset(trainset, teacher, transform_train)
  
    
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                             shuffle=True, num_workers=4)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=True, num_workers=4)
    
    return train_loader, val_loader, test_loader
