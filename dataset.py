import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

## Import Data Loaders ##
from dataloader import *

# Visda transforms
def get_transforms_comp(imageSize):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transforms_comp = transforms.Compose([
        transforms.RandomResizedCrop(imageSize),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    return transforms_comp

def get_dataset(dataset, root_dir, imageSize, batchSize, workers=1):
    
    if dataset == 'cifar10':
        train_dataset = dset.CIFAR10(root=root_dir, download=True, train=True,
                                      transform=transforms.Compose([
                                      transforms.Resize(imageSize),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]))
        test_dataset = dset.CIFAR10(root=root_dir, download=True, train=False,
                                      transform=transforms.Compose([
                                      transforms.Resize(imageSize),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                      ]))
    elif dataset == 'mnist':
        train_dataset = dset.MNIST(root=root_dir, train=True, download=True,
                                    transform=transforms.Compose([
                                    transforms.Resize(imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    ]))
        test_dataset = dset.MNIST(root=root_dir, train=False, download=True,
                                    transform=transforms.Compose([
                                    transforms.Resize(imageSize),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.1307,), (0.3081,)),
                                    ]))
    elif dataset == 'mnistm':
        train_dataset = MNIST_M(root=root_dir, train=True,
                                 transform=transforms.Compose([
                                 transforms.Resize(imageSize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ]))
        test_dataset = MNIST_M(root=root_dir, train=False,
                                 transform=transforms.Compose([
                                 transforms.Resize(imageSize),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                 ]))
    elif dataset == 'usps':
        train_dataset = USPS(root=root_dir, train=True,
                              image_size=imageSize,
                              transform=transforms.Compose([
                              transforms.Resize(imageSize),
                              ]))
        test_dataset = USPS(root=root_dir, train=False,
                              image_size=imageSize,
                              transform=transforms.Compose([
                              transforms.Resize(imageSize),
                              ]))
    elif dataset == 'visdas':
        train_dataset = MyImageFolder(root=root_dir, train=True, source=True,
                                      transform=get_transforms_comp(imageSize))
        test_dataset =  MyImageFolder(root=root_dir, train=False, source=True,
                                      transform=get_transforms_comp(imageSize))
    elif dataset == 'visdat':
        train_dataset = MyImageFolder(root=root_dir, train=True, source=False,
                                      transform=get_transforms_comp(imageSize))
        test_dataset =  MyImageFolder(root=root_dir, train=False, source=False,
                                      transform=get_transforms_comp(imageSize))

    assert train_dataset, test_dataset
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize,
                                                   shuffle=True, num_workers=int(workers))
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=batchSize,
                                                   shuffle=False, num_workers=int(workers))
    return train_dataloader, test_dataloader
