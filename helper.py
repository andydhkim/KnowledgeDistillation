'''
helper functions
'''

import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

'''
Returns model
'''
def getModel(model):

    if model == 'resnet18':
        from models.resnet import resnet
        net = resnet(groups = [1, 1, 1, 1], depth = 18, width = [64, 128, 256, 512], dataset = "cifar100").cuda()
    elif model == 'mobilenet':
        from models.mobilenet import mobilenet
        net = mobilenet(num_classes = 100, cifar = True).cuda()
    else:
        print('check the name of the model')
        sys.exit()

    return net

'''
Returns train data loader
'''
def getTrainloader(batch_size = 128, shuffle=True):

    # data augmentation
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    
    # obtain data loader
    trainset = torchvision.datasets.CIFAR100(root='./data', 
                                                      train=True, 
                                                      download=True, 
                                                      transform=transform_train)
    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return trainloader

    
'''
Returns test data loader
'''    
def getTestloader(batch_size = 128, shuffle=True):

    # data augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ])
    
    # obtain data loader
    testset  = torchvision.datasets.CIFAR100(root='./data', 
                                             train=False, 
                                             download=True, 
                                             transform=transform_test)
    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return testloader
 
