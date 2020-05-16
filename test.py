'''
The following program tests the performance of a specified neural network by
calculating the top 1 and top 5 error rate. It also calculates precision, recall
and macro F1-score.
'''

import argparse
import os
import numpy as np
import torch
from torch.autograd import Variable
from helper import getModel, getTestloader
from torchnet.meter import ConfusionMeter

if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True, help='model architecture')
    parser.add_argument('-state', type=str, required=True, help='saved state of model')
    parser.add_argument('-shuffle', type=bool, default=True, help='turn shuffling of dataset on/off')
    parser.add_argument('-cm', type=bool, default=False, help='whether to save confusion matrix')
    args = parser.parse_args()
    
    # obtain test data
    testloader = getTestloader(shuffle=args.shuffle)
    
    # obtain network
    model = getModel(args.model)
    model.load_state_dict(torch.load(args.state), True)
    model.eval()
    
    # obtain top1 and top 5 accuracy rate
    top1Acc, top5Acc = 0.0, 0.0

    # activation matrix
    activations = []
    
    for n_iter, (image, label) in enumerate(testloader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = model(image)

        _, pred = output.topk(5, 1, largest=True, sorted=True)
        
        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        #compute top 5
        top5Acc += correct[:, :5].sum()

        #compute top1 
        top1Acc += correct[:, :1].sum()

    print("Top 1 err: ", 1 - top1Acc / len(testloader.dataset))
    print("Top 5 err: ", 1 - top5Acc / len(testloader.dataset))
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))