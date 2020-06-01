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

def compute_metrics(confusion_matrix):
    sum_precision = 0
    sum_recall = 0
    sum_f1 = 0
    for i in range(confusion_matrix.shape[0]):
        tp = confusion_matrix[i, i]
        tfp = np.sum(confusion_matrix[i, :])
        tpfn = np.sum(confusion_matrix[:, [i]])

        if tp == 0:
            if tfp == 0 and tpfn == 0:
                precision = 1
                recall = 1
                f1 = 1
            else:
                if tpfn == 0:
                    precision = 1
                    recall = 0
                elif tfp == 0:
                    precision = 0
                    recall = 1
                else:
                    precision = 0
                    recall = 0

                f1 = 0
        else:
            precision = float(tp) / tfp
            recall = float(tp) / tpfn
            f1 = float(2 * precision * recall) / (precision + recall)

        sum_precision += precision
        sum_recall += recall
        sum_f1 += f1

    average_precision = float(sum_precision) / confusion_matrix.shape[0]
    average_recall = float(sum_recall) / confusion_matrix.shape[0]
    average_f1 = float(sum_f1) / confusion_matrix.shape[0]

    return average_precision, average_recall, average_f1

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
    
    # confusion matrix initialization
    confusion_matrix = ConfusionMeter(100)

    # activation matrix
    activations = []
    
    for n_iter, (image, label) in enumerate(testloader):
        print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(testloader)))
        image = Variable(image).cuda()
        label = Variable(label).cuda()
        output = model(image)
        with torch.no_grad():
            _, activation = model.forward(image, True)
        
        activations.append(activation)
        _, pred = output.topk(5, 1, largest=True, sorted=True)

        confusion_matrix.add(output.data, label.data)
        
        label = label.view(label.size(0), -1).expand_as(pred)
        correct = pred.eq(label).float()

        #compute top 5
        top5Acc += correct[:, :5].sum()

        #compute top1 
        top1Acc += correct[:, :1].sum()
    
    activations = torch.cat(activations)
    activationSVD = torch.svd(activations)
    S = activationSVD.S.cpu().numpy()
    print(np.mean(S), np.std(S))
    # create folder to save confusion matrix
    if args.cm:
        if not os.path.exists('confusion'):
            os.makedirs('confusion')
        confusionpath = os.path.join('confusion', args.model + '-' + args.state.split("/")[2] +'.txt')
        np.savetxt(confusionpath, confusion_matrix.value().astype(int))

    print("Top 1 err: ", 1 - top1Acc / len(testloader.dataset))
    print("Top 5 err: ", 1 - top5Acc / len(testloader.dataset))
    prec, recall, f1 = compute_metrics(confusion_matrix.value())
    print("Precision: ", prec)
    print("Recall: ", recall)
    print("Macro F1: ", f1)
    print("Parameter numbers: {}".format(sum(p.numel() for p in model.parameters())))
