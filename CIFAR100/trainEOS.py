'''
The following program trains a neural network using SGD and Knowledge Distillation
with Eigenspace Overlap Score.
'''

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from helper import getModel, getTrainloader, getTestloader
from warmup import WarmUpScheduler

def train(epoch):
    model.train()
    curr_index = 0
    for batch_index, (images, labels) in enumerate(trainLoader):
        images, labels = Variable(images), Variable(labels)

        labels, images = labels.cuda(), images.cuda()
        
        curr_size = images.shape[0]
        
        optimizer.zero_grad()
        outputs, activations = model.forward(images, True)
        #teacher_output = teacherModel(images).detach()
        
        teacher_output, teacher_U = get_teacher_output(curr_index, curr_size)
        student_activations[curr_index:curr_index+curr_size, :] = activations
        if curr_index % 1000 < curr_size:
            P = update_P()
            
        loss = lossFnDistill(outputs, labels, teacher_output, activations, P, teacher_U, eos_scale, T, alpha, beta)
        loss.backward()
        optimizer.step()
        
        curr_index += curr_size

        if epoch <= args.warm:
            warmup_scheduler.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.batch + len(images),
            total_samples=len(trainLoader.dataset)
        ))

    # output training loss for each epoch
    writer.add_scalar('Train/loss', loss.item(), epoch)
    
def eval_training(epoch):
    model.eval()

    # note that the test_loss is just calculated as cross_entropy instead of the
    # new error function to allow for easier comparison
    test_loss, correct = 0.0, 0.0

    for (images, labels) in testloader:
        images, labels = Variable(images), Variable(labels)

        labels, images = labels.cuda(), images.cuda()

        outputs = model(images)
        loss = lossFn(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(testloader.dataset),
        correct.float() / len(testloader.dataset)
    ))
    print()

    # output test accuracy and loss for each epoch
    writer.add_scalar('Test/Average loss', test_loss / len(testloader.dataset), epoch)
    writer.add_scalar('Test/Accuracy', correct.float() / len(testloader.dataset), epoch)

    return correct.float() / len(testloader.dataset)

def update_P():
    with torch.no_grad():
        X = student_activations
        return torch.pinverse(X.t() @ X) @ X.t() @ teacher_Us   
    
def get_teacher_output(curr_index, curr_size):
    with torch.no_grad():
        curr_output = teacher_outputs[curr_index:curr_index + curr_size, :]
        curr_U = teacher_Us[curr_index:curr_index + curr_size, :]
        return curr_output, curr_U
    
def lossFnDistill(outputs, labels, teacher_output, activations, P, teacher_U, eos_scale, T, alpha, beta):
    return  nn.MSELoss()(activations @ P, teacher_U) * eos_scale * alpha + \
            nn.KLDivLoss()(F.log_softmax(outputs/T), F.softmax(teacher_output/T)) * (T*T * beta) + \
            F.cross_entropy(outputs, labels) * (1. - alpha - beta)
    
if __name__ == '__main__':
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-model', type=str, required=True, help='model architecture')
    parser.add_argument('-batch', type=int, default=128, help='mini batch size')
    parser.add_argument('-epochs', type=int, default=201, help='mini batch size')
    parser.add_argument('-shuffle', type=bool, default=True, help='turn shuffling of dataset on/off')
    parser.add_argument('-warm', type=int, default=1, help='warm up iterations')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    # KD specific
    parser.add_argument('-tSaved', type=str, required=True, help='the state file you want to distill from')
    parser.add_argument('-tModel', type=str, required=True, help='pre-trained teacher net type')
    parser.add_argument('-temp', type=float, default=10.0, help='temperature value')
    parser.add_argument('-alpha', type=float, default=0.8, help='alpha value')    
    parser.add_argument('-beta', type=float, default=0.1, help='beta value')    
    args = parser.parse_args()

    # obtain train data
    trainLoader = getTrainloader(batch_size=args.batch, shuffle=args.shuffle)
    # obtain test data
    testloader = getTestloader(batch_size=args.batch, shuffle=args.shuffle)
    
    # obtain teacher network
    teacherModel = getModel(args.tModel)
    teacherModel.load_state_dict(torch.load(args.tSaved), True)
    teacherModel.eval()
    
    # tensorboard set up
    if not os.path.exists('runs'):
        os.mkdir('runs')
    writer = SummaryWriter(log_dir=os.path.join(
            'runs', args.model, datetime.now().isoformat()))

    # create folder to save models
    savedModelsPath = os.path.join('SavedModels', args.model, datetime.now().isoformat())
    if not os.path.exists(savedModelsPath):
        os.makedirs(savedModelsPath)
    savedModelsPath = os.path.join(savedModelsPath, '{model}-{epoch}.pth')
    
    # set model architecture
    model = getModel(args.model)
    
    # training set up
    lossFn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    milestones = [60, 120, 160]
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.2)
    warmup_scheduler = WarmUpScheduler(optimizer, len(trainLoader) * args.warm)
    T = args.temp
    alpha = args.alpha  
    beta = args.beta
    
    # obtain teacher output and U matrices
    teacher_outputs = []
    student_activations = []
    teacher_activations = []
    torch.manual_seed(0)
    with torch.no_grad():
        print('Generating teacher outputs')
        for i, (images, _) in enumerate(trainLoader):
            images = Variable(images)
            images = images.cuda()

            _, student_activation = model.forward(images, True)
            curr_output, curr_activations = teacherModel.forward(images, True)
            student_activations.append(student_activation)
            teacher_outputs.append(curr_output)
            teacher_activations.append(curr_activations)
        student_activations = torch.cat(student_activations)
        teacher_outputs = torch.cat(teacher_outputs)
        teacher_activations = torch.cat(teacher_activations)
        train_svd = torch.svd(teacher_activations)
        teacher_Us = train_svd.U
        # we use eos_scale to scale the EOS loss during training, so that the same 
        # learning rates that were good for MSE loss are also good for EOS loss.
        eos_scale = torch.norm(teacher_activations) / torch.norm(teacher_Us)
        print('Done!')

    # train!
    torch.manual_seed(0)
    bestAcc = 0.0
    for epoch in range(1, args.epochs):
        train(epoch)
        acc = eval_training(epoch)

        if epoch > args.warm:
            train_scheduler.step()
        
        # save model anytime dev performance has improved
        if epoch > milestones[1] and bestAcc < acc:
            torch.save(model.state_dict(), savedModelsPath.format(model=args.model, epoch=epoch))
            bestAcc = acc
            continue
        
        # save last model
        if epoch == args.epochs - 1:
            torch.save(model.state_dict(), savedModelsPath.format(model=args.model, epoch=epoch))

    writer.close()
