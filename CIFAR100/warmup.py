'''
warmup class

Gradually warm-up(increasing) learning rate in optimizer.
Proposed in 'Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour'.
'''

import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmUpScheduler(_LRScheduler):
    '''
    initialize with optimizer (which includes the target LR) and the total_epochs
    to reach that LR
    '''
    def __init__(self, optimizer, total_epoch):
        self.total_epoch = total_epoch
        super().__init__(optimizer)

    '''
    increase learning rate linearly for each iteration until it reaches
    the desired learning rate
    '''
    def get_lr(self):
        return [base_lr * self.last_epoch / self.total_epoch for base_lr in self.base_lrs]
