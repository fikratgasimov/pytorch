##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
## Created by: Hang Zhang
## ECE Department, Rutgers University
## Email: zhang.hang@rutgers.edu
## Copyright (c) 2017
##
## This source code is licensed under the MIT-style license found in the
## LICENSE file in the root directory of this source tree
##+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

import math
# Learning Rate defintion:
class LR_Scheduler(object):
    """Learning Rate Scheduler

    Step mode: ``lr = baselr * 0.1 ^ {floor(epoch-1 / lr_step)}``

    Cosine mode: ``lr = baselr * 0.5 * (1 + cos(iter/maxiter))``

    Poly mode: ``lr = baselr * (1 - iter/maxiter) ^ 0.9``

    Args:
        args:
          :attr:`args.lr_scheduler` lr scheduler mode (`cos`, `poly`),
          :attr:`args.lr` base learning rate, :attr:`args.epochs` number of epochs,
          :attr:`args.lr_step`

        iters_per_epoch: number of iterations per epoch
    """
    def __init__(self, mode, base_lr, num_epochs, iters_per_epoch=0,
                 lr_step=0, warmup_epochs=0):
        # set the mode of learning rate
        self.mode = mode
        print('Using {} LR Scheduler!'.format(self.mode))
        # set learning rate as base_lr:
        self.lr = base_lr
        # then, if statement mode is seen as 'step'
        if mode == 'step':
            # analysize the statement
            assert lr_step
        # if condition return True, create constructor, then set transformation which makes drct for lr_step
        self.lr_step = lr_step
        # create constructor and set transformation, which maakes drct for iter_per_epoch
        self.iters_per_epoch = iters_per_epoch
        #once iters_per_epoch defined, then find self.N
        self.N = num_epochs * iters_per_epoch
        # declare self.epoch as (- 1) which compared below:
        self.epoch = -1
        # finally, declare self.warmup_iters as the mult.(warmup_epochs,iters_per_epoch)
        self.warmup_iters = warmup_epochs * iters_per_epoch
    # declare definite call with optimized,i, epoch, best_pred:
    def __call__(self, optimizer, i, epoch, best_pred):
        # then, set T = as mult(epoch,self.iters_per_epoch) + i
        T = epoch * self.iters_per_epoch + i
        
        # finally, we can ensure about whether this mode is 'cos','poly','step'
        if self.mode == 'cos':
            #if yes, then set lr as the following:
            lr = 0.5 * self.lr * (1 + math.cos(1.0 * T / self.N * math.pi))
            
        # elifs statements,if 'poly', then lr will be seen:
        elif self.mode == 'poly':
            lr = self.lr * pow((1 - 1.0 * T / self.N), 0.9)
            
        # elifs statements, if 'step', ultimate lr will be:
        elif self.mode == 'step':
            lr = self.lr * (0.1 ** (epoch // self.lr_step))
            
        # Nothing above represented:
        else:
            raise NotImplemented
            
        # warm up lr schedule: self.warmup_iters which is seen as multiplication:
        if self.warmup_iters > 0 and T < self.warmup_iters:
            lr = lr * 1.0 * T / self.warmup_iters
        #self.epoch is negative (1)
        if epoch > self.epoch:
            print('\n=>Epoches %i, learning rate = %.4f, \
                previous best = %.4f' % (epoch, lr, best_pred))
            # finally, officially defined epoch
            self.epoch = epoch
        # and check lr rate
        assert lr >= 0
        # adjust learning rate in order to avoid overfitting
        self._adjust_learning_rate(optimizer, lr)
    # then, declare class of _adjust_learning_rate:
    def _adjust_learning_rate(self, optimizer, lr):
        if len(optimizer.param_groups) == 1:
            optimizer.param_groups[0]['lr'] = lr
        else:
            # enlarge the lr at the head
            optimizer.param_groups[0]['lr'] = lr
            for i in range(1, len(optimizer.param_groups)):
                optimizer.param_groups[i]['lr'] = lr * 10
