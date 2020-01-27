import torch
import torch.nn as nn

class SegmentationLosses(object):
    # init make constructor and set transformation,which creates directory containing all as an argument to the constructor
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        #self.ignore_index
        self.ignore_index = ignore_index
        # self.weight
        self.weight = weight
        # self.size_average
        self.size_average = size_average
        # self.batch_average
        self.batch_average = batch_average
        # self.cuda 
        self.cuda = cuda
    #define build_loss with 'ce' which means cross-entropy loss
    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        # define statements with CrossEntropyLoss
        if mode == 'ce':
            return self.CrossEntropyLoss
        # or elif with 'focal' loss
        elif mode == 'focal':
            return self.FocalLoss
        # else with 'else'
        else:
            raise NotImplementedError
    # Then Determine CrossEntropyLoss with logit and target
    def CrossEntropyLoss(self, logit, target):
        # form n, c, h, w with respect to logit.size
        n, c, h, w = logit.size()
        # here, crossentropy loss takes into __init__constructor account to set them into barcket
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        # once we did not self.cuda, in order to add it to 'if' statement
        if self.cuda:
            criterion = criterion.cuda()
        # if this is case, return loss of self.cuda
        loss = criterion(logit, target.long())
        # as the same as self.cuda, self.batch_average allude to:
        if self.batch_average:
            loss /= n

        return loss
    # As a Second loss, define FocalLoss with a gamma,alpha
    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        #second time define n, c, h, w as being equal to logit.size()
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()
        # modification happens wrp to previous 
        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt
 
        if self.batch_average:
            loss /= n

        return loss

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())




