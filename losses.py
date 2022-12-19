import os, logging, traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
    n, c, h, w = logit.size()
    # logit = logit.permute(0, 2, 3, 1)
    target = target.squeeze(1)
    if weight is None:
        criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
    else:
        criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
    loss = criterion(logit, target.long())

    if size_average:
        loss /= (h * w)

    if batch_average:
        loss /= n

    return loss


class CrossEntropy2D(object):
    def __init__(self, weight=None, size_average=True, batch_average=True):
        self.size_average = size_average
        self.batch_average = batch_average
        if weight is None:
            self.criterion = nn.CrossEntropyLoss(weight=weight, size_average=False)
        else:
            self.criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), size_average=False)
    
    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        target = target.squeeze(1)
        loss = self.criterion(logit, target.long())
        if self.size_average:
            loss /= (h * w)

        if self.batch_average:
            loss /= n
        return loss

class DiceLoss2D(nn.Module):
    def __init__(self, n_classes, smooth = 1):
        self.n_classes = n_classes
        self.smooth = smooth
    
    def __call__(self, input, target):
        
#         assert input.size() == target.size(), "Input sizes must be equal."
#         assert input.dim() == 4, "Input must be a 4D Tensor."
        
        target = self._onehot(target, self.n_classes)
        probs=F.softmax(input)
        num=probs*target#b,c,h,w--p*g
        num=torch.sum(num,dim=3)#b,c,h
        num=torch.sum(num,dim=2)

        den1=probs*probs#--p^2
        den1=torch.sum(den1,dim=3)#b,c,h
        den1=torch.sum(den1,dim=2)

        den2=target*target#--g^2
        den2=torch.sum(den2,dim=3)#b,c,h
        den2=torch.sum(den2,dim=2)#b,c

        dice=(2*num+self.smooth)/(den1+den2)
        dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

        dice_total= 1 - torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz
        return dice_total
    
    def _onehot(self, logits, n_classes):
        y_onehot = torch.cuda.FloatTensor(logits.size(0), n_classes, logits.size(1), logits.size(2)).zero_()
        return y_onehot.scatter_(1, torch.unsqueeze(logits, 1),1)
        

class DiceLoss1D(nn.Module):
    def __init__(self, smooth = 1):
        self.smooth = 1
    
    def __call__(self, logits, labels):
        """The shape of logits is supposed to be (num_elements, num_classes),
        The shape of labels is supposed to be (num_elements) since each element represents the class index"""
        
        probs = F.softmax(logits, 1)
        labels = self._one_hot(labels, probs.size(1)).cuda()
        intersection = probs * labels
        score = (2. * intersection.sum(0) + self.smooth) / (probs.sum(0) + labels.sum(0) + self.smooth)  
        return 1 - score.mean()

    def _one_hot(self, index, n_classes):
        target_size = index.size() + (n_classes, )
        index = torch.unsqueeze(index, 1)
        mask = torch.cuda.FloatTensor(target_size).zero_()
        return mask.scatter_(1, index, 1.)
