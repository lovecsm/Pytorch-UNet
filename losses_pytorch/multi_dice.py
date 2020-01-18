"""
Dice loss
用来处理分割过程中的前景背景像素非平衡的问题
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, pred, target):
        smooth = 1e-5
        # have to use contiguous since they may from a torch.view op
        iflat = pred.contiguous().view(-1)
        tflat = target.contiguous().view(-1)
        intersection = (iflat * tflat).sum()

        A_sum = torch.sum(iflat)
        B_sum = torch.sum(tflat)
        return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth))


class MultiDiceLoss(nn.Module):
    def __init__(self, C, weight=None):
        super(MultiDiceLoss, self).__init__()
        self.dice = DiceLoss()
        self.weight = weight
        if self.weight == None:
            self.weight = [1 / C] * C
        self.weight = torch.FloatTensor(self.weight).cuda()

    def forward(self, pred, target):
        B, C, H, W = pred.shape
        total_dice = 0.0
        for i in range(C):
            total_dice += (self.dice(pred[:, i, :, :], target[:, i, :, :]) * self.weight[i])

        return total_dice / C


class CrossEntropy2d(nn.Module):

    def __init__(self, C, size_average=True, ignore_label=255, weight=None):
        super(CrossEntropy2d, self).__init__()
        self.size_average = size_average
        self.ignore_label = ignore_label
        self.weight = weight
        if self.weight == None:
            self.weight = [1 / C] * C
        self.weight = torch.FloatTensor(self.weight).cpu()
        print("loss weight: ", self.weight)

    def forward(self, predict, target):
        """
            Args:
                predict:(n, c, h, w)
                target:(n, h, w)
                weight (Tensor, optional): a manual rescaling weight given to each class.
                                           If given, has to be a Tensor of size "nclasses"
        """
        target = torch.argmax(target, 1)
        assert not target.requires_grad
        assert predict.dim() == 4
        assert target.dim() == 3
        assert predict.size(0) == target.size(0), "{0} vs {1} ".format(predict.size(0), target.size(0))
        assert predict.size(2) == target.size(1), "{0} vs {1} ".format(predict.size(2), target.size(1))
        assert predict.size(3) == target.size(2), "{0} vs {1} ".format(predict.size(3), target.size(3))
        n, c, h, w = predict.size()
        target_mask = (target >= 0) * (target != self.ignore_label)
        target = target[target_mask]
        if not target.data.dim():
            return torch.zeros(1)
        predict = predict.transpose(1, 2).transpose(2, 3).contiguous()
        predict = predict[target_mask.view(n, h, w, 1).repeat(1, 1, 1, c)].view(-1, c)
        print(predict)
        print(target)
        loss = F.cross_entropy(predict, target, weight=self.weight, size_average=self.size_average)
        return loss
