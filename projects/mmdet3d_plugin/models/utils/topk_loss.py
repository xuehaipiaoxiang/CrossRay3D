import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class TopkLoss(nn.Module):
    '''
    foreground TOP-k
    '''

    def __init__(self, loss_weight=1):
        super(TopkLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight, avg_factor=None):
        num_item, num_classes = pred.shape
        target = F.one_hot(target, num_classes=num_classes + 1)
        target = target[:, :num_classes]
        target = target.type_as(pred)
        loss = F.binary_cross_entropy_with_logits(
            pred, target, reduction='none')
        loss = loss * weight
        eps = torch.finfo(torch.float32).eps
        loss = loss.sum() / (avg_factor + eps)
        return loss
    