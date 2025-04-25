import torch
from mmdet.core.bbox.match_costs.builder import MATCH_COST
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

@MATCH_COST.register_module()
class ForeFocalLossCost(nn.Module):
    '''
    foreground focal loss cost
    target shape [n,]
    input shape [n, classes]
    assert classes == 2
    '''
    def __init__(self, gamma=2.0, eps=1e-7):
        super(ForeFocalLossCost, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def one_hot(self, target, classes):
        size = target.size() + (classes,)
        view = target.size() + (1,)

        mask = torch.Tensor(*size).fill_(0).to(target.device)

        target = target.view(*view)
        ones = 1.

        if isinstance(target, Variable):
            ones = Variable(torch.Tensor(target.size()).fill_(1).to(target.device))
            mask = Variable(mask, volatile=target.volatile)
        return mask.scatter_(1, target, ones)

    def forward(self, input, target):
        y = self.one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)

        loss = -1 * y * torch.log(logit) # cross entropy
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.mean()




@MATCH_COST.register_module()
class BBox3DL1Cost(object):
    """BBox3DL1Cost.
     Args:
         weight (int | float, optional): loss_weight
    """

    def __init__(self, weight=1.):
        self.weight = weight

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * self.weight


@MATCH_COST.register_module()
class BBoxBEVL1Cost(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, bboxes, gt_bboxes, pc_range):
        pc_start = bboxes.new(pc_range[0:2])
        pc_range = bboxes.new(pc_range[3:5]) - bboxes.new(pc_range[0:2])
        # normalize the box center to [0, 1]
        normalized_bboxes_xy = (bboxes[:, :2] - pc_start) / pc_range
        normalized_gt_bboxes_xy = (gt_bboxes[:, :2] - pc_start) / pc_range
        reg_cost = torch.cdist(normalized_bboxes_xy, normalized_gt_bboxes_xy, p=1)
        return reg_cost * self.weight


@MATCH_COST.register_module()
class IoU3DCost(object):
    def __init__(self, weight):
        self.weight = weight

    def __call__(self, iou):
        iou_cost = - iou
        return iou_cost * self.weight