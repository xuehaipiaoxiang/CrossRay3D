# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Huiming Yang
# ------------------------------------------------------------------------
from functools import partial
import torch
import torch.nn as nn
from mmdet.models.builder import BACKBONES
import numpy as np
from mmcv.runner import force_fp32
from ..utils.misc import  draw_heatmap_gaussian
from mmdet.core import (multi_apply,
                        reduce_mean, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from einops import rearrange
from mmdet.models import build_loss
from mmdet3d.models.utils import clip_sigmoid
import mmcv
from matplotlib import pyplot as plt
import os
import torch.nn.functional as F
from projects.mmdet3d_plugin.models.utils.misc import topk_gather
from mmcv.cnn import ConvModule


try:
    from ..utils.spconv_utils import spconv, replace_feature
except:
    from projects.mmdet3d_plugin.models.utils.spconv_utils import spconv, replace_feature
# from ..utils.focal_sparse_utils import FocalLoss
# from ..ops.roi_aware_pool3d.roiaware_pool3d_utils import points_of_pillar_in_gt_gpu


def c_p(v0, v1):
    return v0[..., 0]*v1[..., 1] - v0[..., 1]*v1[..., 0]


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

@BACKBONES.register_module()
class FocalSparseBEVBackBone(nn.Module):
    def __init__(self,  input_channels = 5, grid_size = [1024, 1024, 40],
                 point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0], voxel_size = [0.1, 0.1, 0.2],
                 train_ratio = 1.0, infer_ratio = 1.0,
                 num_classes = 10, down_stride = 8,
                 topk_weight = 1.5,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0), 
                  loss_pts_topk=dict(
                     type='TopkLoss',
                     loss_weight=1.0), 
                
                  **kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        spconv_kernel_sizes =  [3, 3, 3, 3]
        channels = [16, 32, 64, 128, 128]
        out_channel =256
        self.voxel_stride = 8
        self.voxel_size = torch.tensor(voxel_size).cuda()
        self.point_cloud_range = torch.tensor(point_cloud_range).cuda()
        self.train_ratio = train_ratio
        self.infer_ratio = infer_ratio
        self.sparse_shape = np.array(grid_size)[::-1] + [1, 0, 0] #[41, 1024, 1024]
        self.down_shape = torch.tensor(grid_size).cuda() / down_stride
        self.down_stride = down_stride
        self.out_channel = out_channel
        self.num_classes = num_classes
        self.down_stride_pixel = (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.down_shape[0]
        self.topk_weight = topk_weight

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, channels[0], 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(channels[0]),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(channels[0], channels[0], norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(channels[0], channels[1], spconv_kernel_sizes[0], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[0]//2), indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(channels[1], channels[1], norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(channels[1], channels[2], spconv_kernel_sizes[1], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[1]//2), indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(channels[2], channels[2], norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 6]
            block(channels[2], channels[3], spconv_kernel_sizes[2], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[2]//2), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(channels[3], channels[3], norm_fn=norm_fn, indice_key='res4'),
        )

        self.conv5 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[3], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv5', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res5'),
        )
        
        self.conv6 = spconv.SparseSequential(
            # [200, 176, 6] <- [100, 88, 3]
            block(channels[4], channels[4], spconv_kernel_sizes[3], norm_fn=norm_fn, stride=2, padding=int(spconv_kernel_sizes[3]//2), indice_key='spconv6', conv_type='spconv'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
            SparseBasicBlock(channels[4], channels[4], norm_fn=norm_fn, indice_key='res6'),
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv2d(channels[3], out_channel, 3, stride=1, padding=1, bias=False, indice_key='spconv_down2'),
            norm_fn(out_channel),
            nn.ReLU(),
        )

        self.shared_conv = spconv.SparseSequential(
            spconv.SubMConv2d(out_channel, out_channel, 3, stride=1, padding=1, bias=True),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(True),
        )

        self.forward_ret_dict = {}
        self.num_point_features = out_channel
        self.backbone_channels = {
            'x_conv1': channels[0],
            'x_conv2': channels[1],
            'x_conv3': channels[2],
            'x_conv4': channels[3]
        }

        self.shared_conv_modality = ConvModule(
            self.out_channel,
            self.out_channel,
            kernel_size=3,
            padding=1,
            conv_cfg=dict(type="Conv2d"),
            norm_cfg=dict(type="BN2d")
        )

        self.shared_semantic= nn.Sequential(
                                 nn.Conv2d(self.out_channel, self.out_channel, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.out_channel),
                                 nn.GELU(),
                                 )

        self.semantic_head = nn.Conv2d(self.out_channel, self.num_classes, kernel_size=1)
        # self.loss_pts_cls = build_loss(loss_cls)
        self.loss_pts_topk = build_loss(loss_pts_topk)

    def bev_out(self, x_conv):
        features_cat = x_conv.features
        indices_cat = x_conv.indices[:, [0, 2, 3]]
        spatial_shape = x_conv.spatial_shape[1:]

        indices_unique, _inv = torch.unique(indices_cat, dim=0, return_inverse=True)
        features_unique = features_cat.new_zeros((indices_unique.shape[0], features_cat.shape[1]))
        features_unique.index_add_(0, _inv, features_cat)

        x_out = spconv.SparseConvTensor(
            features=features_unique,
            indices=indices_unique,
            spatial_shape=spatial_shape,
            batch_size=x_conv.batch_size
        )
        return x_out
    
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']

        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)
        x_conv5 = self.conv5(x_conv4)
        x_conv6 = self.conv6(x_conv5)

        x_conv5.indices[:, 1:] *= 2
        x_conv6.indices[:, 1:] *= 4
        x_conv4 = x_conv4.replace_feature(torch.cat([x_conv4.features, x_conv5.features, x_conv6.features]))
        x_conv4.indices = torch.cat([x_conv4.indices, x_conv5.indices, x_conv6.indices])

        out = self.bev_out(x_conv4)

        out = self.conv_out(out)
        out = self.shared_conv(out)

        return self.generate_sparse_features(out)
    
    def generate_sparse_features(self, sp_out):
        '''
        to collect foreground instance
        '''
        pillar_feats = sp_out.dense()
        pillar_feats = self.shared_conv_modality(pillar_feats)
        bs, c, h, w = pillar_feats.shape
        sample_ratio = self.train_ratio if self.training else self.infer_ratio

        semantic_feats = self.shared_semantic(pillar_feats)
        semantic_logit = self.semantic_head(semantic_feats)

        semantic_logit = semantic_logit.permute(0,2,3,1).reshape(bs, -1, self.num_classes) #（b t c）
        semantic_score, cls_index = semantic_logit.max(dim = -1, keepdim=True)
        semantic_score = semantic_score.detach()

        pillar_feats = pillar_feats.permute(0,2,3,1).reshape(bs,-1,c) #（b t c）

        num_tokens = h*w
        num_sample_tokens = int(num_tokens * sample_ratio)
        _, topk_indexes = torch.topk(semantic_score, num_tokens, dim=1)
        pillar_feats = topk_gather(pillar_feats, topk_indexes[:, :num_sample_tokens, :])
        outs = {
            'pillar_feats':pillar_feats,
            'sample_weight':semantic_score,
            'topk_indexes':topk_indexes,
            'num_sample_tokens':num_sample_tokens
        }    

        if self.training:
            outs = self._pre_for_topk_loss(outs, semantic_logit, cls_index)
        return outs
    
    @force_fp32(apply_to=('pts_dict'))
    def loss(self, preds_dicts, img_metas, raw_x=None):
        ''' cross modality loss  '''
        gt_boxes, gt_labels, gt_volume= multi_apply(self._get_label_single, img_metas)
        nums_gt = [label.size(0) for label in gt_labels]

        gt_boxes_x = torch.cat(gt_boxes)
        gt_boxes_x = gt_boxes_x.mean(dim=1)
        gt_centers = torch.split(gt_boxes_x, nums_gt, dim=0)

        bev_shape = self.down_shape[:2].int().cpu().numpy().tolist()
        labels_list = self.get_targets(gt_boxes, gt_centers, gt_labels, bev_shape, gt_volume)

        ################################### START slot to visualize labels & uncomment slot in Detector
        sample_weight = preds_dicts['sample_weight']
        cls_index = preds_dicts['cls_index']
        # try:
        multi_apply(self._visualize, raw_x, gt_centers, gt_boxes, sample_weight, cls_index, gt_labels) # for pre vis
        # multi_apply(self._visualize, raw_x, gt_centers, gt_boxes, labels_list) # for target vis
        # except:
        #     pass
        ################################### END slot to visualize labels

        (gt_cls_num, ) = self.get_targets_cls_nums_x(labels_list)

        # flatten all
        flatten_labels = torch.cat(labels_list)

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=flatten_labels.device)
        num_pos = max(reduce_mean(num_pos), 1.0)


        # cls_scores = preds_dicts['semantic_logit']
        # flatten_cls_scores = [cs for cs in cls_scores]
        # flatten_cls_scores = torch.cat(flatten_cls_scores)
        # loss_pts_cls = self.loss_pts_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        loss_pts_topk = self.cal_loss_topk(preds_dicts, labels_list, gt_cls_num, avg_factor=num_pos)

        return {
            'loss_pts_topk':loss_pts_topk * 0,
            # 'loss_pts_cls' : loss_pts_cls,
                } 

    def get_targets(self, gt_box, gt_centers, gt_labels, bev_shape, gt_volume):
        (labels_list, ) = multi_apply(self._get_target_single, gt_box, gt_centers, gt_labels, gt_volume, bev_shape=bev_shape)
        return labels_list
    
    def _get_target_single(self, gt_bboxes, gt_centers, gt_labels, gt_volume, bev_shape):
        '''
        If the ray hit the OBB 
        '''
        INF = 1e8
        feat_w, feat_h = bev_shape
        shift_x = torch.arange(0, feat_w, device=gt_labels.device)
        shift_y = torch.arange(0, feat_h, device=gt_labels.device)
        
        yy, xx = torch.meshgrid(shift_y, shift_x, indexing='ij')
        xs, ys = xx.reshape(-1), yy.reshape(-1)
        num_points = xs.size(0)
        num_gts = gt_labels.size(0)
        gt_volume = gt_volume[None].repeat(num_points, 1)

        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)
        pxy = torch.stack((xs, ys), dim=-1) 

        pxy_diff = (torch.rand(pxy.shape, device='cuda') * 2 - 1 ) * 0.15

        pxy = (pxy + 0.5 + pxy_diff) * self.down_stride_pixel + self.point_cloud_range[None, :2]

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4, 2)
        o0, o1, o2, o3 = gt_bboxes.split(1, dim=-2)
        o0, o1, o2, o3 = o0.squeeze(dim=-2), o1.squeeze(dim=-2), o2.squeeze(dim=-2), o3.squeeze(dim=-2)
        v0 = o0 - o3
        v1 = pxy - o3

        v2 = o2 - o1
        v3 = pxy - o1

        v4 = o1 - o0
        v5 = pxy - o0

        v6 = o3 - o2
        v7 = pxy - o2

        inside_gt_bbox_mask = ( (c_p(v0, v1) * c_p(v2, v3)) > 0 ) * ( (c_p(v4, v5) * c_p(v6, v7) ) > 0 )

        gt_volume[inside_gt_bbox_mask == 0] = INF
        min_area, min_area_inds = gt_volume.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG

        gt_centers = (gt_centers + self.point_cloud_range[None, [3, 4]] ) / self.down_stride_pixel
        gt_centers = (gt_centers.round()).to(torch.int64)
        gt_centers = gt_centers.clip(min=0, max=self.down_shape[0]-1)

        gather_index = gt_centers[:, 0] + gt_centers[:, 1] * int(self.down_shape[0])
        labels[gather_index] = gt_labels

        return (labels, )

    def get_targets_cls_nums(self, gt_boxes, gt_labels2d_list):
        (out_list, ) = multi_apply(self._get_targets_cls_nums, gt_boxes, gt_labels2d_list)
        return out_list
        
    def _get_targets_cls_nums(self, gt_boxes, gt_labels):
        '''
        '''
        w = (gt_boxes[:, 2] / self.down_stride_pixel + 0.5).to(torch.int64)
        h = (gt_boxes[:, 3] / self.down_stride_pixel +0.5).to(torch.int64)
        gt_boxes_area = (w*h).clip(min=1)
        gt_labels_repeat = gt_labels.repeat_interleave(gt_boxes_area)
        gt_tensor = torch.bincount(gt_labels_repeat, minlength=self.num_classes)
        return (gt_tensor ,)

    
    def cal_loss_topk(self, preds_dicts, labels_list, gt_cls_num, avg_factor=None):
        '''
        distribution supervision
        '''
        topk_indexes = preds_dicts['topk_indexes']
        cls_index = preds_dicts['cls_index']
        all_batch_cls_num = preds_dicts['all_batch_cls_num']
        semantic_score = preds_dicts['sample_weight']
        semantic_logit = preds_dicts['semantic_logit']

        bs, n = semantic_logit.shape[:2]

        semantic_score = topk_gather(semantic_score, topk_indexes)
        semantic_logit = topk_gather(semantic_logit, topk_indexes)
        cls_index = topk_gather(cls_index, topk_indexes)
        labels_list = torch.stack(labels_list, dim=0)
        labels_list = labels_list.unsqueeze(-1)
        labels_list = topk_gather(labels_list, topk_indexes)

        weight = semantic_score.sigmoid()
        
        all_batch_cls_num = torch.tensor(all_batch_cls_num, device='cuda').to(torch.int64)
        gt_cls_num = torch.tensor(gt_cls_num, device='cuda').to(torch.int64)
        cls_num = torch.where(gt_cls_num <= all_batch_cls_num, gt_cls_num, all_batch_cls_num)

        cls_index = cls_index.squeeze(-1)
        for b in range(bs):
            for n in range(self.num_classes):
                cur_cls_num = cls_num[b][n]
                if cur_cls_num == 0:
                    continue
                to_index = torch.nonzero(cls_index[b] == n, as_tuple=True)[0]
                to_index = to_index[:cur_cls_num]
                weight[b][to_index] = self.topk_weight

        # flatten all
        flatten_semantic_logit = [sl for sl in semantic_logit]
        flatten_semantic_logit = torch.cat(flatten_semantic_logit)
        flatten_labels_list = [ll for ll in labels_list]
        flatten_labels_list = torch.cat(flatten_labels_list)
        flatten_labels_list = flatten_labels_list.squeeze(-1)
        flatten_weight = [w for w in weight]
        flatten_weight = torch.cat(flatten_weight)

        loss_pts_topk = self.loss_pts_topk(flatten_semantic_logit, flatten_labels_list, flatten_weight, avg_factor)
        return loss_pts_topk
    
    def get_targets_cls_nums_x(self, labels_list):
        gt_cls_num = multi_apply(self._get_targets_cls_nums_x, labels_list)
        return gt_cls_num
    def _get_targets_cls_nums_x(self, labels):
        batch_cls_num = []
        for i in range(self.num_classes):
            cls_cur = int( torch.sum(labels == i) )
            batch_cls_num.append(cls_cur)
        return (batch_cls_num, )

    def _pre_for_topk_loss(self, outs, semantic_logit, cls_index):
        '''
        Prepare for top-k loss
        '''
        cls_index = cls_index.squeeze(-1)
        bs = cls_index.size(0)
        all_batch_cls_num = []
        for b in range(bs):
            batch_cls_num = []
            cur_cls_index = cls_index[b]
            for i in range(self.num_classes):
                to_index = cur_cls_index == i
                cls_cur = int( torch.sum(to_index) )
                batch_cls_num.append(cls_cur)
            all_batch_cls_num.append(batch_cls_num)
        outs['all_batch_cls_num'] = all_batch_cls_num
        outs['cls_index'] = cls_index.unsqueeze(-1)
        outs['semantic_logit'] = semantic_logit
        return outs


    def _get_label_single(self, img_meta):
        '''
        convert to gt boxes in bev
        '''
        LiBoxes = img_meta['gt_bboxes_3d'].data
        labels = img_meta['gt_labels_3d'].data
        gt_boxes = LiBoxes.corners
        gt_boxes, labels = gt_boxes.cuda(), labels.cuda()
        gt_volume = LiBoxes.volume
        # xyxyxyxy  left-top -> clockwise (n, 4, 2)
        return (gt_boxes[:, [4, 7, 3, 0], :2], labels, gt_volume.cuda())
    


    # def _visualize(self, lidar, center_points, bboxes, labels_list):
    def _visualize(self, lidar, center_points, bboxes, weight, cls_index, labels, labels_list=None):
        '''
        Modified by Huiming Yang
        '''
        class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]

        OBJECT_PALETTE = {
        "car": (255, 158, 0),
        "truck": (255, 99, 71),
        "construction_vehicle": (233, 150, 70),
        "bus": (255, 69, 0),
        "trailer": (255, 140, 0),
        "barrier": (112, 128, 144),
        "motorcycle": (255, 61, 99),
        "bicycle": (220, 20, 60),
        "pedestrian": (0, 0, 230),
        "traffic_cone": (47, 79, 79),
        }

        OBJECT_PALETTE_list = []
        for key, values in OBJECT_PALETTE.items():
            OBJECT_PALETTE_list.append(values)
        OBJECT_PALETTE_tensor = torch.tensor(OBJECT_PALETTE_list, device='cuda')
        OBJECT_PALETTE_tensor = OBJECT_PALETTE_tensor / 255
        center_points = center_points.cpu().numpy()

        i = np.random.randint(0,1000)
        fpath = "vis/pictures_foreground/LiDAR/" + str(i) +".png"
        index = np.array([0, 3], dtype=np.int32)
        xlim = self.point_cloud_range.cpu().numpy()[index]
        ylim = self.point_cloud_range.cpu().numpy()[index+1]
        fig = plt.figure(figsize=(xlim[1] - xlim[0], ylim[1] - ylim[0]))
        ax = plt.gca()
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect(1)
        ax.set_axis_off()

        palette1 = (255, 158, 0)
        color = np.array(palette1) / 255,
        thickness = 6.0

        # if labels_list is not None: #TODO label map
        #     lidar_o = lidar
        #     lidar_o = lidar_o.cpu().numpy()[:,:2]
        #     lidar = ((lidar[:, :2] +54.0) / 108.0) * 180
        #     lidar = (lidar).to(torch.int64)
        #     gather_index = lidar[:, 0] +lidar[:, 1] * 180
        #     # color = torch.tensor((1, 0, 0), device='cuda')
        #     black = OBJECT_PALETTE_tensor.new_tensor((0, 0, 0))
        #     OBJECT_PALETTE_tensor = torch.cat([OBJECT_PALETTE_tensor, black[None]], dim=0)
        #     color = OBJECT_PALETTE_tensor[labels_list.squeeze(-1)]
        #     gather_index = gather_index[:,None].repeat(1, 3)
        #     gather_index = gather_index.clip(min=0, max=color.size(0)-1)
        #     g_color = torch.gather(color, dim = 0, index = gather_index)
        #     # g_color = g_weight * g_color
        #     g_color = g_color.cpu().numpy()
        #     plt.scatter(
        #         lidar_o[:, 0],
        #         lidar_o[:, 1],
        #         s=1.0,
        #         c=g_color,
        #     )

        # if lidar is not None: #TODO point clouds 
        #     lidar = lidar.cpu().numpy()[:,:2]
        #     plt.scatter(
        #         lidar[:, 0],
        #         lidar[:, 1],
        #         s=1.0,
        #         c="black",
        #     )
            
        if weight is not None: #TODO  TOP-K select
            lidar_o = lidar
            lidar_o = lidar_o.cpu().numpy()[:,:2]
            lidar = ((lidar[:, :2] + 54.0) / 108.0) * 180 
            lidar = (lidar).to(torch.int64)
            gather_index = lidar[:, 0] +lidar[:, 1] * 180
            gather_index = gather_index.clip(min=0, max=weight.size(0)-1)
            g_weight = torch.gather(weight, dim = 0, index = gather_index[:,None])
            color = OBJECT_PALETTE_tensor[cls_index.squeeze(-1)]
            num_lidar_point = g_weight.size(0)
            num_lidar_point = int(num_lidar_point * 0.75) # reverse
            topk_index = torch.topk(g_weight, num_lidar_point, dim=0, largest=False)[1]
            gather_index = gather_index[:,None].repeat(1, 3)
            g_color = torch.gather(color, dim = 0, index = gather_index)
            g_color = g_color.new_tensor((1, 0, 0)).repeat(g_color.size(0),1)
            g_color[topk_index.squeeze(-1)] = 1
            g_color = g_color.cpu().numpy()
            plt.scatter(
                lidar_o[:, 0],
                lidar_o[:, 1],
                s=1.0,
                c=g_color,
            )

        # if center_points is not None: #TODO center point
        #     plt.scatter(
        #         center_points[:, 0],
        #         center_points[:, 1],
        #         # s=600,
        #         s=3,
        #         # vmin=100,
        #         # marker='o',
        #         c="red",
        #         # edgecolor='none',
        #     )

        if bboxes is not None and len(bboxes) > 0: #TODO 2D box
            bboxes = torch.cat([bboxes, bboxes[:, [0], :]], dim=1)
            bboxes = bboxes.cpu().numpy()
            for index in range(bboxes.shape[0]):
                name = class_names[labels[index]]
                plt.plot(
                    bboxes[index, :, 0],
                    bboxes[index, :, 1],
                    linewidth=thickness,
                    color=np.array(OBJECT_PALETTE[name]) / 255,
                )

        mmcv.mkdir_or_exist(os.path.dirname(fpath))
        fig.savefig(
            fpath,
            dpi=10,
            facecolor="white",
            format="png",
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close()
