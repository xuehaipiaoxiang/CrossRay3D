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
from projects.mmdet3d_plugin.models.utils.topk_loss import TopkLoss



try:
    from ..utils.spconv_utils import spconv, replace_feature
except:
    from projects.mmdet3d_plugin.models.utils.spconv_utils import spconv, replace_feature
# from ..utils.focal_sparse_utils import FocalLoss
# from ..ops.roi_aware_pool3d.roiaware_pool3d_utils import points_of_pillar_in_gt_gpu

    
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
        self.sparse_shape = np.array(grid_size)[::-1] + [1, 0, 0]
        self.down_shape = torch.tensor(grid_size).cuda() / down_stride
        self.down_stride = down_stride
        self.out_channel = out_channel
        self.num_classes = num_classes
        self.down_stride_pixel = (self.point_cloud_range[3] - self.point_cloud_range[0]) / self.down_shape[0]

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

        self.shared_semantic= nn.Sequential(
                                 nn.Conv2d(self.out_channel, self.out_channel, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.out_channel),
                                 nn.GELU(),
                                 )

        self.semantic_head = nn.Conv2d(self.out_channel, self.num_classes, kernel_size=1)
        self.loss_pts_cls = build_loss(loss_cls)
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
        _, topk_indexes = torch.topk(semantic_score.detach(), num_tokens, dim=1)
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
        ''' gt_boxes in bev [center, length]'''

        cls_scores = preds_dicts['semantic_logit']
        gt_boxes, gt_labels = multi_apply(self._get_label_single, img_metas)
        gt_cls_num = self.get_targets_cls_nums(gt_boxes, gt_labels)
        nums_gt = [label.size(0) for label in gt_labels]
        gt_boxes = torch.cat(gt_boxes)
        gt_center = (gt_boxes[:, :2] - self.point_cloud_range[None, :2]) / self.down_stride_pixel
        gt_center = (gt_center + 0.5).int()
        gt_radius = torch.ceil(torch.max(gt_boxes[:,2:4], dim=-1)[0] / self.down_stride_pixel / 2) 
        gt_radius = torch.clamp(gt_radius, 1.0) 

        bev_shape = self.down_shape[:2].int().cpu().numpy().tolist()
        
        closure_gt_boxes = bbox_cxcywh_to_xyxy(torch.cat([gt_center, 2*gt_radius[:, None], 2*gt_radius[:, None]], dim=-1))
        closure_gt_boxes = closure_gt_boxes.clip(min=0, max=self.down_shape[0]-1)
        closure_gt_boxes_list = torch.split(closure_gt_boxes, nums_gt, dim=0)

        #################################### START slot to visualize labels & uncomment slot in Detector
        # try:
        #     multi_apply(self._visualize, raw_x, gt_center_list, closure_gt_boxes_list) # for target vis
        # except:
        #     pass
        #################################### END slot to visualize labels

        labels_list = self.get_targets(closure_gt_boxes_list, gt_labels, bev_shape)


        # flatten all
        flatten_labels = torch.cat(labels_list)
        flatten_cls_scores = [cs for cs in cls_scores]
        flatten_cls_scores = torch.cat(flatten_cls_scores)


        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=flatten_labels.device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        loss_pts_cls = self.loss_pts_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)
        loss_pts_topk = self.cal_loss_topk(preds_dicts, labels_list, gt_cls_num, avg_factor=num_pos)
        return {'loss_pts_cls':loss_pts_cls,
                'loss_pts_topk':loss_pts_topk,
                } 

    def get_targets(self, closure_gt_box, gt_labels, bev_shape):
        (labels_list, ) = multi_apply(self._get_target_single, closure_gt_box, gt_labels, bev_shape=bev_shape)
        return labels_list
    
    def _get_target_single(self, gt_bboxes, gt_labels, bev_shape):
        INF = 1e8
        feat_w, feat_h = bev_shape
        shift_x = torch.arange(0, feat_w, device=gt_labels.device)
        shift_y = torch.arange(0, feat_h, device=gt_labels.device)
        
        yy, xx = torch.meshgrid(shift_y, shift_x, indexing='ij')
        xs, ys = xx.reshape(-1), yy.reshape(-1)
        num_points = xs.size(0)
        num_gts = gt_labels.size(0)

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)

        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] >= 0
        areas[inside_gt_bbox_mask == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG
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

        all_batch_cls_num = torch.tensor(all_batch_cls_num).to(torch.int64).cuda()
        gt_cls_num = torch.stack(gt_cls_num)
        cls_num = torch.where(gt_cls_num <= all_batch_cls_num, gt_cls_num, all_batch_cls_num)

        cls_index = cls_index.squeeze(-1)
        for b in range(bs):
            for n in range(self.num_classes):
                cur_cls_num = cls_num[b][n]
                if cur_cls_num == 0:
                    continue
                to_index = torch.nonzero(cls_index[b] == n, as_tuple=True)[0]
                to_index = to_index[:cur_cls_num]
                weight[b][to_index] = 1.0

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
        gt_boxes = LiBoxes.bev[:, :4]
        return (gt_boxes.cuda(), labels.cuda())
    

    def _visualize(self, lidar, points, bboxes=None):
        '''
        to visuallize centerpoint in train, sample_weight mask in inference stage
        '''
        points = points / self.down_shape[None, 0:2]
        points = points * (self.point_cloud_range[3:5] - self.point_cloud_range[0:2])[None] + self.point_cloud_range[None, :2]
        lidar = lidar.cpu().numpy()[:,:2]
        points = points.cpu().numpy()

        i = np.random.randint(0,1000)
        fpath = "vis/pictures/LiDAR_gt/" + str(i) +".png"
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
        thickness = 2.0

        if lidar is not None:
            plt.scatter(
                lidar[:, 0],
                lidar[:, 1],
                s=1.0,
                c="green",
            )
        
        if points is not None:
            plt.scatter(
                points[:, 0],
                points[:, 1],
                s=600,
                # vmin=100,
                # marker='o',
                c="red",
                # edgecolor='none',
            )

        if bboxes is not None and len(bboxes) > 0:
            from matplotlib.patches import Rectangle
            bboxes = bboxes / self.down_shape[None, 0:1]
            bboxes = bboxes * (self.point_cloud_range[3] - self.point_cloud_range[0])[None] + self.point_cloud_range[None, 0]
            bboxes = bboxes.cpu().numpy()
            bboxes_w = bboxes[:,2] - bboxes[:,0]
            bboxes_h = bboxes[:,3] - bboxes[:,1]
            for i in range(len(bboxes)):
                rect = Rectangle(bboxes[i,0:2], bboxes_w[i], bboxes_h[i], fill=None, color='red')
                ax.add_patch(rect)
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


if __name__ =='__main__':
    pass

    # backbone = FocalSparseBEVBackBone()
    # backbone_dict = backbone.state_dict()

    # context = torch.load('ckpts/voxelnext_nuscenes_kernel1.pth')
    # context_dict = context['model_state']
    
    # renew_dict = { k : context_dict['backbone_3d.'+k] for k in  backbone_dict.keys() \
    #                if  ('backbone_3d.'+k) in context_dict}

    # backbone_dict.update( renew_dict)
    # backbone.load_state_dict( backbone_dict)
