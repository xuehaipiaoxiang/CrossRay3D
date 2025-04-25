# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# ------------------------------------------------------------------------
# use slot to do vis
# ------------------------------------------------------------------------
# Aligned 3D: Modality Alignment Sparse Transformer
# ------------------------------------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mmcv.runner import force_fp32, auto_fp16
from mmdet.core import multi_apply
from mmdet.models import DETECTORS
from mmdet.models.builder import build_backbone
from mmdet3d.core import (Box3DMode, Coord3DMode, bbox3d2result,
                          merge_aug_bboxes_3d, show_result)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from projects.mmdet3d_plugin.models.utils.grid_mask import GridMask
from einops import rearrange


@DETECTORS.register_module()
class Ali3DDetector(MVXTwoStageDetector):

    def __init__(self,
                 use_grid_mask=False,
                 use_3d_mask=False,
                 **kwargs):
        
        self.pts_voxel_cfg = kwargs.get('pts_voxel_layer', None)
        kwargs['pts_voxel_layer'] = None
        super(Ali3DDetector, self).__init__(**kwargs)
        self.use_grid_mask = use_grid_mask
        self.use_3d_mask = use_3d_mask
        self.grid_mask = GridMask(True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        if self.pts_voxel_cfg:
            from projects.mmdet3d_plugin import SPConvVoxelization
            self.pts_voxel_layer = SPConvVoxelization(**self.pts_voxel_cfg)

    def init_weights(self):
        """Initialize model weights."""
        super(Ali3DDetector, self).init_weights()
    

    @auto_fp16(apply_to=('img'), out_fp32=True) 
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""

        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            if  not isinstance(img_metas, (list, tuple) ):
                img_metas = [ img_metas ]
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
            else:
                B, N = 1, img.size(0)

            if self.use_grid_mask:
                img = self.grid_mask(img)
            
            # # to mask
            # img[0] = 0

            img_feats = self.img_backbone(img.float())

            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        view_list = [B, N, *img_feats.shape[-3:]] 
        img_feats = img_feats.view(view_list)
        return img_feats

    @force_fp32(apply_to=('pts', 'img_feats'))
    def extract_pts_feat(self, pts, img_feats, img_metas):
        """
        Extract features of points.
        reference to spconv
        # features =  your features with shape [N, num_channels]
        # indices =  your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
        # spatial_shape =  spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
        # batch_size =  batch size of your sparse tensor.
        
        """
        if not self.with_pts_bbox or pts is None:
            return None
        voxels, num_points, coors = self.voxelize(pts)

        # to mask 
        # voxels[((coors[:, 2] - 400)<0 ) + (coors[:, 3] - 400 <0)] = 0

        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,)
        batch_dict = dict(voxel_features=voxel_features, voxel_coords=coors, batch_size=coors[-1,0]+1)
        out_dict = self.pts_backbone(batch_dict)
        out_dict['raw_x'] = pts # slot for pts 
        return out_dict


    def forward_train(self,
                      points=None,
                      img=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_bboxes_ignore=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """
        losses = dict()

        img_feats, pts_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        
        if  pts_dict is not None and pts_dict['topk_indexes'] is not None :
            loss_focal_pillar= self.pts_backbone.loss(pts_dict, img_metas,
                                                      raw_x = pts_dict['raw_x'], # slot for pts 
                                                       )
            if loss_focal_pillar is not None:
                losses.update(loss_focal_pillar)
        
        if  pts_dict is not None or img_feats is not None:
            losses_pts = self.forward_pts_train(pts_dict, img_feats, gt_bboxes_3d, gt_labels_3d, img_metas,
                                                # imgs = img, # slot for imgs
                                                )
            losses.update(losses_pts)

        return losses

    @force_fp32(apply_to=('pts_feats', 'img_feats'))
    def forward_pts_train(self,
                          pts_dict,
                          img_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                        #   imgs = None, # slot for imgs
                          ):
        """
        """
        pts_dict = [None] if pts_dict is None else [pts_dict]
        if img_feats is None:
            img_dict = [None]
        else:
            img_dict = self.img_roi_head(img_feats)
            loss_focal_imgs = self.img_roi_head.loss(img_dict, img_metas,
                                                # imgs = imgs, # slot for imgs
                                                )            
            img_dict = [img_dict]
        
        # Modal Mask 3D
        if self.use_3d_mask:
            seed = np.random.rand()
            if seed > 0.75:
                img_dict[0]['img_feats'] = img_dict[0]['img_feats'] * 0.0
            elif seed > 0.5:
                pts_dict[0]['pillar_feats'] = pts_dict[0]['pillar_feats'] * 0.0

        outs = self.pts_bbox_head(pts_dict, img_dict, img_metas)
        losses = self.pts_bbox_head.loss(gt_bboxes_3d, gt_labels_3d, outs)
        if img_feats is not None:
            losses.update(loss_focal_imgs)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """
        if points is None:
            points = [None]
        if img is None:
            img = [None]
        return self.simple_test(points[0], img_metas[0], img[0])
        # return self.aug_test(points, img_metas, img)

    
    
    @force_fp32(apply_to=('x', 'x_img'))
    def simple_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch."""
        pts_dict = x
        if x_img is None:
            img_dict = [None]
        else:
            img_dict = self.img_roi_head(x_img)
            img_dict = [ img_dict ]

        # to mask 
        # pts_dict[0]['pillar_feats'] = pts_dict[0]['pillar_feats'] * 0.0
        # img_dict[0]['img_feats'] = img_dict[0]['img_feats'] * 0.0

        outs = self.pts_bbox_head(pts_dict, img_dict, img_metas)
        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ] 
        return bbox_results


    def simple_test(self, points, img_metas, img=None, rescale=False):

        img_feats, pts_dict = self.extract_feat(
            points, img=img, img_metas=img_metas)
        
        pts_dict = [ pts_dict ]
        bbox_list = [dict() for i in range(len(img_metas))]

        bbox_pts = self.simple_test_pts(
            pts_dict, img_feats, img_metas, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox

        return bbox_list
    
    def set_epoch(self, epoch, max_epoch): 
        '''
            H00K
        '''
        if self.with_img_roi_head:
            self.img_roi_head.epoch = epoch 
            self.img_roi_head.total_epoch = max_epoch 
        if self.with_pts_backbone:
            self.pts_backbone.epoch = epoch 
            self.pts_backbone.total_epoch = max_epoch 

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        """Apply dynamic voxelization to points.

        Args:
            points (list[torch.Tensor]): Points of each sample.

        Returns:
            tuple[torch.Tensor]: Concatenated points, number of points
                per voxel, and coordinates.
        """
        voxels, coors, num_points = [], [], []
        for res in points:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        return voxels, num_points, coors_batch


    def aug_test(self, points, img_metas, img=None, rescale=False):
        """Test function with augmentaiton."""
        # only support aug_test for one sample

        img_list, pts_list = multi_apply(self.extract_feat, points, img, img_metas)
        bbox_pts = self.aug_test_pts(
            pts_list, img_list, img_metas, rescale=rescale)
        bbox_list = []
        bbox_list.append({'pts_bbox': bbox_pts})
        return bbox_list

    def aug_test_pts(self, x, x_img, img_metas, rescale=False):
        """Test function of point cloud branch with augmentaiton."""
        # only support aug_test for one sample
        pts_dicts = x
        img_dicts = list()
        for img in x_img:
            img_dicts.append(self.img_roi_head(img))

        aug_bboxes = []
        for pts_dict, img_dict, img_meta in zip(pts_dicts, img_dicts, img_metas):
            img_dict = [ img_dict ]
            pts_dict = [ pts_dict ]
            outs = self.pts_bbox_head(pts_dict, img_dict, img_meta)
            bbox_list = self.pts_bbox_head.get_bboxes(
                outs, img_meta, rescale=rescale)
            bbox_list = [
                dict(boxes_3d=bboxes, scores_3d=scores, labels_3d=labels)
                for bboxes, scores, labels in bbox_list
            ]
            aug_bboxes.append(bbox_list[0])
        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes = merge_aug_bboxes_3d(aug_bboxes, img_metas, self.pts_bbox_head.test_cfg)
        return merged_bboxes

        
        



        

