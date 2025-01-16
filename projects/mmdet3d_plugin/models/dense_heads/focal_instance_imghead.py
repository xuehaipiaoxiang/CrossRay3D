# ------------------------------------------------------------------------
# Modified from mmdetection3d (https://github.com/open-mmlab/mmdetection3d)
# Copyright (c) OpenMMLab. All rights reserved.
# Modified by Huiming Yang
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
from mmcv.cnn import bias_init_with_prob
from mmcv.runner import force_fp32
from mmdet.core import (build_assigner, build_sampler, multi_apply,
                        reduce_mean, bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh)
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from ..utils.misc import  draw_heatmap_gaussian, apply_center_offset, apply_ltrb
from mmdet.core import bbox_overlaps
from mmdet3d.models.utils import clip_sigmoid
import mmcv
import cv2
import numpy as np
import os
from mmcv.image.photometric import imdenormalize
from einops import rearrange
from projects.mmdet3d_plugin.models.utils.misc import topk_gather
import torch.nn.functional as F
import copy
from mmdet3d.core import LiDARInstance3DBoxes




@HEADS.register_module()
class FocalInstanceImgHead(AnchorFreeHead): 
    """Implements the DETR transformer head.
    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.
    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        num_query (int): Number of query in Transformer.
        num_reg_fcs (int, optional): Number of fully-connected layers used in
            `FFN`, which is then used for the regression head. Default 2.
        transformer (obj:`mmcv.ConfigDict`|dict): Config for transformer.
            Default: None.
        sync_cls_avg_factor (bool): Whether to sync the avg_factor of
            all ranks. Default to False.
        positional_encoding (obj:`mmcv.ConfigDict`|dict):
            Config for position encoding.
        loss_cls (obj:`mmcv.ConfigDict`|dict): Config of the
            classification loss. Default `CrossEntropyLoss`.
        loss_bbox (obj:`mmcv.ConfigDict`|dict): Config of the
            regression loss. Default `L1Loss`.
        loss_iou (obj:`mmcv.ConfigDict`|dict): Config of the
            regression iou loss. Default `GIoULoss`.
        tran_cfg (obj:`mmcv.ConfigDict`|dict): Training config of
            transformer head.
        test_cfg (obj:`mmcv.ConfigDict`|dict): Testing config of
            transformer head.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Default: None
    """
    def __init__(self,
                 num_classes,
                 in_channels = 256,
                 embed_dims = 256,
                 down_stride = 16,
                 train_ratio=1.0,
                 infer_ratio=1.0,
                 topk_weight = 1.5,
                 sync_cls_avg_factor=True,
                 down_stride_shape = [50, 20],
                loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0), 
                loss_pts_topk=dict(
                     type='TopkLoss',
                     loss_weight=1.0), 
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):
        # NOTE here use `AnchorFreeHead` instead of `TransformerHead`,
        # since it brings inconvenience when the initialization of
        # `AnchorFreeHead` is called.
        self.sync_cls_avg_factor = sync_cls_avg_factor

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.out_channel = embed_dims
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.down_stride = down_stride
        self.down_stride_shape = down_stride_shape
        self.train_ratio=train_ratio
        self.infer_ratio=infer_ratio
        self.topk_weight = topk_weight

        super(FocalInstanceImgHead, self).__init__(num_classes, in_channels, init_cfg = init_cfg)

        # self.loss_img_cls = build_loss(loss_cls)
        self.loss_img_topk = build_loss(loss_pts_topk)


    def _init_layers(self):
        self.shared_semantic= nn.Sequential(
                                 nn.Conv2d(self.out_channel, self.out_channel, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.out_channel),
                                 nn.GELU(),
                                 )

        self.semantic_head = nn.Conv2d(self.out_channel, self.num_classes, kernel_size=1)


    def forward(self, img_feats):
  
        b, n, c, h, w= img_feats.shape

        sample_ratio = self.train_ratio if self.training else self.infer_ratio

        img_feats = img_feats.flatten(0, 1)

        semantic_feats = self.shared_semantic(img_feats)
        semantic_logit = self.semantic_head(semantic_feats)

        semantic_logit = rearrange(semantic_logit, 'b c h w ->  b (h w) c')

        semantic_score, cls_index = semantic_logit.max(dim = -1, keepdim=True)
        semantic_score = semantic_score.detach()

        img_feats = rearrange(img_feats, '(b n) c h w ->  b (n h w) c', n=n)

        num_tokens = n*h*w
        num_sample_tokens = int(num_tokens * sample_ratio)
        semantic_score_ = rearrange(semantic_score, '(b n) t c ->  b (n t) c', n=n)
        _, topk_indexes = torch.topk(semantic_score_.detach(), num_tokens, dim=1)
        img_feats = topk_gather(img_feats, topk_indexes[:, :num_sample_tokens, :])
        outs = {
            'img_feats' : img_feats,    
            'topk_indexes' : topk_indexes,
            'sample_weight' : semantic_score,
            'num_sample_tokens' : num_sample_tokens
        }
        if self.training:
            outs = self._pre_for_topk_loss(outs, semantic_logit, cls_index)
        return outs
    
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


    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             preds_dicts,
             img_metas,
             imgs = None, # slot for debug and visualization
             ):
        ''' gt in rv'''

        # make list all bs & n
        (gt_bboxes2d_list, gt_labels2d_list) = multi_apply(self._get_label_single, img_metas)
        gt_bboxes2d_list = [bboxes2d.cuda() for b in gt_bboxes2d_list for bboxes2d in b]
        gt_labels2d_list = [labels2d.cuda() for b in gt_labels2d_list for labels2d in b]

        gt_labels3d_list = [im['gt_labels_3d'].data.cuda() for im in img_metas for _ in range(6) ]

        nums_gt = [label.size(0) for label in gt_labels2d_list] # bn

        img_shape = self.down_stride_shape[1], self.down_stride_shape[0]

        gt_boxes = torch.cat(gt_bboxes2d_list)
        gt_center = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        gt_center_list = torch.split(gt_center, nums_gt, dim=0)


        labels_list = self.get_targets(gt_bboxes2d_list, gt_center_list, gt_labels2d_list, gt_labels3d_list, \
                                       bev_shape=img_shape)

        #################   START slot to visualize labels & uncomment slot in Detector

        # if imgs is not None:
        #     imgs = imgs.flatten(0, 1)
        #     imgs = [imgs[i].cpu().data.numpy().transpose(1, 2, 0).astype(np.float64)
        #              for i in range(imgs.size(0))]   
        # sample_weight = preds_dicts['sample_weight']

        # self._visualize(imgs, img_metas,
        #                     gt_bboxes2d_list, 
        #                     gt_labels2d_list,
        #                     centers2d_list = gt_center_list ,
        #                     mask_x = None,
        #                     labels_list = labels_list,
        #                     sample_weight = sample_weight,
        #                     choice ='gt')
        
        #################   END the slot to visualize ##########################################

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
        # loss_img_cls = self.loss_img_cls(flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        loss_img_topk = self.cal_loss_topk(preds_dicts, labels_list, gt_cls_num, avg_factor=num_pos)
        return {
            # 'loss_img_cls':loss_img_cls,
            'loss_img_topk':loss_img_topk * 0,

        } 


    def _get_label_single(self, img_meta):
        '''
        gt_bboxes2d_list, gt_labels2d_list
        '''
        return(
        img_meta['gt_bboxes'].data,
        img_meta['gt_labels'].data,
        )
        
    def get_targets(self, gt_boxes, gt_centers, gt_labels, gt_labels3d, bev_shape):
        (labels_list, ) = multi_apply(self._get_target_single, gt_boxes, gt_centers, gt_labels, \
                                      gt_labels3d, bev_shape=bev_shape)
        return labels_list

    def _get_target_single(self, gt_bboxes, gt_centers, gt_labels, gt_labels3d, bev_shape):
        INF = 1e8
        feat_h, feat_w = bev_shape
        shift_x = torch.arange(0, feat_w, device=gt_labels.device)
        shift_y = torch.arange(0, feat_h, device=gt_labels.device)
        yy, xx = torch.meshgrid(shift_y, shift_x, indexing='ij')
        xs, ys = xx.reshape(-1), yy.reshape(-1)
        num_points = xs.size(0)
        num_gts = gt_labels.size(0)

        if num_gts == 0:
            return (gt_labels.new_full((num_points,), self.num_classes), )

        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (gt_bboxes[:, 3] - gt_bboxes[:, 1])
        areas = areas[None].repeat(num_points, 1)

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)

        pxy_diff = (torch.rand((num_points, 2), device ='cuda') * 2 - 1 ) * 0.15

        xs = (xs + 0.5 + pxy_diff[:, 0]) * self.down_stride
        ys = (ys + 0.5 + pxy_diff[:, 1]) * self.down_stride

        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        areas[inside_gt_bbox_mask == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)
        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes  # set as BG

        gt_centers = gt_centers / self.down_stride
        gt_centers = (gt_centers.round()).to(torch.int64)
        gt_centers[:, 0] = gt_centers[:, 0].clip(min=0, max = self.down_stride_shape[0]-1)
        gt_centers[:, 1] = gt_centers[:, 1].clip(min=0, max = self.down_stride_shape[1]-1)
        gather_index = gt_centers[:, 0] + gt_centers[:, 1] * self.down_stride_shape[0]

        labels[gather_index] = gt_labels

        for i in range(self.num_classes):
            if (i in gt_labels) and (i not in gt_labels3d):
                labels[labels==i] = self.num_classes

        return (labels, )

    
    def get_targets_cls_nums_x(self, labels_list):
        gt_cls_num = multi_apply(self._get_targets_cls_nums_x, labels_list)
        return gt_cls_num
    
    def _get_targets_cls_nums_x(self, labels):
        batch_cls_num = []
        for i in range(self.num_classes):
            cls_cur = int( torch.sum(labels == i) )
            batch_cls_num.append(cls_cur)
        return (batch_cls_num, )
    

    def cal_loss_topk(self, preds_dicts, labels_list, gt_cls_num, avg_factor=None):
        '''
        distribution supervision
        '''
        topk_indexes = preds_dicts['topk_indexes']
        cls_index = preds_dicts['cls_index']
        all_batch_cls_num = preds_dicts['all_batch_cls_num']
        semantic_score = preds_dicts['sample_weight']
        semantic_logit = preds_dicts['semantic_logit']

        cls_index = rearrange(cls_index, '(b n) t c -> b (n t) c', n =6)
        semantic_score = rearrange(semantic_score, '(b n) t c -> b (n t) c', n =6)
        semantic_logit = rearrange(semantic_logit, '(b n) t c -> b (n t) c', n =6)

        bs, n = semantic_logit.shape[:2]

        semantic_score = topk_gather(semantic_score, topk_indexes)
        semantic_logit = topk_gather(semantic_logit, topk_indexes)
        cls_index = topk_gather(cls_index, topk_indexes)
        labels_list = torch.stack(labels_list, dim=0)
        labels_list = labels_list.unsqueeze(-1)
        labels_list = rearrange(labels_list, '(b n) t c -> b (n t) c', n =6)
        labels_list = topk_gather(labels_list, topk_indexes)

        weight = semantic_score.sigmoid()

        # slot vis
        # weight[0,:1000].flatten()[cls_index[0,:1000].flatten()==0]
        # weight.flatten()[cls_index.flatten()==1].topk(100,0)
        # slot vis


        all_batch_cls_num = torch.tensor(all_batch_cls_num, device='cuda').to(torch.int64)
        all_batch_cls_num = rearrange(all_batch_cls_num, '(b n) c -> b n c', n =6)
        all_batch_cls_num = torch.sum(all_batch_cls_num, dim=1)
        gt_cls_num = torch.tensor(gt_cls_num, device='cuda').to(torch.int64)
        gt_cls_num = rearrange(gt_cls_num, '(b n) c -> b n c', n =6)
        gt_cls_num = torch.sum(gt_cls_num, dim=1)

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

        loss_img_topk = self.loss_img_topk(flatten_semantic_logit, flatten_labels_list, flatten_weight, avg_factor)
        return loss_img_topk
    
    def _visualize(self, imgs, img_metas, gt_bboxes_list, gt_labels2d_list, centers2d_list,  \
                   mask_x = None, labels_list = None, sample_weight = None, \
                    choice ='gt'):
        '''
        Modified by Huiming Yang
        '''
        class_names = [
            'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
            'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
        ]
        
        LINE_INDICES = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7),
                            (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))

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

        assert choice in ('gt', 'pred') ,'error of choices'
        assert imgs is not None, 'error of canvas'
        thickness = 2
        out_dir = "vis/pictures_foreground"
        radius = 3
        mean=np.array([103.530, 116.280, 123.675])
        std=np.array([57.375, 57.120, 58.395])

        # img_shape = (800, 320)
        img_shape = (1600, 640)

        # TODO 3D box
        # for i, img_mt in enumerate(img_metas, 0):
        #     bboxes = img_mt["gt_bboxes_3d"].data.tensor.numpy()
        #     labels_ = img_mt["gt_labels_3d"].data.numpy()
        #     filenames = img_mt['filename']
        #     bboxes = LiDARInstance3DBoxes(bboxes, box_dim=9)
        #     for j, file_name in  enumerate(filenames, 0):
        #         k = np.random.randint(0,1000)
        #         fpath = os.path.join(out_dir, f"bs_{i}", f"camera_{j}_num{k}.png")
        #         canvas = imgs[i*6 + j]
        #         canvas = imdenormalize(canvas, mean, std, to_bgr = False)
        #         canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2BGR)
        #         canvas = canvas.astype(np.float64)
        #         if bboxes is not None and len(bboxes) > 0:
        #                 corners = bboxes.corners
        #                 num_bboxes = corners.shape[0]
        #                 coords = np.concatenate(
        #                     [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        #                 )
        #                 transform=img_mt["lidar2img"][j]
        #                 transform = copy.deepcopy(transform).reshape(4, 4)
        #                 coords = coords @ transform.T
        #                 coords = coords.reshape(-1, 8, 4)
        #                 indices = np.all(coords[..., 2] > 0, axis=1)
        #                 coords = coords[indices]
        #                 labels = labels_[indices]
        #                 indices = np.argsort(-np.min(coords[..., 2], axis=1))
        #                 coords = coords[indices]
        #                 labels = labels[indices]
        #                 coords = coords.reshape(-1, 4)
        #                 coords[:, 2] = np.clip(coords[:, 2], a_min=1e-5, a_max=1e5)
        #                 coords[:, 0] /= coords[:, 2]
        #                 coords[:, 1] /= coords[:, 2]
        #                 coords = coords[..., :2].reshape(-1, 8, 2)
        #                 for index in range(coords.shape[0]):
        #                     name = class_names[labels[index]]
        #                     for start, end in LINE_INDICES:
        #                         cv2.line(
        #                             canvas,
        #                             coords[index, start].astype(np.int),
        #                             coords[index, end].astype(np.int),
        #                             OBJECT_PALETTE[name],
        #                             thickness,
        #                             cv2.LINE_AA,
        #                         )
        #         canvas = canvas.astype(np.uint8)


        # TODO (1)2D box ,(2)TOPK-label and (3)TOPK-select
        for i, img_mt in enumerate(img_metas, 0):
            filenames = img_mt['filename']
            for j, file_name in  enumerate(filenames, 0):
                canvas = imgs[i*6 + j]
                canvas = imdenormalize(canvas, mean, std, to_bgr = False)
                canvas = cv2.cvtColor(canvas.astype(np.uint8), cv2.COLOR_RGB2BGR)
                canvas = canvas.astype(np.float64)
                gt_bboxes = gt_bboxes_list[i*6 + j]
                gt_bboxes = gt_bboxes.round()
                gt_labels2d =gt_labels2d_list[i*6 + j]
                centers2d = centers2d_list[i*6 + j]
                centers2d = centers2d.round()
                num_gt = gt_bboxes.size(0)
                k = np.random.randint(0,1000)
                fpath = os.path.join(out_dir, f"bs_{i}", f"camera_{j}_num{k}.png")

                # if choice == 'gt': # TODO 2D
                #     for k in range( num_gt):
                #             palette2 = OBJECT_PALETTE[class_names[gt_labels2d[k]]]
                #             pts = gt_bboxes[k].cpu().data.numpy().astype(np.int64)
                #             center_pt = centers2d[k].cpu().data.numpy().astype(np.int64)
                #             cv2.rectangle(
                #                 canvas,
                #                 pts[:2],
                #                 pts[2:],
                #                 palette2, 
                #                 thickness,
                #                 # cv2.LINE_AA,
                #             )
                #             cv2.circle(canvas,center_pt, radius, palette2, -1)

                # if labels_list is not None: #TODO (2)TOPK-label 
                #     labels = labels_list[i*6 + j]
                #     for h in range(img_shape[1]):
                #         for w in range(img_shape[0]):
                #             index = int(h / self.down_stride)* self.down_stride_shape[0]  + int(w / self.down_stride)
                #             index_c = labels[index]
                #             if index_c==10:
                #                 continue
                #             palette2 = OBJECT_PALETTE[class_names[index_c]]
                #             palette2 = np.array(palette2).clip(min=10, max=245)
                #             canvas[h][w] = canvas[h][w] * palette2 /255

                if sample_weight is not None: #TODO TOPK-select
                    sample_score = sample_weight[i*6 + j]
                    num_tokens = sample_score.size(0)
                    num_sample_tokens = int(sample_score.size(0) * 0.75)
                    _, topk_indexes = torch.topk(sample_score.detach(), num_sample_tokens, dim=0)
                    mask_x = topk_indexes.new_zeros((num_tokens, 1), requires_grad = False,)
                    mask_src = topk_indexes.new_ones((num_sample_tokens, 1), requires_grad = False)
                    mask_x.scatter_add_(0, topk_indexes, mask_src)
                    for h in range(img_shape[1]):
                        for w in range(img_shape[0]):
                            index = int(h / self.down_stride)* self.down_stride_shape[0]  + int(w / self.down_stride)
                            index_c = mask_x[index]
                            if index_c==0:
                                palette2 = canvas[h][w] + 150
                                palette2 = np.array(palette2).clip(min=120, max=237)
                                # palette2 = np.array(palette2).clip(min=237, max=237)
                                canvas[h][w] = palette2

                # TODO image write
                canvas = canvas.astype(np.uint8)
                canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
                mmcv.mkdir_or_exist(os.path.dirname(fpath))
                mmcv.imwrite(canvas, fpath)




