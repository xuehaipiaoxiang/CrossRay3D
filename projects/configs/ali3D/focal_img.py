plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# ratio ::= the keep fore(the higher score) / total all 

point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]
sparse_shape = [1440, 1440, 40]

max_voxels = (120000, 160000)

stride_used = 16

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

out_size_factor = 8
evaluation = dict(interval=20)
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

collection_focal = ('gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'cam_intrinsic')


img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)

#CMT
ida_aug_conf = {
        "resize_lim": (0.47, 0.625),
        "final_dim": (320, 800),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0), # must sure
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }

train_pipeline = [

    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    # dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
        with_bbox=True,with_label=True, with_bbox_depth=True),
    dict(type='GlobalRotScaleTransImage',
        rot_range=[-0.3925, 0.3925],
        translation_std=[0, 0, 0],
        scale_ratio_range=[0.95, 1.05],
        reverse_angle=True,
        training=True,
        ),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),

    dict(type='CustomResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True, with_2d = True),

    dict(type='NormalizeMultiviewImage', **img_norm_cfg),

    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img', 'gt_bboxes_3d', 'gt_labels_3d'] ,
         meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'rot_degree',
                    'gt_bboxes_3d', 'gt_labels_3d') + collection_focal )
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='CustomResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img'],
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'rot_degree',
                    'gt_bboxes_3d', 'gt_labels_3d') + collection_focal 
                 ),
        ])
]

data = dict(
    samples_per_gpu = 2,
    workers_per_gpu = 6,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '/nuscenes_infos_train.pkl',
            # ann_file=data_root + '/nuscenes_infos_val.pkl',
            # ann_file=[data_root + '/nuscenes_infos_val.pkl', data_root + '/nuscenes_infos_train.pkl'],
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR',
            )
    ),

    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        # ann_file=data_root + '/nuscenes_infos_train.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'),

    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
        # ann_file=data_root + '/nuscenes_infos_train.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        # pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        # test_mode=False,
        box_type_3d='LiDAR'), 
        )

model = dict(
    type='SparseASformerDetector',

    use_grid_mask=True,
    stride = stride_used,

    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(2, 3),
        frozen_stages=-1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        with_cp=True,
        style='pytorch'),

    img_neck=dict(
        type='CPFPN',
        in_channels=[1024, 2048],
        out_channels=256,
        num_outs=2),
    
    img_roi_head=dict(
        type='FocalSparseImgHead',
        num_classes=10,
        in_channels = 256,
        train_ratio = 1, 
        infer_ratio = 1, 
        stride  = stride_used,
        loss_cls2d=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=2.0),
        loss_centerness=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox2d=dict(type='L1Loss', loss_weight=5.0),
        loss_iou2d=dict(type='GIoULoss', loss_weight=2.0),
        loss_centers2d=dict(type='L1Loss', loss_weight=10.0),
        train_cfg=dict(
        assigner2d=dict(
            type='HungarianAssigner2D',
            # foreground cls_cost is used in assigner2d
            cls_cost=dict(type='FocalLossCost', weight=2.),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0),
            centers2d_cost=dict(type='BBox3DL1Cost', weight=10.0)))
    ),

    pts_bbox_head=dict(
        type='SparseImgHead',
        in_channels=256,
        hidden_dim=256,
        downsample_scale=8,
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        # separate_head=dict(
        #     type='SeparateTaskHead', init_bias=-2.19, final_kernel=1),
        transformer=dict(
            type='AsfTransformer',
            decoder=dict(
                type='PETRTransformerDecoder',
                return_intermediate=True,
                num_layers=6,
                transformerlayers=dict(
                    type='PETRTransformerDecoderLayer',
                    with_cp=False,
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='PETRMultiheadFlashAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        ],
                    ffn_cfgs=dict(
                        type='FFN',
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.,
                        act_cfg=dict(type='ReLU', inplace=True),
                    ),

                    feedforward_channels=1024, #unused
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')),
            )),
        loss_cls=dict(type='FocalLoss', use_sigmoid=True, gamma=2, alpha=0.25, reduction='mean', loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        loss_heatmap=dict(type='GaussianFocalLoss', reduction='mean', loss_weight=1.0),
    ),

    train_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            assigner=dict(
                type='HungarianAssigner3D',
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            grid_size = sparse_shape,
            voxel_size=voxel_size,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),

    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size = sparse_shape,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=200,
            )
        )
)

optimizer = dict(
    type='AdamW',
    # lr=0.00014,
    # lr=0.00014, adapt to single GPU
    # lr=0.000035,
    lr=0.0002,
    # lr=0.0002,
    # lr=0.0,
    # lr=0.000025,
    # lr=0.0000175,

    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.01, decay_mult=5),
            'img_neck': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(
    type='CustomFp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2),
    # custom_fp16=dict(pts_voxel_encoder=False, pts_bbox_head=False)
    )

lr_config = dict(
    policy='cyclic',
    target_ratio=(2, 0.0001),

    cyclic_times=1,
    step_ratio_up=0.4)
momentum_config = dict(
    policy='cyclic',
    target_ratio=(0.8947368421052632, 1),
    cyclic_times=1,
    step_ratio_up=0.4)



total_epochs = 20
checkpoint_config = dict(interval = 1, max_keep_ckpts = 5)

log_config = dict(
    interval = 50, #  frequently 
hooks=[dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='CustomSetEpochInfoHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None

load_from='ckpts/nuscenes_res50_voxel_backbone.pth'
# load_from='ckpts/nuim_r50.pth'


resume_from = None
# resume_from = 'work_dirs/asf_r50_sparse_imgOnly/epoch_18.pth'

workflow = [('train', 1)]
gpu_ids = range(0, 8)

# dn 16 0.0002 2 
