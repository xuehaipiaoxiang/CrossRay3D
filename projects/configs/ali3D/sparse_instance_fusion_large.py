# ratio ::= the keep fore(the higher score) / total all 
plugin=True
plugin_dir='projects/mmdet3d_plugin/'
point_cloud_range = [-54.0, -54.0, -5.0, 54.0, 54.0, 3.0]
voxel_size = [0.075, 0.075, 0.2]
sparse_shape = [1440, 1440, 41]
max_voxels = (120000, 160000)
stride_used = 16
out_size_factor = 8
# evaluation = dict(interval=20)
dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'

class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)
collection_focal = ('gt_bboxes', 'gt_labels', 'centers2d', 'depths', 'cam_intrinsic')
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False)

ida_aug_conf = {
        "resize_lim": (0.94, 1.25),
        "final_dim": (640, 1600),
        "bot_pct_lim": (0.0, 0.0),
        "rot_lim": (0.0, 0.0),
        "H": 900,
        "W": 1600,
        "rand_flip": True,
    }


train_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
        with_bbox=True, with_label=True, with_bbox_depth=True),

    dict(
        type='UnifiedObjectSample',
        sample_2d=True,
        mixup_rate=0.5,
        db_sampler=dict(
            type='UnifiedDataBaseSampler',
            data_root = 'data/nuscenes/',    
            info_path=data_root + 'nuscenes_dbinfos_train.pkl',
            rate=1.0,
            prepare=dict(
                filter_by_difficulty=[-1],
                filter_by_min_points=dict(
                    car=5,
                    truck=5,
                    bus=5,
                    trailer=5,
                    construction_vehicle=5,
                    traffic_cone=5,
                    barrier=5,
                    motorcycle=5,
                    bicycle=5,
                    pedestrian=5)),
            classes=class_names,
            sample_groups=dict(
                car=2,
                truck=3,
                construction_vehicle=7,
                bus=4,
                trailer=6,
                barrier=2,
                motorcycle=6,
                bicycle=6,
                pedestrian=2,
                traffic_cone=2),
            points_loader=dict(
                type='LoadPointsFromFile',
                coord_type='LIDAR',
                load_dim=5,
                use_dim=[0, 1, 2, 3, 4],
            ))),
    
    dict(
        type='GlobalRotScaleTransAll',
        rot_range=[-0.3925 * 2, 0.3925 * 2],
        scale_ratio_range=[0.9, 1.1],
        translation_std=[0.5, 0.5, 0.5]),

    dict(
        type='CustomRandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5),

    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='PointShuffle'),
    dict(type='CustomResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=True, with_2d = True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
            meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'rot_degree',
                    'gt_bboxes_3d', 'gt_labels_3d') + collection_focal 
    )
]
test_pipeline = [
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(
        type='LoadPointsFromMultiSweeps',
        sweeps_num=10,
        use_dim=[0, 1, 2, 3, 4],
    ),
    dict(type='LoadMultiViewImageFromFiles'),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1.0, 1.0],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(type='CustomResizeCropFlipImage', data_aug_conf = ida_aug_conf, training=False),
            dict(type='NormalizeMultiviewImage', **img_norm_cfg),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points', 'img'],
                    meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img',
                    'depth2img', 'cam2img', 'pad_shape',
                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                    'transformation_3d_flow', 'rot_degree',
                    'gt_bboxes_3d', 'gt_labels_3d') + collection_focal 
                )
        ])
]
data = dict(
    samples_per_gpu = 3,
    workers_per_gpu = 6,
    train=dict(
        type='CBGSDataset',
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            ann_file=data_root + '/nuscenes_infos_val.pkl',
            # ann_file= [data_root + '/nuscenes_infos_val.pkl', data_root + '/nuscenes_infos_train.pkl'],
            load_interval=1,
            pipeline=train_pipeline,
            classes=class_names,
            modality=input_modality,
            test_mode=False,
            box_type_3d='LiDAR')),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + '/nuscenes_infos_val.pkl',
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
        # ann_file=data_root + '/nuscenes_infos_test.pkl',
        load_interval=1,
        pipeline=test_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=True,
        box_type_3d='LiDAR'))
model = dict(
    type='Ali3DDetector',
    use_grid_mask=True,
    use_3d_mask = True,

img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=-1,
        input_ch=3,
        out_features=('stage4','stage5',)),
    img_neck=dict(
        type='CPFPN',
        in_channels=[768, 1024],
        out_channels=256,
        num_outs=2),
    img_roi_head=dict(
        type='FocalInstanceImgHead',
        num_classes=10,
        in_channels = 256,
        train_ratio = 0.4, 
        infer_ratio = 0.4, 
        topk_weight = 1,
        down_stride_shape = [100, 40],
        stride  = stride_used,
        loss_pts_topk=dict(
        type='TopkLoss',
        loss_weight=1.0), 
    ),
    # img_roi_head=dict( # for auxiliary supervision only
    #     type='YOLOXHeadCustom',
    #     num_classes=10,
    #     in_channels=256,
    #     strides=[8, 16, 32, 64],
    #     train_cfg=dict(assigner=dict(type='SimOTAAssigner', center_radius=2.5)),
    #     test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.65)),),
    depth_branch=dict( # for auxiliary supervision only
        type='MyDepthSegHead',
        scale_num=4,
        keep_threshold=0.1,
        grid_config=grid_config,
        input_size=ida_aug_conf['final_dim'],
        depth_loss_cfg=dict(alpha=0.25, gamma=2),
        fg_loss_cfg=dict(alpha=0.25, gamma=2),
        ins_channels=[256, 256, 256, 256],
        depthnet_cfg=dict(use_dcn=False, aspp_mid_channels=96),
        loss_depth_weight=[37, 75, 150, 300],
        loss_fg_weight=[8, 16, 33, 67],
        downsamples=[64, 32, 16, 8]),
    pts_voxel_layer=dict(
        num_point_features=5,
        max_num_points=10,
        voxel_size=voxel_size,
        max_voxels=(120000, 160000),
        point_cloud_range=point_cloud_range),

    pts_voxel_encoder=dict(
        type='HardSimpleVFE',
        num_features=5,
    ),

    pts_backbone=dict(
        type='FocalSparseBEVBackBoneMedian',
        sparse_shape=sparse_shape,
        base_channels=32,
        output_channels=256,
        encoder_channels=((32, 32), (64, 64, 64), (128, 128, 128), (256, 256, 256),(256, 256, 256),(256, 256, 256)),
        sparse_conv_kernel = (5, 3, 3, 3, 3),
        order=('conv', 'norm', 'act'),
        block_type='basicblock',
        in_channels = 5,
        grid_size = sparse_shape,
        point_cloud_range = point_cloud_range,
        voxel_size = voxel_size,
        train_ratio = 0.3, 
        infer_ratio = 0.3, 
        topk_weight = 1,
        loss_pts_topk=dict(
        type='TopkLoss',
        loss_weight=1.0),         
        ),
    
    pts_bbox_head=dict(
        type='SparseFusionHead',
        grid_size = sparse_shape,
        down_img_shape = [40, 100],
        in_channels=256,
        hidden_dim=256,
        bbox_coder=dict(
            type='MultiTaskBBoxCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        transformer=dict(
            type='AliTransformer',
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
                # cls_cost=dict(type='ClassificationCost', weight=2.0),
                cls_cost=dict(type='FocalLossCost', weight=2.0),
                reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
                iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
                pc_range=point_cloud_range,
                code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            ),
            pos_weight=-1,
            gaussian_overlap=0.1,
            min_radius=2,
            voxel_size=voxel_size,
            grid_size=sparse_shape,
            out_size_factor=out_size_factor,
            code_weights=[2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
            point_cloud_range=point_cloud_range)),

    test_cfg=dict(
        pts=dict(
            dataset='nuScenes',
            grid_size=sparse_shape,
            out_size_factor=out_size_factor,
            pc_range=point_cloud_range,
            voxel_size=voxel_size,
            nms_type=None,
            nms_thr=0.2,
            use_rotate_nms=True,
            max_num=200
            )
        )
)

optimizer = dict(
    type='AdamW',
    lr=0.00003,
    # lr=0.00001,
    # lr=0.00012,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone' : dict(lr_mult=0.01, decay_mult=2),
            # 'img_neck' : dict(lr_mult=0.5),
            # 'img_roi_head' : dict(lr_mult=0.01),
            'pts_backbone' : dict(lr_mult=0.01, decay_mult=2),
            # 'pts_bbox_head' : dict(lr_mult=0.25,  decay_mult=2),
        }),
    weight_decay=0.01)  # for 8gpu * 2sample_per_gpu
optimizer_config = dict(
    type='CustomFp16OptimizerHook',
    loss_scale='dynamic',
    grad_clip=dict(max_norm=35, norm_type=2),
    custom_fp16=dict(pts_voxel_encoder=False, pts_bbox_head=False))

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

total_epochs = 5
checkpoint_config = dict(interval = 1, max_keep_ckpts = 5)

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='CustomSetEpochInfoHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = None

load_from = None
resume_from = None



workflow = [('train', 1)]
gpu_ids = range(0, 8)
