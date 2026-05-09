auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
num_classes = 4
base_lr = 0.004
data_root = '/netscratch/millerdurai/Datasets/COCO_KPTS/2017/COCO_WholeBody'
dataset_type = 'CombinedDataset'
arm_anno_dir = '/netscratch/millerdurai/Datasets/COCO_KPTS/2017'
arm_anno_path = '/fscratch/millerdurai/Datasets/arctic/arm_annotations_with_hands.json'

# train_batch_size = 72
train_batch_size = 96
test_batch_size = 32
num_workers = 16

# train_batch_size = 32
# num_workers = 0

pin_memory = False
persistent_workers = True
prefetch_factor = 8 if num_workers > 0 else None
persistent_workers = persistent_workers if num_workers > 0 else False

pretrained_checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/rtmdet/cspnext_rsb_pretrain/cspnext-tiny_imagenet_600e.pth'
val_interval = 10
stage2_num_epochs = 5
max_epochs = 900

# load_from = '/cmillerd/Projects/Aria/arm_detection/mmdet_test/work_dirs/rtmdet_tiny_8xb32-300e_combined_cutmix/epoch_40.pth'
resume = True


default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
interval = 10

log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)

model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.167,
        expand_ratio=0.5,
        init_cfg=dict(
            checkpoint=pretrained_checkpoint,
            prefix='backbone.',
            type='Pretrained'),
        norm_cfg=dict(type='SyncBN'),
        type='CSPNeXt',
        widen_factor=0.375),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        # exp_on_reg=False,
        feat_channels=96,
        in_channels=96,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(type='SyncBN'),
        num_classes=num_classes,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type='RTMDetInsSepBNHead',
        loss_mask=dict(
            type='SmoothL1Loss', reduction='mean',
            loss_weight=1e-2),
        with_objectness=False),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            96,
            192,
            384,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=1,
        out_channels=96,
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        max_per_img=30,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
optim_wrapper = dict(
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=150,
        begin=150,
        by_epoch=True,
        convert_to_iter_based=True,
        end=max_epochs,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]

test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=test_batch_size,
    dataset=dict(
        type=dataset_type,
        ann_file=arm_anno_dir + '/val_arm_annotations_with_hands.json',
        backend_args=None,
        data_prefix=dict(subset='val'),
        data_root=data_root,
        pipeline=[
            dict(backend_args=None, type='LoadImageFromDataPipe'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(
                pad_val=dict(img=(
                    114,
                    114,
                    114,
                )),
                size=(
                    640,
                    640,
                ),
                type='Pad'),
            dict(type='LoadAnnotationsFromJSONS', with_bbox=True, with_keypoints=True, backend_args=dict(json_paths=[arm_anno_dir + '/val_arm_annotations_with_hands.json', 
                                                                                                                     arm_anno_path])),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True),
    drop_last=False,
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    sampler=dict(shuffle=False, type='DefaultSampler'))
    
test_evaluator = dict(
    ann_file=arm_anno_dir + '/val_arm_annotations_with_hands.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    proposal_nums=(
        100,
        1,
        10,
    ),
    type='EgoWholeBodyMetric')

train_pipeline = [
    dict(backend_args=None, type='LoadImageFromDataPipe'),
    dict(type='LoadAnnotationsFromJSONS', with_bbox=True, with_keypoints=True, backend_args=dict(json_paths=[arm_anno_dir + '/train_arm_annotations_with_hands.json',
                                                                                                             arm_anno_path])),
    dict(keep_ratio=True,
        ratio_range=(
            0.9,
            2.5,
        ),
        scale=(
            640,
            640,
        ),
        type='RandomResizeWithKeypoints',
    ),
    dict(crop_size=(
        640,
        640,
    ), type='RandomCropWithKeypoints'),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='HandColorAugmentation', prob=0.5,),
    # dict(prob=0.5,
    #      max_rotate_degree=360,
    #      max_translate_ratio=0.2,
    #      scaling_ratio_range=(1.0, 2.5),
    #      max_shear_degree=20,
    #      type='RandomAffineWithKeypoints'),
    dict(
        img_scale=(
            640,
            640,
        ),
        max_cached_images=10,
        prob=0.5,
        random_pop=False,
        ratio_range=(
            0.5,
            0.9,
        ),
        type='CutMixWithKeypoints'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(type='PackDetInputs'),
]

train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromDataPipe'),
    dict(type='LoadAnnotationsFromJSONS', with_bbox=True, with_keypoints=True, backend_args=dict(json_paths=[arm_anno_dir + '/train_arm_annotations_with_hands.json', 
                                                                                                             arm_anno_path])),
    dict(keep_ratio=True,
        ratio_range=(
            0.9,
            2.5,
        ),
        scale=(
            640,
            640,
        ),
        type='RandomResizeWithKeypoints',
    ),
    dict(crop_size=(
        640,
        640,
    ), type='RandomCropWithKeypoints'),
    dict(type='YOLOXHSVRandomAug'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(type='PackDetInputs'),
]


test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(pad_val=dict(img=(
        114,
        114,
        114,
    )), size=(
        640,
        640,
    ), type='Pad'),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]

train_cfg = dict(
    dynamic_intervals=[
        (
            max_epochs - 2 * stage2_num_epochs,
            1,
        ),
    ],
    max_epochs=max_epochs,
    type='EpochBasedTrainLoop',
    val_interval=val_interval)


train_dataloader = dict(
    batch_sampler=None,
    batch_size=train_batch_size,
    dataset=dict(
        type=dataset_type,
        ann_file=arm_anno_dir + '/train_arm_annotations_with_hands.json',
        backend_args=None,
        data_prefix=dict(subset='train'),
        data_root=data_root,
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline),
    num_workers=num_workers,
    prefetch_factor=prefetch_factor,
    persistent_workers=persistent_workers,
    pin_memory=pin_memory,
    sampler=dict(shuffle=True, type='DefaultSampler'))


val_cfg = dict(type='ValLoop')
val_dataloader = test_dataloader
val_evaluator = test_evaluator

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=vis_backends)

visualization=dict( # user visualization of validation and test results
    type='DetVisualizationHook',
    draw=True,
    interval=20,
    test_out_dir='valeval',
    show=False
    )


default_hooks = dict(
    checkpoint=dict(interval=val_interval, max_keep_ckpts=5, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=visualization)


custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2,
        type='PipelineSwitchHook'),
]

