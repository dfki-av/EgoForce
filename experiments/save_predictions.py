import collections
import argparse
import os
import sys
import pickle
import torch
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from datasets import HO3DV2Loader, HOT3DLoader, ArcticLoader, H2OLoader, Arm3DDataset, AnyCalibDatasetPin, AnyCalibDataset624
from camera_models import PinholeCameraPytorch3D, Rational8CameraPytorch3D, FishEyeCamera624Pytorch3D, KannalaBrandtK3CameraPytorch3D, EquisolidCameraPytorch3D, EquirectangularCameraPytorch3D, StereographicCameraPytorch3D
from models import HALOAblations, LimbModel, DepthModelWrapper, DGPModelWrapper
from core import compute_camera_space_mesh, get_limb

from settings import config as cfg 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


HOT3D_DATASET_NAMES = {
    'HOT3D',
    'HOT3D_PER',
    'HOT3D_PINHOLE',
    'HOT3D_EQUISOLID',
    'HOT3D_EQUIRECTANGULAR',
    'HOT3D_STEREOGRAPHIC',
}

LEGACY_HOT3D_MODE_BY_DATASET = {
    'HOT3D': 'none',
    'HOT3D_PER': 'pinhole',
    'HOT3D_PINHOLE': 'pinhole',
    'HOT3D_EQUISOLID': 'equisolid',
    'HOT3D_EQUIRECTANGULAR': 'equirectangular',
    'HOT3D_STEREOGRAPHIC': 'stereographic',
}

HOT3D_SUFFIX_BY_MODE = {
    'none': 'undistort_inp_true',
    'pinhole': 'undistort_inp_false_pinhole',
    'equisolid': 'undistort_inp_false_equisolid',
    'equirectangular': 'undistort_inp_false_equirectangular',
    'stereographic': 'undistort_inp_false_stereographic',
}



def _cfg_get(path, default):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Run EgoForce predictions and save outputs as a pickle file.')
    parser.add_argument(
        '--test-dataset-name',
        default=_cfg_get(['DATASET', 'TEST_NAME'], 'ARCTIC'),
        choices=[
            'ARCTIC',
            'H2O',
            'HO3D',
            'HOT3D',
            'HOT3D_PER',
            'HOT3D_PINHOLE',
            'HOT3D_EQUISOLID',
            'HOT3D_EQUIRECTANGULAR',
            'HOT3D_STEREOGRAPHIC',
        ],
        help='Evaluation dataset name. Legacy HOT3D_* names are supported.',
    )
    parser.add_argument(
        '--checkpoint-path',
        default=_cfg_get(['POSE_3D', 'CHECKPOINT_PATH'], ''),
        help='Model checkpoint path. Required if not set in config.',
    )
    parser.add_argument(
        '--hot3d-conversion',
        default='auto',
        choices=['auto', 'none', 'pinhole', 'equisolid', 'equirectangular', 'stereographic'],
        help='HOT3D conversion mode. "auto" infers from legacy HOT3D_* dataset aliases.',
    )

    parser.add_argument('--no-undistort-inp', action='store_true', help='Disable undistortion for Arm3DDataset input.')
    parser.add_argument('--no-cit', action='store_true', help='Disable CIT module in HALOAblations.')
    parser.add_argument('--no-arm-prior', action='store_true', help='Disable arm prior in HALOAblations.')
    parser.add_argument('--no-arm-input', action='store_true', help='Disable arm input in HALOAblations.')

    anycalib_group = parser.add_mutually_exclusive_group()
    anycalib_group.add_argument('--anycalib-624', action='store_true', help='Use AnyCalib 624 wrapper.')
    anycalib_group.add_argument('--anycalib-pin', action='store_true', help='Use AnyCalib pinhole wrapper.')

    parser.add_argument('--depth-model', action='store_true', help='Enable depth-model refinement.')
    parser.add_argument('--dgp-model', action='store_true', help='Enable DGP-model refinement.')

    parser.add_argument('--batch-size', type=int, default=32, help='Validation batch size.')
    parser.add_argument('--num-workers', type=int, default=8, help='DataLoader worker count.')
    parser.add_argument('--prefetch-factor', type=int, default=4, help='DataLoader prefetch factor when num_workers > 0.')
    parser.add_argument('--persistent-workers', action='store_true', help='Enable persistent DataLoader workers when num_workers > 0.')

    return parser.parse_args(argv)


def _resolve_hot3d_mode(test_dataset_name, requested_mode):
    inferred_mode = LEGACY_HOT3D_MODE_BY_DATASET.get(test_dataset_name, 'none')
    if requested_mode == 'auto':
        mode = inferred_mode
    else:
        mode = requested_mode

    if test_dataset_name not in HOT3D_DATASET_NAMES and mode != 'none':
        raise ValueError(
            '--hot3d-conversion only applies to HOT3D datasets. '
            f'Got test dataset {test_dataset_name} with conversion mode {mode}.',
        )

    return mode


def _resolve_hot3d_suffix(mode):
    if mode not in HOT3D_SUFFIX_BY_MODE:
        raise ValueError(f'Unknown HOT3D conversion mode: {mode}')

    suffix = HOT3D_SUFFIX_BY_MODE[mode]
    undistort_inp = mode == 'none'
    return suffix, undistort_inp


def _iter_settings(prefix, value):
    if isinstance(value, dict):
        for key in sorted(value.keys()):
            child_prefix = f'{prefix}.{key}' if prefix else str(key)
            yield from _iter_settings(child_prefix, value[key])
        return

    yield prefix, value


def _print_runtime_settings(args, config, resolved_settings):
    all_settings = {
        'args': vars(args),
        'resolved': resolved_settings,
        'cfg': config,
    }

    print('=== Runtime Settings ===')
    for key, value in _iter_settings('', all_settings):
        print(f'{key}: {value}')
    print('=== End Runtime Settings ===')


def get_j2d_from_j3d(cfg, meta, j3d, pred_type='hand'):
    device = j3d.device

    focal_length = meta['focal_length'].to(device)
    principal_point = meta['principal_point'].to(device)
    projection_params = meta['projection_params'].to(device)
    image_size = meta['org_img_size'].to(device)
    camera_type = meta['camera_type'].to(device)
    K_T = meta[f'K_{pred_type}'].to(device)

    if len(focal_length.shape) == 3:
        B, T = focal_length.shape[:2]
        BxT = B * T
        focal_length = focal_length.view(BxT, 2)
        principal_point = principal_point.view(BxT, 2)
        projection_params = projection_params.view(BxT, -1)
        image_size = image_size.view(BxT, 2)
        camera_type = camera_type.view(BxT)
        j3d = j3d.view(BxT, -1, 3)
        K_T = K_T.view(BxT, 3, 3)

    image_size = image_size.flip(1) # flip to get (H, W)
    mask_pinhole = (camera_type == 0)
    mask_rational = (camera_type == 2)
    mask_fisheye_624 = (camera_type == 3)
    mask_fisheye_kb3 = (camera_type == 4)
    mask_equisolid = (camera_type == 5)
    mask_equirectangular = (camera_type == 6)
    mask_stereographic = (camera_type == 7)

    idx_pinhole = mask_pinhole.nonzero(as_tuple=True)[0]
    idx_rational = mask_rational.nonzero(as_tuple=True)[0]
    idx_fisheye_624 = mask_fisheye_624.nonzero(as_tuple=True)[0]
    idx_fisheye_kb3 = mask_fisheye_kb3.nonzero(as_tuple=True)[0]
    idx_equisolid = mask_equisolid.nonzero(as_tuple=True)[0]
    idx_equirectangular = mask_equirectangular.nonzero(as_tuple=True)[0]
    idx_stereographic = mask_stereographic.nonzero(as_tuple=True)[0]

    j2d_out = torch.zeros(j3d.shape[0], j3d.shape[1], 2, device=device)
    if idx_pinhole.numel() > 0:
        focal_pinhole = focal_length[idx_pinhole]
        principal_pinhole = principal_point[idx_pinhole]

        camera = PinholeCameraPytorch3D(focal_pinhole,
                                             principal_pinhole,
                                             image_size=image_size[idx_pinhole],
                                             device=device)
        j2d_pinhole = camera.transform_points_screen_cv2(j3d[idx_pinhole])[..., :2]  
        j2d_out[idx_pinhole] = j2d_pinhole

    if idx_rational.numel() > 0:
        focal_rational = focal_length[idx_rational]
        principal_rational = principal_point[idx_rational]
        proj_params_rational = projection_params[idx_rational][:, :8]
        camera = Rational8CameraPytorch3D(focal_rational,
                                                principal_rational,
                                                proj_params_rational,
                                                image_size=image_size[idx_rational],
                                                device=device)
        j2d_rational = camera.transform_points_screen_cv2(j3d[idx_rational])[..., :2]  
        j2d_out[idx_rational] = j2d_rational

    if idx_fisheye_624.numel() > 0:
        focal_fisheye = focal_length[idx_fisheye_624]
        principal_fisheye = principal_point[idx_fisheye_624]

        proj_params_fisheye = projection_params[idx_fisheye_624][:, 3:]
        camera = FishEyeCamera624Pytorch3D(focal_fisheye,
                                                principal_fisheye,
                                                proj_params_fisheye,
                                                image_size=image_size[idx_fisheye_624],
                                                device=device)
        j2d_fisheye = camera.transform_points_screen_cv2(j3d[idx_fisheye_624])[..., :2]
        j2d_out[idx_fisheye_624] = j2d_fisheye

    if idx_fisheye_kb3.numel() > 0:
        focal_fisheye = focal_length[idx_fisheye_kb3]
        principal_fisheye = principal_point[idx_fisheye_kb3]

        proj_params_fisheye = projection_params[idx_fisheye_kb3][:, :4]
        camera = KannalaBrandtK3CameraPytorch3D(
            focal_fisheye,
            principal_fisheye,
            proj_params_fisheye,
            image_size=image_size[idx_fisheye_kb3],
            device=device,
        )
        j2d_fisheye = camera.transform_points_screen_cv2(j3d[idx_fisheye_kb3])[..., :2]
        j2d_out[idx_fisheye_kb3] = j2d_fisheye

    if idx_equisolid.numel() > 0:
        camera = EquisolidCameraPytorch3D(
            focal_length[idx_equisolid],
            principal_point[idx_equisolid],
            image_size=image_size[idx_equisolid],
            device=device,
        )
        j2d_equisolid = camera.transform_points_screen_cv2(j3d[idx_equisolid])[..., :2]
        j2d_out[idx_equisolid] = j2d_equisolid

    if idx_equirectangular.numel() > 0:
        camera = EquirectangularCameraPytorch3D(
            focal_length[idx_equirectangular],
            principal_point[idx_equirectangular],
            image_size=image_size[idx_equirectangular],
            device=device,
        )
        j2d_equirectangular = camera.transform_points_screen_cv2(j3d[idx_equirectangular])[..., :2]
        j2d_out[idx_equirectangular] = j2d_equirectangular

    if idx_stereographic.numel() > 0:
        camera = StereographicCameraPytorch3D(
            focal_length[idx_stereographic],
            principal_point[idx_stereographic],
            image_size=image_size[idx_stereographic],
            device=device,
        )
        j2d_stereographic = camera.transform_points_screen_cv2(j3d[idx_stereographic])[..., :2]
        j2d_out[idx_stereographic] = j2d_stereographic

    return j2d_out


def infer(config, model, limb_model, batch, device):
    data, meta = batch
    samplekeys = meta['samplekey'].squeeze(0).cpu().numpy()
    samplekeys = [''.join(map(chr, s)).rstrip("\x00") for s in samplekeys]

    N = data['hand_crop'].shape[0]
    
    hand_crop = data['hand_crop'].unsqueeze(0).to(device)
    hand_sparse_kpe = data['hand_sparse_kpe'].unsqueeze(0).to(device)
    gt_hand_type = data['hand_type'].unsqueeze(0).to(device)
    gt_hand_pose = data['hand_pose'].unsqueeze(0).to(device) 
    gt_global_orient = data['global_orient'].unsqueeze(0).to(device)
    gt_betas = data['betas'].unsqueeze(0).to(device)    
    gt_transl = data['transl'].unsqueeze(0).to(device)
    
    arm_crop = data['arm_crop'].unsqueeze(0).to(device)
    arm_full_kpe = data['arm_sparse_kpe'].unsqueeze(0).to(device)
    arm_T = data['arm_T'].unsqueeze(0).to(device)
    arm_R = data['arm_R'].unsqueeze(0).to(device)
    arm_shape = data['arm_shape'].unsqueeze(0).to(device)

    limb_output = get_limb(config, limb_model, gt_global_orient, gt_betas, gt_hand_pose, gt_transl, gt_hand_type, arm_shape, arm_R)
    gt_hand_j3d = limb_output.hand.joints
    gt_arm_j3d = limb_output.arm.joints
    gt_hand_vertices = limb_output.hand.vertices
    gt_arm_vertices = limb_output.arm.vertices
    gt_hand_j2d = get_j2d_from_j3d(config, meta, gt_hand_j3d)  
    gt_arm_j2d = get_j2d_from_j3d(config, meta, gt_arm_j3d, pred_type='arm')  

    with torch.no_grad():
        outputs = model(hand_crop, hand_sparse_kpe, arm_crop, arm_full_kpe)

    B, T = outputs['global_orient'].shape[0], outputs['global_orient'].shape[1]
    zT = torch.zeros(B, T, 3).to(device)
    limb_output = get_limb(config, limb_model, 
                           outputs['global_orient'], outputs['betas'], outputs['hand_pose'], zT, gt_hand_type, 
                           outputs['arm_shape'], outputs['arm_R'])
    limb_output.hand.crop_j2d = outputs['hand_kpts_2d'].squeeze(0)
    limb_output.arm.crop_j2d = outputs['arm_kpts_2d'].squeeze(0)
    limb_output.hand.confidence = outputs['hand_kpt_w'].squeeze(0)
    limb_output.arm.confidence = outputs['arm_kpt_w'].squeeze(0)
    cs_limb_output = compute_camera_space_mesh(config, meta, limb_output)
    pred_hand_vertices = cs_limb_output.hand.vertices
    pred_hand_j3d = cs_limb_output.hand.joints
    pred_arm_vertices = cs_limb_output.arm.vertices
    pred_arm_j3d = cs_limb_output.arm.joints
    pred_transl = cs_limb_output.transl

    visible_hand = data['visible_hand'].cpu().numpy()
    visible_arm = data['visible_arm'].cpu().numpy()

    pred_hand_j2d = get_j2d_from_j3d(config, meta, pred_hand_j3d)  
    pred_arm_j2d = get_j2d_from_j3d(config, meta, pred_arm_j3d, pred_type='arm')

    pred_transl = pred_transl.cpu().numpy()

    gt_hand_vertices = gt_hand_vertices.cpu().numpy() 
    gt_arm_vertices = gt_arm_vertices.cpu().numpy() 

    gt_hand_j3d = gt_hand_j3d.cpu().numpy()
    gt_arm_j3d = gt_arm_j3d.cpu().numpy()

    gt_hand_j2d = gt_hand_j2d.cpu().numpy()     
    gt_arm_j2d = gt_arm_j2d.cpu().numpy() 

    pred_hand_vertices = pred_hand_vertices.cpu().numpy() 
    pred_arm_vertices = pred_arm_vertices.cpu().numpy() 

    pred_hand_j3d = pred_hand_j3d.cpu().numpy()
    pred_arm_j3d = pred_arm_j3d.cpu().numpy()         
             
    pred_hand_j2d = pred_hand_j2d.cpu().numpy()
    pred_arm_j2d = pred_arm_j2d.cpu().numpy()  

    return {
        'samplekeys': samplekeys,
        'visible_hand': visible_hand,
        'visible_arm': visible_arm,
        'pred_transl': pred_transl,
        'gt_hand_vertices': gt_hand_vertices,
        'gt_arm_vertices': gt_arm_vertices,
        'gt_hand_j3d': gt_hand_j3d,
        'gt_arm_j3d': gt_arm_j3d,
        'gt_hand_j2d': gt_hand_j2d,
        'gt_arm_j2d': gt_arm_j2d,
        'pred_hand_vertices': pred_hand_vertices,
        'pred_arm_vertices': pred_arm_vertices,
        'pred_hand_j3d': pred_hand_j3d,
        'pred_arm_j3d': pred_arm_j3d,
        'pred_hand_j2d': pred_hand_j2d,
        'pred_arm_j2d': pred_arm_j2d,
    }


def predict_two_hands(config, dataset_name, left_loader, right_loader, model, limb_model, device, suffix='', depth_model=None, DGP_model=None):
    _DATA_DIR = os.path.join(ROOT_DIR, '_DATA')

    save_path = f'{_DATA_DIR}/predictions/{dataset_name}_{suffix}_predictions.pkl'
    if os.path.exists(save_path):
        print(f'Predictions already exist at {save_path}, skipping...')
        return

        
    model.eval()
    
    left_iterator = iter(left_loader)
    right_iterator = iter(right_loader)
    
    n_batches = len(left_iterator)

    output_data = collections.defaultdict(dict)
    current_index = 0
    batch_index = 0
    progress_bar = tqdm(total=n_batches, desc='Validation', unit='frame')
    while True:
        with torch.no_grad():
            try:
                left_batch = next(left_iterator)
                right_batch = next(right_iterator)

                batch_index += 1
            except StopIteration: break

            left_outs = infer(config, model, limb_model, left_batch, device)
            right_outs = infer(config, model, limb_model, right_batch, device)
            
            if depth_model is not None:
                left_outs, right_outs = depth_model(left_batch, right_batch, left_outs, right_outs)

            if DGP_model is not None:
                left_outs, right_outs = DGP_model(left_batch, right_batch, left_outs, right_outs)

            N = len(left_outs['samplekeys'])

            for idx in range(N):
                for htype, out in [['left', left_outs], ['right', right_outs]]:
                    samplekey = out['samplekeys'][idx]

                    output_data[samplekey][htype] = {
                        'pred_transl': out['pred_transl'][idx],

                        'hand': {
                                'visible': out['visible_hand'][idx],

                                'gt_vertices': out['gt_hand_vertices'][idx],
                                'gt_j3d': out['gt_hand_j3d'][idx],
                                'gt_j2d': out['gt_hand_j2d'][idx],

                                'pred_vertices': out['pred_hand_vertices'][idx],
                                'pred_j3d': out['pred_hand_j3d'][idx],
                                'pred_j2d': out['pred_hand_j2d'][idx],
                        },
                        'arm': {
                                'visible': out['visible_arm'][idx],

                                'gt_vertices': out['gt_arm_vertices'][idx],
                                'gt_j3d': out['gt_arm_j3d'][idx],
                                'gt_j2d': out['gt_arm_j2d'][idx],

                                'pred_vertices': out['pred_arm_vertices'][idx],
                                'pred_j3d': out['pred_arm_j3d'][idx],
                                'pred_j2d': out['pred_arm_j2d'][idx],
                        }
                    }
            
            current_index += N
            progress_bar.update(1)

    progress_bar.close()

    print(f'Saving predictions to {save_path}')
    with open(save_path, 'wb') as f:
        pickle.dump(output_data, f)


def main(argv=None):
    args = _parse_args(argv)
    test_dataset_name = args.test_dataset_name

    hot3d_mode = _resolve_hot3d_mode(test_dataset_name, args.hot3d_conversion)
    suffix, undistort_inp = _resolve_hot3d_suffix(hot3d_mode)

    if args.no_undistort_inp:
        suffix = 'undistort_inp_false'
        undistort_inp = False

    no_cit = args.no_cit
    if no_cit:
        suffix += '_no_cit'

    no_arm_prior = args.no_arm_prior
    if no_arm_prior:
        suffix += '_no_arm_prior'

    no_arm_input = args.no_arm_input
    if no_arm_input:
        suffix += '_no_arm_input'

    anycalib_624 = args.anycalib_624
    if anycalib_624:
        suffix += '_anycalib_624'

    anycalib_pin = args.anycalib_pin
    if anycalib_pin:
        suffix += '_anycalib_pin'

    depth_model = None
    if args.depth_model:
        suffix += '_depth_model'
        depth_model = DepthModelWrapper()

    DGP_model = None
    if args.dgp_model:
        suffix += '_DGP_model'
        DGP_model = DGPModelWrapper()


    if not args.checkpoint_path:
        raise ValueError('Missing checkpoint path. Pass --checkpoint-path or set cfg.POSE_3D.CHECKPOINT_PATH in settings.')

    cfg.POSE_3D.CHECKPOINT_PATH = args.checkpoint_path
    cfg.DATASET.TEST_NAME = test_dataset_name
    cfg.DATASET.NAME = test_dataset_name

    if args.batch_size <= 0:
        raise ValueError(f'Invalid --batch-size {args.batch_size}. Must be > 0.')
    if args.num_workers < 0:
        raise ValueError(f'Invalid --num-workers {args.num_workers}. Must be >= 0.')
    if args.num_workers > 0 and args.prefetch_factor <= 0:
        raise ValueError(f'Invalid --prefetch-factor {args.prefetch_factor}. Must be > 0 when workers are enabled.')

    batch_size = args.batch_size
    n_workers = args.num_workers
    prefetch_factor = args.prefetch_factor if n_workers > 0 else None
    persistent_workers = args.persistent_workers and n_workers > 0

    resolved_settings = {
        'device': str(device),
        'hot3d_mode': hot3d_mode,
        'suffix': suffix,
        'undistort_inp': undistort_inp,
        'no_cit': no_cit,
        'no_arm_prior': no_arm_prior,
        'no_arm_input': no_arm_input,
        'anycalib_624': anycalib_624,
        'anycalib_pin': anycalib_pin,
        'depth_model_enabled': depth_model is not None,
        'dgp_model_enabled': DGP_model is not None,
        'batch_size': batch_size,
        'num_workers': n_workers,
        'prefetch_factor': prefetch_factor,
        'persistent_workers': persistent_workers,
    }
    _print_runtime_settings(args, cfg, resolved_settings)

    model = HALOAblations(cfg, use_cit=not no_cit, use_arm_prior=not no_arm_prior, use_arm_input=not no_arm_input)

    print('Loading ', cfg.POSE_3D.CHECKPOINT_PATH)
    model.load_state_dict(torch.load(cfg.POSE_3D.CHECKPOINT_PATH, map_location=device), strict=True)
    model.eval()
    model.cuda()

    if test_dataset_name == 'ARCTIC':
        dataset = ArcticLoader(cfg.DATASET.ARCTIC_ROOT, get_camera=True, split='val', config=cfg, cam=0); dataset_name = 'ARCTIC'
    elif test_dataset_name == 'H2O':
        dataset = H2OLoader(cfg.DATASET.H2O_ROOT, get_camera=True, split='test', config=cfg, cam=4); dataset_name = 'H2O'
    elif test_dataset_name == 'HO3D':
        dataset = HO3DV2Loader(cfg.DATASET.HO3D_ROOT, get_camera=True, split='val', config=cfg, cam=0); dataset_name = 'HO3D'
    elif test_dataset_name in HOT3D_DATASET_NAMES:
        dataset = HOT3DLoader(
            cfg.DATASET.HOT3D_ROOT,
            get_camera=True,
            split='val',
            config=cfg,
            conversion_mode=hot3d_mode,
        )
        dataset_name = 'HOT3D'
    else:
        raise ValueError(f'Unknown dataset {test_dataset_name}')    

    if anycalib_624:
        dataset = AnyCalibDataset624(dataset)
    elif anycalib_pin:
        dataset = AnyCalibDatasetPin(dataset)

    left_loader = torch.utils.data.DataLoader(
        Arm3DDataset(cfg, dataset, 
                    undistort_inp=undistort_inp, return_complete_image=True, hand_type='left'), 
        batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True,
        prefetch_factor=prefetch_factor, 
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    right_loader = torch.utils.data.DataLoader(
        Arm3DDataset(cfg, dataset, 
                    undistort_inp=undistort_inp, return_complete_image=False, hand_type='right'), 
        batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=True,
        prefetch_factor=prefetch_factor, 
        persistent_workers=persistent_workers,
        drop_last=False,
    )
    limb_model = LimbModel(cfg, device=device, use_pose_pca=False, n_components=5)

    predict_two_hands(cfg, dataset_name, left_loader, right_loader, model, limb_model, device, suffix=suffix, depth_model=depth_model, DGP_model=DGP_model)


if __name__ == '__main__':
    main()