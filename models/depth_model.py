import sys
import torch
import numpy as np
import cv2
from typing import Dict
from camera_models import *


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def init_depth_model():
    sys.path.extend(
    ['/netscratch/millerdurai/Repos/Depth-Anything-V2/metric_depth/']
    )
    from depth_anything_v2.dpt import DepthAnythingV2
    from torchvision.transforms import Compose
    from depth_anything_v2.dpt import Resize, NormalizeImage, PrepareForNet

    DEPTH_ANYTHING_V2_DIR = '/netscratch/millerdurai/models/DepthAnythingV2/'
    model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    }

    encoder = 'vitl' # or 'vits', 'vitb'
    dataset = 'hypersim' # 'hypersim' for indoor model, 'vkitti' for outdoor model
    max_depth = 20 # 20 for indoor model, 80 for outdoor model


    depth_model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    depth_model.load_state_dict(torch.load(f'{DEPTH_ANYTHING_V2_DIR}/checkpoints/depth_anything_v2_metric_{dataset}_{encoder}.pth', map_location='cpu'))
    depth_model = depth_model.to(device)
    depth_model.eval()

    def image2tensor_depth(input_size=518):        
        transform = Compose([
            Resize(
                width=input_size,
                height=input_size,
                resize_target=False,
                keep_aspect_ratio=True,
                ensure_multiple_of=14,
                resize_method='lower_bound',
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            PrepareForNet(),
        ])

        return transform    

    return depth_model, image2tensor_depth()


def sample_depth_nearest(depth_hw, j2d_uv):
    H, W = depth_hw.shape
    u = np.rint(j2d_uv[:,0]).astype(np.int32)
    v = np.rint(j2d_uv[:,1]).astype(np.int32)
    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    out = np.full(j2d_uv.shape[0], np.nan, dtype=depth_hw.dtype)
    out[in_bounds] = depth_hw[v[in_bounds], u[in_bounds]]
    return out, in_bounds



def strategy(rr_joints: np.ndarray, rr_jz: np.ndarray, in_bounds: np.ndarray) -> int:
    """
    Pick index i so that using jz_new = rr_joints[:,2] + jz[i] best aligns with the
    original depths jz (up to a constant offset), considering only in-bounds joints.

    Since rr_jz = jz - jz[0], the optimal offset o* that minimizes
        sum_{mask} ( (rr_joints[:,2] + o) - jz )^2
    equals mean_{mask}(jz - rr_joints[:,2]) = jz[0] + mean_{mask}(rr_jz - rr_joints[:,2]).
    Constraining o to be jz[i] is equivalent to choosing i whose
        rr_jz[i] is closest to mean_{mask}(rr_jz - rr_joints[:,2]).
    """
    # safety mask
    mask = (in_bounds.astype(bool) &
            np.isfinite(rr_jz) &
            np.isfinite(rr_joints[:, 2]))
    if not np.any(mask):
        return None  # fallback to root if nothing is usable

    # differences between predicted relative depths and relative Z from joints
    diffs = rr_jz - rr_joints[:, 2]

    # robust target using median (use np.mean for pure L2)
    target = np.median(diffs[mask])

    # choose candidate i whose rr_jz[i] is closest to target + rr_joints[i,2]  ⇔  diffs[i] closest to target
    candidates = np.flatnonzero(mask)
    i_best = candidates[np.argmin(np.abs(diffs[candidates] - target))]
    return int(i_best)


def construct_camera_model(focal_length, principal_point, projection_params, width, height, camera_type):
    focal_length = focal_length.cpu().numpy()
    principal_point = principal_point.cpu().numpy()
    projection_params = projection_params.cpu().numpy()
    width = int(width)
    height = int(height)
    
    if camera_type.item() == 0:
        camera_model = PinholeCameraModel(focal_length, principal_point, width, height)
    elif camera_type.item() == 2:
        camera_model = Rational8CameraModel(focal_length, principal_point, projection_params[:8], width, height)
    elif camera_type.item() == 3: 
        camera_model = OVR624CameraModel(focal_length, principal_point, projection_params[3:], width, height)
    elif camera_type.item() == 4: 
        camera_model = KannalaBrandtK3CameraModel(focal_length, principal_point, projection_params[:4], width, height)
    else:
        assert False, f"Camera type {camera_type} not supported"
    
    return camera_model


class DepthModelWrapper:
    def __init__(self):
        self.depth_model, self.depth_proc = init_depth_model()

    def __call__(self, left_batch, right_batch, left_outs, right_outs):
        data, meta = left_batch

        images = meta['image'].cpu().numpy()
        
        B = images.shape[0]
        
        depth_pres = []
        for i in range(B):
            depth_pre = self.depth_proc({'image': images[i].astype(np.float32) / 255.0})['image']
            depth_pre = torch.from_numpy(depth_pre)
            depth_pres.append(depth_pre)    
        
        depth_pre_batch = torch.stack(depth_pres).to(device)

        with torch.no_grad():   
            depths = self.depth_model.forward(depth_pre_batch)


        height, width = images.shape[1:3]


        for outs, hand_type in [(left_outs, 'left'), (right_outs, 'right')]:
            for limb in ['hand', 'arm']:
                if hand_type == 'left':
                    j2d_batch = left_outs[f'pred_{limb}_j2d']
                    weakp_cam_joints = left_outs[f'pred_{limb}_j3d']
                    weakp_cam_vertices = left_outs[f'pred_{limb}_vertices']
                else:
                    j2d_batch = right_outs[f'pred_{limb}_j2d']
                    weakp_cam_joints = right_outs[f'pred_{limb}_j3d']
                    weakp_cam_vertices = right_outs[f'pred_{limb}_vertices']


                for n in range(B):
                    camera_model = construct_camera_model(meta['focal_length'][n],
                                                        meta['principal_point'][n],
                                                        meta['projection_params'][n],
                                                        width,
                                                        height,
                                                        meta['camera_type'][n])

                    depth = depths[n].cpu().numpy()
                    depth = cv2.resize(depth, (width, height), interpolation=cv2.INTER_LINEAR)

                    j2d = j2d_batch[n]
                    jz, in_bounds = sample_depth_nearest(depth, j2d)

                    rr_joints = weakp_cam_joints[n] - weakp_cam_joints[n][:1]
                    rr_jz = jz[:] - jz[:1]

                    index = strategy(rr_joints, rr_jz, in_bounds)
                    if index is None:
                        jz = weakp_cam_joints[n][:, 2]
                    else:
                        jz = rr_joints[:, 2] + jz[index]

                    uvz = np.concatenate([j2d, jz[:,None]], axis=1)
                    # camera_j3d = camera_model.uvz_to_camera(uvz)  

                    camera_hand_root = camera_model.uvz_to_camera(uvz)[:1]
                    camera_j3d = rr_joints + camera_hand_root 

                    camera_v3d = (weakp_cam_vertices[n] - weakp_cam_joints[n][:1]) + camera_j3d[:1]
                    uv_exact = camera_model.camera_to_uv(camera_j3d)

                    if hand_type == 'left':
                        left_outs[f'pred_{limb}_j3d'][n] = camera_j3d
                        left_outs[f'pred_{limb}_vertices'][n] = camera_v3d
                        left_outs[f'pred_{limb}_j2d'][n] = uv_exact
                    else:
                        right_outs[f'pred_{limb}_j3d'][n] = camera_j3d
                        right_outs[f'pred_{limb}_vertices'][n] = camera_v3d
                        right_outs[f'pred_{limb}_j2d'][n] = uv_exact


        return left_outs, right_outs