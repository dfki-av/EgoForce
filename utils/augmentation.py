import numpy as np
import cv2
import random
import math
import torch
import copy
# from mmcv.transforms.utils import avoid_cache_randomness, cache_randomness
from typing import List, Optional, Sequence, Tuple, Union

from utils.rotations import rotation_6d_to_matrix_np, matrix_to_rotation_6d_np


def get_aug_config():
    scale_factor = 0.25
    rot_factor = 30
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    
    hue_delta = 5 
    saturation_delta = 30
    value_delta = 30
    hsv_gains = np.random.uniform(-1, 1, 3) * [hue_delta, saturation_delta, value_delta]
    hsv_gains *= np.random.randint(0, 2, 3)
    hsv_gains = hsv_gains.astype(np.int16)

    do_flip = False

    return scale, rot, hsv_gains, do_flip


def generate_patch_image(cvimg, scale, rot, do_flip, out_shape):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
   
    img_cx, img_cy = img_width / 2.0, img_height / 2.0
    
    if do_flip:
        img = img[:, ::-1, :]
        img_cx = img_width - img_cx - 1

    trans = gen_trans_from_patch_cv(img_cx, img_cy, img_width, img_height, out_shape[1], out_shape[0], scale, rot)
    img_patch = cv2.warpAffine(img, trans, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)
    inv_trans = gen_trans_from_patch_cv(img_cx, img_cy, img_width, img_height, out_shape[1], out_shape[0], scale, rot, inv=True)

    return img_patch, trans, inv_trans


def generate_patch_mask(cvimg, T, do_flip, out_shape):
    img = cvimg.copy()
       
    if do_flip:
        img = img[:, ::-1, :]

    img_patch = cv2.warpAffine(img, T, (int(out_shape[1]), int(out_shape[0])), flags=cv2.INTER_LINEAR)

    return img_patch


def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)

    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir
    
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    trans = trans.astype(np.float32)
    return trans


def augmentation(img, data_split, out_size, enforce_flip=None):
    if data_split == 'train':
        scale, rot, color_scale, do_flip = get_aug_config()
    else:
        scale, rot, color_scale, do_flip = 1.0, 0.0, np.array([1,1,1]), False
    
    if enforce_flip is None:
        pass
    elif enforce_flip is True:
        do_flip = True
    elif enforce_flip is False:
        do_flip = False
    
    img, trans, inv_trans = generate_patch_image(img, scale, rot, do_flip, out_size)
    img = img.astype(np.float32)
    img = np.clip(img * color_scale[None,None,:], 0, 255)
    return img, trans, inv_trans, rot, scale, do_flip


def color_augmentation(img, hsv_gains):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)

    img_hsv[..., 0] = (img_hsv[..., 0] + hsv_gains[0]) % 180
    img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_gains[1], 0, 255)
    img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_gains[2], 0, 255)
    cv2.cvtColor(img_hsv.astype(img.dtype), cv2.COLOR_HSV2BGR, dst=img)


    return img.astype(np.uint8)


def rotate_point(pt, center, angle_rad):
    """Rotate a point counterclockwise by a given angle around a given origin."""
    ox, oy = center
    px, py = pt

    qx = ox + np.cos(angle_rad) * (px - ox) - np.sin(angle_rad) * (py - oy)
    qy = oy + np.sin(angle_rad) * (px - ox) + np.cos(angle_rad) * (py - oy)
    return [qx, qy]


# def bbox_augmentation(bbox, image_size, scale, rot):
#     """
#     bbox: [x1, y1, x2, y2] format
#     image_size: (W, H)
#     scale: float scaling factor
#     rot: float rotation angle in degrees
#     """
#     x1, y1, x2, y2 = bbox
#     w = x2 - x1
#     h = y2 - y1
#     x = x1
#     y = y1

#     cx = x + w / 2.0
#     cy = y + h / 2.0

#     # Scale the bounding box
#     new_w = w * scale
#     new_h = h * scale

#     # Create the scaled bbox centered at the same point
#     new_bbox = [
#         cx - new_w / 2.0,
#         cy - new_h / 2.0,
#         new_w,
#         new_h
#     ]

#     # Get the four corners of the new bbox
#     corners = np.array([
#         [new_bbox[0], new_bbox[1]],  # top-left
#         [new_bbox[0] + new_w, new_bbox[1]],  # top-right
#         [new_bbox[0] + new_w, new_bbox[1] + new_h],  # bottom-right
#         [new_bbox[0], new_bbox[1] + new_h]  # bottom-left
#     ])

#     # Rotate the corners around the bbox center
#     angle_rad = np.deg2rad(rot)
#     rotated_corners = np.array([rotate_point(pt, (cx, cy), angle_rad) for pt in corners])

#     # Get new axis-aligned bounding box from rotated corners
#     x_min = np.min(rotated_corners[:, 0])
#     y_min = np.min(rotated_corners[:, 1])
#     x_max = np.max(rotated_corners[:, 0])
#     y_max = np.max(rotated_corners[:, 1])

#     # Clip to image boundaries
#     x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

#     x1, y1 = x_min, y_min
#     x2, y2 = x_max - x_min, y_max - y_min

#     x_min = max(0, x_min)
#     y_min = max(0, y_min)
#     x_max = min(image_size[0], x_max)
#     y_max = min(image_size[1], y_max)

#     final_bbox = [
#         x_min, y_min, x_max, y_max
#     ]

#     return final_bbox


def bbox_augmentation(bbox, image_size, scale, rot=0, max_shift=0.1):
    """
    bbox:    [x1, y1, x2, y2]
    image_size: (W, H)
    scale:   float scaling factor
    max_shift: maximum random shift as fraction of box dim (e.g. 0.1 → ±10%)
    
    Returns:
        [x1', y1', x2', y2'] clipped to image.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # center of the original box
    cx = x1 + 0.5 * w
    cy = y1 + 0.5 * h

    # scale box
    new_w = w * scale
    new_h = h * scale

    # random shift in box-centered coords
    shift_x = np.random.uniform(-max_shift, max_shift) * new_w
    shift_y = np.random.uniform(-max_shift, max_shift) * new_h

    # compute new top‑left
    x1n = cx - new_w/2 + shift_x
    y1n = cy - new_h/2 + shift_y
    x2n = x1n + new_w
    y2n = y1n + new_h

    # clip to image boundaries
    W, H = image_size
    x1n = np.clip(x1n, 0, W-1)
    y1n = np.clip(y1n, 0, H-1)
    x2n = np.clip(x2n, x1n+1, W)  # ensure positive width
    y2n = np.clip(y2n, y1n+1, H)  # ensure positive height

    return [int(x1n), int(y1n), int(x2n), int(y2n)]


def transform_kpt2d(kpt2d, T, img_size, do_flip):
    """
    Transforms 2D points using the affine transformation matrix T.
    
    Parameters:
      kpt2d: numpy array of shape (N, 2), where each row is a 2D point (x, y)
      T: affine transformation matrix of shape (2, 3)
      
    Returns:
      kpt2d_transformed: numpy array of shape (N, 2) with the transformed points
    """
    # Convert kpt2d to homogeneous coordinates (N, 3)
    kpt2d = kpt2d.copy()

    if do_flip:
        w = img_size[0]
        kpt2d[:, 0] = w - kpt2d[:, 0] - 1

    ones = np.ones((kpt2d.shape[0], 1), dtype=kpt2d.dtype)
    kpt2d_hom = np.concatenate([kpt2d, ones], axis=1)  # shape: (N, 3)
    
    # Apply the transformation: (N, 3) x (3, 2) -> (N, 2)
    kpt2d_transformed = np.dot(kpt2d_hom, T.T)

    return kpt2d_transformed


def transform_kpt3d(kpt3d, rot_deg, scale, do_flip):
    # Copy to avoid modifying the original keypoints.
    kpt3d_transformed = kpt3d.copy()
    
    # Handle horizontal flip: adjust the x coordinate in the image plane.
    if do_flip:
        kpt3d_transformed[..., 0] *= -1

    theta = np.deg2rad(-rot_deg)
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32) # Rotation around z-axis as we only augment in the (x,y) 


    print('rot_deg ', rot_deg, scale, do_flip)

    kpt3d_transformed = (rot_mat @ kpt3d_transformed.T).T

    # scale_mat = np.diag([scale, scale, 1.0])
    # kpt3d_transformed = kpt3d_transformed @ scale_mat

    return kpt3d_transformed



def transform_root_pose_6d(root_pose_6d, rot_deg):
    # Copy to avoid modifying the original keypoints.
    root_pose = rotation_6d_to_matrix_np(root_pose_6d)
    
    theta = np.deg2rad(-rot_deg)
    rot_mat = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1]
    ], dtype=np.float32) # Rotation around z-axis as we only augment in the (x,y) 

    root_pose_transformed = np.dot(rot_mat, root_pose)
    root_pose_transformed_6d = matrix_to_rotation_6d_np(root_pose_transformed)

    return root_pose_transformed_6d


def transform_intrinsics(K, scale, rot_deg, do_flip, image_shape):
    """
    Transforms the intrinsic matrix K according to image augmentations.
    
    Args:
        K (np.ndarray): Original intrinsic matrix (3x3).
        scale (float): Scaling factor.
        rot_deg (float): Rotation in degrees (counter-clockwise).
        do_flip (bool): Whether horizontal flip was applied.
        image_shape (tuple): Original image shape (H, W).
        
    Returns:
        np.ndarray: Transformed intrinsic matrix (3x3).
    """
    H, W = image_shape
    K_aug = K.copy()

    # --- Scaling ---
    K_aug[0, 0] *= scale  # f_x
    K_aug[1, 1] *= scale  # f_y
    K_aug[0, 2] *= scale  # c_x
    K_aug[1, 2] *= scale  # c_y

    # --- Rotation around image center ---
    if rot_deg != 0:
        theta = np.deg2rad(-rot_deg)
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta),  np.cos(theta)]
        ])
        center = K_aug[:2, 2].copy()  # Original principal point (c_x, c_y) 
        principal_point = K_aug[:2, 2]

        principal_point = R @ (principal_point - center) + center
        K_aug[0, 2], K_aug[1, 2] = principal_point

    # # --- Horizontal Flip ---
    # if do_flip:
    #     K_aug[0, 2] = W - K_aug[0, 2] - 1

    return K_aug


def crop_intrinsics(K, crop_x, crop_y):
    """
    Adjust the intrinsic matrix K after cropping the image.
    
    Parameters:
      K (np.ndarray): Original intrinsic matrix (3x3).
      crop_x (int or float): The x-coordinate (column) of the crop's top-left corner.
      crop_y (int or float): The y-coordinate (row) of the crop's top-left corner.
      
    Returns:
      np.ndarray: New intrinsic matrix adjusted for the crop.
    """
    K_new = K.copy()
    # Adjust the principal point coordinates
    K_new[0, 2] = K[0, 2] - crop_x  # New c_x
    K_new[1, 2] = K[1, 2] - crop_y  # New c_y
    return K_new


def undistort_local_patch(
    distorted_img: np.ndarray,
    roi,                    # (x1, y1, x2, y2) in the *full* distorted image
    K_orig: np.ndarray,     # 3x3 intrinsics of the full distorted image
    distortion_model,       # OVR624Distortion instance (evaluate expects undistorted normalized coords)
    K_new: np.ndarray=None, # 3x3 intrinsics for the *destination pinhole*; if None, use sub-intrinsics
    R: np.ndarray=None,     # 3x3 rotation (SO(3)); if None, identity
    out_size=None,          # (out_w, out_h); if None, ROI size
    interpolation: str="linear",  # "linear" or "nearest"
    border_value: float=0.0
):
    """
    Inverse warp (dst→src) of a *local ROI* from a fisheye OVR624 image to a pinhole view.

    Steps:
      For each destination pixel u_d:
        1) Unproject with pinhole K_new  -> ray r_d
        2) Rotate r_s = R * r_d (if R provided)
        3) Project into source fisheye:
             p = (r_s.x / r_s.z, r_s.y / r_s.z)
             q = distortion_model.evaluate(p)         # forward fisheye distortion
             [u_s, v_s] = K_sub * q                   # pixel coords in the *sub-image*
        4) Sample distorted sub-image at (u_s, v_s)

    Returns:
      undistorted_img: np.ndarray with shape (out_h, out_w[, C])
      K_new:           the destination intrinsics actually used
    """
    x1, y1, x2, y2 = map(int, roi)
    sub_img = distorted_img[y1:y2, x1:x2]
    h_sub, w_sub = sub_img.shape[:2]

    # Adjust intrinsics into the sub-image coordinates
    K_sub = K_orig.astype(np.float32).copy()
    K_sub[0, 2] -= x1
    K_sub[1, 2] -= y1

    # Destination intrinsics and size
    if K_new is None:
        K_new = K_sub.copy()
    K_new = K_new.astype(np.float32)

    if out_size is None:
        out_w, out_h = w_sub, h_sub
    else:
        out_w, out_h = map(int, out_size)

    if distortion_model.is_distorted is False:
        return sub_img, K_new

    # Build destination pixel grid (OpenCV convention: meshgrid gives (V,Y)-(U,X) as (rows, cols))
    u = np.arange(out_w, dtype=np.float32)
    v = np.arange(out_h, dtype=np.float32)
    U, V = np.meshgrid(u, v)  # (out_h, out_w)

    # 1) Unproject through pinhole K_new to unit rays
    fx, fy = K_new[0, 0], K_new[1, 1]
    cx, cy = K_new[0, 2], K_new[1, 2]
    qx = (U - cx) / fx
    qy = (V - cy) / fy
    # Rays on z=1 plane, then normalize
    rays = np.stack([qx, qy, np.ones_like(qx)], axis=-1)           # (out_h, out_w, 3)
    nrm  = np.linalg.norm(rays, axis=-1, keepdims=True).clip(min=np.finfo(np.float32).eps)
    rays = rays / nrm

    # 2) Optional 3D rotation
    if R is None:
        R = np.eye(3, dtype=np.float32)
    else:
        R = np.asarray(R, dtype=np.float32)
    rays_src = rays @ R.T                                           # (out_h, out_w, 3)

    # 3) Project into the *source fisheye* (arctan plane -> distort -> pixels)
    eps = np.finfo(np.float32).eps
    Z = np.where(rays_src[..., 2:3] >= 0, np.maximum(rays_src[..., 2:3], eps),
                                   np.minimum(rays_src[..., 2:3], -eps))
    p_undist = np.concatenate([rays_src[..., 0:1] / Z,             # x/z
                               rays_src[..., 1:2] / Z], axis=-1)   # y/z

    # Apply forward fisheye distortion (OVR624)
    q_dist = distortion_model.evaluate(p_undist)                    # (out_h, out_w, 2)

    # Map to *sub-image* pixels with K_sub
    map_x = q_dist[..., 0] * K_sub[0, 0] + K_sub[0, 2]             # (out_h, out_w)
    map_y = q_dist[..., 1] * K_sub[1, 1] + K_sub[1, 2]

    # 4) Resample from sub-image using cv2.remap
    inter = cv2.INTER_LINEAR if interpolation == "linear" else cv2.INTER_NEAREST
    undistorted_img = cv2.remap(
        sub_img,
        map_x.astype(np.float32),
        map_y.astype(np.float32),
        interpolation=inter,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )
    return undistorted_img, K_new


def undistort_local_patch1(
    distorted_img,
    roi,              # (x1, y1, x2, y2) bounding box in the original image
    K_orig,           # 3x3 numpy intrinsic matrix for the full image
    distortion_model,           # 12-element distortion parameter vector for the fisheye624 model
    K_new=None,       # Desired new camera matrix for the undistorted region (if None, we use the adjusted subintrinsics)
    R=np.eye(3),      # Rotation matrix to reorient the ray directions, default identity
    out_size=None     # Desired output size tuple (width, height) of the undistorted image.
    
):
    """
    Undistort a locally cropped fisheye region.
    
    This function works by:
      1. Extracting the sub-image defined by ROI.
      2. Adjusting the original intrinsic matrix to the sub-image coordinate system.
      3. Creating a grid for the output (undistorted) image.
      4. Mapping each undistorted pixel (using new intrinsics, and optional rotation) to a ray in normalized (ideal)
         space.
      5. Applying the fisheye624 distortion via the forward model (OVR624Distortion.evaluate) to compute the
         corresponding distorted normalized coordinate.
      6. Reconstructing the pixel coordinates in the sub-image using the adjusted intrinsic matrix.
      7. Remapping the original (distorted) sub-image pixels into the undistorted frame using interpolation.
         
    Parameters:
      distorted_img : np.ndarray
          The original distorted fisheye image.
      roi : tuple of ints
          (x1, y1, x2, y2) defining the bounding box for the region of interest.
      K_orig : np.ndarray
          The original 3x3 camera intrinsic matrix.
      D_orig : array-like
          The 12-dimensional distortion parameters (fisheye624).
      K_new : np.ndarray, optional
          The new intrinsic matrix for the output view. If None, the adjusted intrinsics for the ROI will be used.
      R : np.ndarray, optional
          A 3x3 rotation matrix applied to the undistorted rays (default is the identity).
      out_size : tuple of ints, optional
          The desired output size (width, height). If None, the output size will match the ROI size.
    
    Returns:
      undistorted_img : np.ndarray
          The undistorted sub-image.
    """
    # Unpack the ROI
    x1, y1, x2, y2 = roi
    sub_img = distorted_img[y1:y2, x1:x2]
    h_sub, w_sub = sub_img.shape[:2]
    
    # Adjust the principal point of the original intrinsic matrix to the sub-image coordinate system.
    K_sub = K_orig.copy()
    K_sub[0, 2] -= x1
    K_sub[1, 2] -= y1
    
    # If a new intrinsic matrix is not provided, keep the sub-image intrinsics.
    if K_new is None:
        K_new = K_sub.copy()
    
    # Determine output size.
    if out_size is None:
        out_w, out_h = w_sub, h_sub
    else:
        out_w, out_h = out_size

    if distortion_model.is_distorted is False:
        return sub_img, K_new

    # Create a meshgrid for the output (undistorted) image in pixel coordinates.
    # Note: Using the OpenCV convention, the grid has shape (height, width).
    u = np.arange(out_w, dtype=np.float32)
    v = np.arange(out_h, dtype=np.float32)
    U, V = np.meshgrid(u, v)  # U: x-coord, V: y-coord, both shaped (out_h, out_w)
    
    # Convert the pixel coordinates from the new intrinsics into normalized coordinates.
    # For each output pixel, we compute (x, y) so that:
    #   x = (u - cx_new) / fx_new  and  y = (v - cy_new) / fy_new
    x_norm = (U - K_new[0, 2]) / K_new[0, 0]
    y_norm = (V - K_new[1, 2]) / K_new[1, 1]
    pts_norm = np.stack((x_norm, y_norm), axis=-1)  # Shape: (out_h, out_w, 2)
    
    # Optionally apply the rotation R. To do so, first convert to homogeneous coordinates.
    # ones = np.ones((out_h, out_w, 1), dtype=np.float32)
    # pts_hom = np.concatenate([pts_norm, ones], axis=-1)  # (out_h, out_w, 3)
    # # Apply the rotation. Note: For a rotation around the optical axis, the third coordinate remains 1.
    # pts_rot = pts_hom @ R.T  # (out_h, out_w, 3)
    # pts_rot = pts_rot[..., :2]  # Use the first two coordinates as the rotated normalized coordinates.
    
    # Create an instance of the distortion model for this sub-image.
    # The distortion model expects the image dimensions (width, height) of the image on which it operates.
    # Apply the forward distortion model: this maps from the undistorted normalized coords to the distorted ones.
    pts_distorted_norm = distortion_model.evaluate(pts_norm)  # (out_h, out_w, 2)
        
    # Map the distorted normalized coordinates back to pixel coordinates in the sub-image
    # using the adjusted intrinsics K_sub.
    map_x = pts_distorted_norm[..., 0] * K_sub[0, 0] + K_sub[0, 2]
    map_y = pts_distorted_norm[..., 1] * K_sub[1, 1] + K_sub[1, 2]
    
    # Remap the pixels from the distorted sub-image into the undistorted output image.
    # cv2.remap requires the mapping arrays (map_x and map_y) to be of type float32.
    undistorted_img = cv2.remap(sub_img, map_x.astype(np.float32), map_y.astype(np.float32),
                                interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    
    return undistorted_img, K_new


def undistort_keypoints(keypoints, K_new, distortion_model):
    """
    Remap keypoints from the distorted sub-image coordinate system to the undistorted view.
    
    keypoints: Nx2 array of keypoint coordinates computed using the linear K_new projection.
    K_new: The new intrinsics used for the output image.
    K_sub: The adjusted sub-image intrinsic matrix (K_orig adjusted for the ROI).
    distortion_model: An instance of OVR624Distortion initialized for the sub-image dimensions.
    
    Returns an array of keypoints aligned with the undistorted image.
    """
    # Convert keypoints to normalized coordinates (as used by K_new)
    keypoints_norm = np.empty_like(keypoints, dtype=np.float32)
    keypoints_norm[:, 0] = (keypoints[:, 0] - K_new[0, 2]) / K_new[0, 0]
    keypoints_norm[:, 1] = (keypoints[:, 1] - K_new[1, 2]) / K_new[1, 1]
    
    # Apply the inverse distortion mapping to get the "ideal" undistorted normalized coordinates.
    # (Note: Your distortion model is non-linear so this step is key for alignment)
    keypoints_undist_norm = distortion_model.inverse_evaluate(keypoints_norm)
    
    # Reproject the undistorted normalized coordinates with K_new (or any new desired intrinsics).
    keypoints_undist = np.empty_like(keypoints, dtype=np.float32)
    keypoints_undist[:, 0] = keypoints_undist_norm[..., 0] * K_new[0, 0] + K_new[0, 2]
    keypoints_undist[:, 1] = keypoints_undist_norm[..., 1] * K_new[1, 1] + K_new[1, 2]
    
    return keypoints_undist


class HandColorAugmentation:
    def __init__(self, prob: float = 1.0):
        self.prob = prob

    def __call__(self, img):
        # Apply augmentations with a certain probability
        if np.random.rand() < self.prob:
            img = self.apply_augmentations(img)

        return img
    
    def apply_augmentations(self, img):
        n_augmentations = np.random.randint(1, 5)
        aug_count = 0

        # Random Brightness Adjustment
        if np.random.rand() < 0.5:
            value = np.random.uniform(-50, 50)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 2] = np.clip(hsv[..., 2] + value, 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            aug_count += 1
        if aug_count >= n_augmentations: return img

        # Random Contrast Adjustment
        if np.random.rand() < 0.5:
            alpha = np.random.uniform(0.5, 1.5)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
            aug_count += 1
        if aug_count >= n_augmentations: return img

        # Random Saturation Adjustment
        if np.random.rand() < 0.5:
            saturation_scale = np.random.uniform(0.5, 1.5)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= saturation_scale
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            aug_count += 1
        if aug_count >= n_augmentations: return img

        # Random Hue Adjustment
        if np.random.rand() < 0.5:
            hue_delta = np.random.randint(-10, 10)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int32)
            hsv[..., 0] = (hsv[..., 0] + hue_delta) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            aug_count += 1
        if aug_count >= n_augmentations: return img

        # Grayscale Conversion
        if np.random.rand() < 0.5:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
            aug_count += 1
        if aug_count >= n_augmentations: return img

        # Gamma Correction
        if np.random.rand() < 0.5:
            gamma = np.random.uniform(0.5, 1.5)
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)]).astype("uint8")
            img = cv2.LUT(img, table)
            aug_count += 1
        if aug_count >= n_augmentations: return img

        # Exposure Adjustment
        if np.random.rand() < 0.5:
            exposure = np.random.uniform(0.5, 1.5)
            img = cv2.convertScaleAbs(img, alpha=exposure, beta=0)
            aug_count += 1
        if aug_count >= n_augmentations: return img

        # Blurring
        if np.random.rand() < 0.5:
            ksize = np.random.choice([3, 5, 7])
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
            aug_count += 1
        if aug_count >= n_augmentations: return img

        # Channel Swapping
        if np.random.rand() < 0.5:
            img = img[..., np.random.permutation(3)]
            aug_count += 1
        if aug_count >= n_augmentations: return img

        # Color Jitter
        if np.random.rand() < 0.5:
            brightness = np.random.uniform(0.8, 1.2)
            contrast = np.random.uniform(0.8, 1.2)
            saturation = np.random.uniform(0.8, 1.2)
            hue = np.random.uniform(-10, 10)

            # Adjust brightness and contrast
            img = cv2.convertScaleAbs(img, alpha=contrast, beta=brightness * 50 - 25)

            # Adjust saturation and hue
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[..., 1] *= saturation
            hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
            hsv[..., 0] = (hsv[..., 0] + hue) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            aug_count += 1
        if aug_count >= n_augmentations: return img

        return img


class CutMix:
    def __init__(self,
                 img_scale: Tuple[int, int] = (640, 640),
                 ratio_range: Tuple[float, float] = (0.1, 0.4),
                 max_cached_images: int = 20,
                 random_pop: bool = True,
                 prob: float = 1.0) -> None:
        assert isinstance(img_scale, tuple), "img_scale must be a tuple."
        assert max_cached_images >= 2, 'The cache size must be >= 2.'
        assert 0 <= prob <= 1.0, 'prob must be between 0 and 1.'

        self.img_scale = img_scale
        self.ratio_range = ratio_range
        self.max_cached_images = max_cached_images
        self.random_pop = random_pop
        self.prob = prob
        self.results_cache: List[dict] = []

    # @cache_randomness
    def _get_random_image(self) -> Union[np.ndarray, None]:
        """Select a random imagen from the cache that has valid annotations."""
        if not self.results_cache:
            return None

        index = random.randint(0, len(self.results_cache) - 1)
        cached_result = self.results_cache[index]
        return cached_result

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(img_scale={self.img_scale}, '
        repr_str += f'ratio_range={self.ratio_range}, '
        repr_str += f'max_cached_images={self.max_cached_images}, '
        repr_str += f'random_pop={self.random_pop}, '
        repr_str += f'prob={self.prob})'
        return repr_str

    def __call__(self, ori_img: np.ndarray, cache_image: np.ndarray) -> dict:
        self.results_cache.append(cache_image.copy()) # Make a copy to avoid modifying the original image
        if len(self.results_cache) > self.max_cached_images:
            if self.random_pop:
                index = random.randint(0, len(self.results_cache) - 1)
            else:
                index = 0
            self.results_cache.pop(index)

        if len(self.results_cache) <= 1:
            return ori_img

        if random.uniform(0, 1) > self.prob:
            return ori_img

        # Get a random image from the cache
        mix_img = self._get_random_image()
        if mix_img is None:
            return ori_img

        h, w = ori_img.shape[:2]
        mix_h, mix_w = mix_img.shape[:2]

        # Determine the size of the patch to be cut and pasted
        cut_ratio = random.uniform(*self.ratio_range)
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)

        # Ensure the patch size does not exceed the dimensions of either image
        cut_w = min(cut_w, w, mix_w)
        cut_h = min(cut_h, h, mix_h)

        if cut_w <= 0 or cut_h <= 0:
            return ori_img  # Skip if patch dimensions are invalid

        # Random position in the original image where the patch will be placed
        if w - cut_w > 0:
            x1 = random.randint(0, w - cut_w)
        else:
            x1 = 0
        if h - cut_h > 0:
            y1 = random.randint(0, h - cut_h)
        else:
            y1 = 0
        x2 = x1 + cut_w
        y2 = y1 + cut_h

        # Random position in the mix image from where the patch will be cut
        if mix_w - cut_w > 0:
            mix_x1 = random.randint(0, mix_w - cut_w)
        else:
            mix_x1 = 0
        if mix_h - cut_h > 0:
            mix_y1 = random.randint(0, mix_h - cut_h)
        else:
            mix_y1 = 0
        mix_x2 = mix_x1 + cut_w
        mix_y2 = mix_y1 + cut_h

        ori_img[y1:y2, x1:x2] = mix_img[mix_y1:mix_y2, mix_x1:mix_x2]

        return ori_img
    



_REFLECTION_MAT = np.diag(np.array([-1.0, 1.0, 1.0]))  # mirror across X

def _mirror_rot_matrix(R):
    """Apply *X* reflection S R S (supports arbitrary batch shapes)."""
    return _REFLECTION_MAT @ R @ _REFLECTION_MAT


def _mirror_axis_angle(aa: np.ndarray) -> np.ndarray:
    """Mirror an axis‑angle rotation (shape [..., 3])."""
    R = axis_angle_to_matrix_np(aa.reshape(-1, 3))  # (N,3,3)
    R_m = _mirror_rot_matrix(R)
    return matrix_to_axis_angle_np(R_m).reshape_as(aa)


def _mirror_rot_6d(rot6d: np.ndarray) -> np.ndarray:
    """Mirror an axis‑angle rotation (shape [..., 3])."""
    R = rotation_6d_to_matrix_np(rot6d.reshape(-1, 6))  # (N,3,3)
    R_m = _mirror_rot_matrix(R)
    mirrored_rot6d = matrix_to_rotation_6d_np(R_m)

    return mirrored_rot6d


def flip_mano_params(global_orient, hand_pose, transl):
    transl = transl.copy()

    pose = np.concatenate([global_orient[None, ...], hand_pose], axis=0)
    pose_mirrored = _mirror_rot_6d(pose)

    global_orient = pose_mirrored[0]
    hand_pose = pose_mirrored[1:]
    transl[0] *= -1.0
    
    return global_orient, hand_pose, transl


def flip_arm_parms(arm_R, arm_T):
    arm_T = arm_T.copy()
    arm_R = _mirror_rot_6d(arm_R)[0]

    arm_T[0] *= -1.0
    
    return arm_R, arm_T



def flip_kpt2d(keypoints, image_width):
    flipped = keypoints.copy()
    flipped[..., 0] = (image_width - 1) - keypoints[..., 0]

    return flipped


def flip_bbox(bbox, image_width):
    flipped = np.asarray(bbox, dtype=np.float32)
    flipped[[0, 2]] = (image_width - 1) - flipped[[2, 0]]

    return flipped


def flip_kpt3d(kpts3d: np.ndarray) -> np.ndarray:
    
    flipped = kpts3d.copy()
    flipped[..., 0] *= -1.0

    # root_transl = kpts3d[:1]  # Assuming the first keypoint is the root joint
    # flipped = kpts3d - root_transl
    # flipped[..., 0] *= -1.0
    # flipped = flipped + root_transl

    return flipped


def flip_image(image):
    image_flipped = cv2.flip(image, 1)
    return image_flipped


def flip_camera_intrinsics(K, image_width):
    '''
    K = [f_x, 0,   c_x,
          0,  f_y, c_y,
          0,  0,   1]
    '''
    K_flipped = K.copy()
    K_flipped[0, 2] = image_width - K[0, 2]  # Flip principal point (cx)

    return K_flipped


def flip_camera_params(camera_params, image):
    image_height, image_width = image.shape[:2]
    px, py = camera_params['principal_point']

    px_flipped = image_width - px    
    
    camera_params['principal_point'] = np.array([px_flipped, py], dtype=np.float32)

    return camera_params

