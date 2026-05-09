import numpy as np
import torch
import utils.math_utils as math_utils
from typing import Sequence


def world_to_eye(T_world_from_eye, v):
    """
    Apply camera inverse extrinsics to points `v` to get eye coords
    """
    return math_utils.rotate_points(
        T_world_from_eye.T, v - T_world_from_eye[:3, 3]
    )


def construct_world_to_eye_matrix(T_world_from_eye):
    """
    This can also be achieved by taking the inverse of the matrix,
    T_eye_from_world = np.linalg.inv(T_world_from_eye)
    """
    R_world_from_eye = T_world_from_eye[:3, :3]  # Rotation part
    t_world_from_eye = T_world_from_eye[:3, 3]   # Translation part

    # Inverse rotation
    R_eye_from_world = R_world_from_eye.T

    # Inverse translation
    # Negation is done to invert the translation direction
    t_eye_from_world = -R_eye_from_world @ t_world_from_eye

    # Construct the 4x4 transformation matrix
    if isinstance(T_world_from_eye, torch.Tensor):
        T_eye_from_world = torch.eye(4, device=T_world_from_eye.device, dtype=T_world_from_eye.dtype)
    else:
        T_eye_from_world = np.eye(4)

    T_eye_from_world[:3, :3] = R_eye_from_world
    T_eye_from_world[:3, 3] = t_eye_from_world

    return T_eye_from_world


def transform_points(matrix, points):
    """
    Transforms 3D points using a 4x4 transformation matrix.

    Args:
        matrix (np.ndarray): 4x4 transformation matrix.
        points (np.ndarray): Nx3 array of 3D points.
    """
    if isinstance(points, torch.Tensor):
        points_h = torch.cat([points, torch.ones((points.shape[0], 1), device=points.device)], dim=1)
        points_transformed = torch.matmul(matrix, points_h.T).T[:, :3]
    else:
        points_h = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)
        points_transformed = (matrix @ points_h.T).T[:, :3]

    return points_transformed


