# Code based on PyTorch3D: https://github.com/facebookresearch/pytorch3d/blob/main/pytorch3d/transforms/rotation_conversions.py
# License from PyTorch3D: https://github.com/facebookresearch/pytorch3d/blob/main/LICENSE

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Union

import math
import torch
import numpy as np
import torch.nn.functional as F

Device = Union[str, torch.device]


"""
The transformation matrices returned from the functions in this file assume
the points on which the transformation will be applied are column vectors.
i.e. the R matrix is structured as

    R = [
            [Rxx, Rxy, Rxz],
            [Ryx, Ryy, Ryz],
            [Rzx, Rzy, Rzz],
        ]  # (3, 3)

This matrix can be applied to column vectors by post multiplication
by the points e.g.

    points = [[0], [1], [2]]  # (3 x 1) xyz coordinates of a point
    transformed_points = R * points

To apply the same matrix to points which are row vectors, the R matrix
can be transposed and pre multiplied by the points:

e.g.
    points = [[0, 1, 2]]  # (1 x 3) xyz coordinates of a point
    transformed_points = points * R.transpose(1, 0)
"""


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def quaternion_to_matrix_np(quat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert quaternions to rotation matrices (w, x, y, z → 3 × 3).

    Parameters
    ----------
    quat : np.ndarray
        Array of shape (..., 4) with real part first.
    eps : float, optional
        Quaternions whose ‖q‖ < eps are treated as “invalid” and map
        to an identity matrix, by default 1 e-8.

    Returns
    -------
    np.ndarray
        Rotation matrices of shape (..., 3, 3).
    """
    quat = np.asarray(quat, dtype=np.float64)
    orig_shape = quat.shape[:-1]

    # Flatten so the broadcast logic stays simple
    quat_flat = quat.reshape(-1, 4)             # (N, 4)
    norm = np.linalg.norm(quat_flat, axis=1)    # (N,)

    valid = norm >= eps

    # Start with identities everywhere
    R_flat = np.broadcast_to(np.eye(3), (quat_flat.shape[0], 3, 3)).copy()

    if np.any(valid):
        q = quat_flat[valid] / norm[valid, None]     # (M,4) normalised
        r, i, j, k = q.T
        two_s = 2.0                                  # because ‖q‖ = 1

        R_flat[valid] = np.stack(
            (
                1 - two_s * (j * j + k * k),
                two_s * (i * j - k * r),
                two_s * (i * k + j * r),
                two_s * (i * j + k * r),
                1 - two_s * (i * i + k * k),
                two_s * (j * k - i * r),
                two_s * (i * k - j * r),
                two_s * (j * k + i * r),
                1 - two_s * (i * i + j * j),
            ),
            axis=-1,
        ).reshape(-1, 3, 3)

    return R_flat.reshape(orig_shape + (3, 3))


def _copysign(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return a tensor where each element has the absolute value taken from the,
    corresponding element of a, with sign taken from the corresponding
    element of b. This is like the standard copysign floating-point operation,
    but is not careful about negative 0 and NaN.

    Args:
        a: source tensor.
        b: tensor whose signs will be used, of the same shape as a.

    Returns:
        Tensor of the same shape as a with the signs of b.
    """
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(batch_dim + (4,))


def matrix_to_quaternion_np(mat: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Convert 3 × 3 rotation matrix/matrices to quaternions (w, x, y, z).

    Parameters
    ----------
    mat : np.ndarray
        Shape (..., 3, 3). Must be proper rotation(s) (det ≈ 1).
    eps : float, optional
        Threshold used only to flag malformed inputs, by default 1e-8.

    Returns
    -------
    np.ndarray
        Quaternions of shape (..., 4) with the real part first.
    """
    if mat.shape[-2:] != (3, 3):
        raise ValueError(f"Invalid rotation matrix shape {mat.shape}")

    # Unpack matrix elements
    m00, m01, m02 = mat[..., 0, 0], mat[..., 0, 1], mat[..., 0, 2]
    m10, m11, m12 = mat[..., 1, 0], mat[..., 1, 1], mat[..., 1, 2]
    m20, m21, m22 = mat[..., 2, 0], mat[..., 2, 1], mat[..., 2, 2]

    # Positive-clamped square-roots of the four “absolute” candidates
    def _sqrt_positive_part(x):
        return np.sqrt(np.clip(x, 0.0, None))

    q_abs = _sqrt_positive_part(
        np.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            axis=-1,
        )
    )                                               # (...,4)

    # Build four candidate quaternions (each row multiplied by r,i,j,k)
    quat_by_rijk = np.stack(
        [
            np.stack([q_abs[..., 0]**2, m21 - m12, m02 - m20, m10 - m01], axis=-1),
            np.stack([m21 - m12, q_abs[..., 1]**2, m10 + m01, m02 + m20], axis=-1),
            np.stack([m02 - m20, m10 + m01, q_abs[..., 2]**2, m12 + m21], axis=-1),
            np.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3]**2], axis=-1),
        ],
        axis=-2,
    )                                               # (...,4,4)

    # Divide by 2·q_abs, flooring the denominator at 0.1 to avoid zero-div
    denom = 2.0 * np.maximum(q_abs[..., None], 0.1)
    quat_candidates = quat_by_rijk / denom          # (...,4,4)

    # Pick the best-conditioned candidate (largest q_abs component)
    idx = np.argmax(q_abs, axis=-1)[..., None, None]          # (...,1,1)
    quat = np.take_along_axis(quat_candidates, idx, axis=-2)[..., 0, :]  # (...,4)

    # Re-normalise for numerical hygiene
    quat /= np.linalg.norm(quat, axis=-1, keepdims=True)
    return quat


def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.

    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler_angles_to_matrix(euler_angles: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.

    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def _angle_from_tan(
    axis: str, other_axis: str, data, horizontal: bool, tait_bryan: bool
) -> torch.Tensor:
    """
    Extract the first or third Euler angle from the two members of
    the matrix which are positive constant times its sine and cosine.

    Args:
        axis: Axis label "X" or "Y or "Z" for the angle we are finding.
        other_axis: Axis label "X" or "Y or "Z" for the middle axis in the
            convention.
        data: Rotation matrices as tensor of shape (..., 3, 3).
        horizontal: Whether we are looking for the angle for the third axis,
            which means the relevant entries are in the same row of the
            rotation matrix. If not, they are in the same column.
        tait_bryan: Whether the first and third axes in the convention differ.

    Returns:
        Euler Angles in radians for each matrix in data as a tensor
        of shape (...).
    """

    i1, i2 = {"X": (2, 1), "Y": (0, 2), "Z": (1, 0)}[axis]
    if horizontal:
        i2, i1 = i1, i2
    even = (axis + other_axis) in ["XY", "YZ", "ZX"]
    if horizontal == even:
        return torch.atan2(data[..., i1], data[..., i2])
    if tait_bryan:
        return torch.atan2(-data[..., i2], data[..., i1])
    return torch.atan2(data[..., i2], -data[..., i1])


def _index_from_letter(letter: str) -> int:
    if letter == "X":
        return 0
    if letter == "Y":
        return 1
    if letter == "Z":
        return 2
    raise ValueError("letter must be either X, Y or Z.")


def matrix_to_euler_angles(matrix: torch.Tensor, convention: str) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to Euler angles in radians.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).
        convention: Convention string of three uppercase letters.

    Returns:
        Euler angles in radians as tensor of shape (..., 3).
    """
    if len(convention) != 3:
        raise ValueError("Convention must have 3 letters.")
    if convention[1] in (convention[0], convention[2]):
        raise ValueError(f"Invalid convention {convention}.")
    for letter in convention:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    i0 = _index_from_letter(convention[0])
    i2 = _index_from_letter(convention[2])
    tait_bryan = i0 != i2
    if tait_bryan:
        central_angle = torch.asin(
            matrix[..., i0, i2] * (-1.0 if i0 - i2 in [-1, 2] else 1.0)
        )
    else:
        print(central_angle.min(), central_angle.max())
        central_angle = torch.acos(matrix[..., i0, i0])

    o = (
        _angle_from_tan(
            convention[0], convention[1], matrix[..., i2], False, tait_bryan
        ),
        central_angle,
        _angle_from_tan(
            convention[2], convention[1], matrix[..., i0, :], True, tait_bryan
        ),
    )
    return torch.stack(o, -1)


def random_quaternions(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random quaternions representing rotations,
    i.e. versors with nonnegative real part.

    Args:
        n: Number of quaternions in a batch to return.
        dtype: Type to return.
        device: Desired device of returned tensor. Default:
            uses the current device for the default tensor type.

    Returns:
        Quaternions as tensor of shape (N, 4).
    """
    if isinstance(device, str):
        device = torch.device(device)
    o = torch.randn((n, 4), dtype=dtype, device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o


def random_rotations(
    n: int, dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate random rotations as 3x3 rotation matrices.

    Args:
        n: Number of rotation matrices in a batch to return.
        dtype: Type to return.
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type.

    Returns:
        Rotation matrices as tensor of shape (n, 3, 3).
    """
    quaternions = random_quaternions(n, dtype=dtype, device=device)
    return quaternion_to_matrix(quaternions)


def random_rotation(
    dtype: Optional[torch.dtype] = None, device: Optional[Device] = None
) -> torch.Tensor:
    """
    Generate a single random 3x3 rotation matrix.

    Args:
        dtype: Type to return
        device: Device of returned tensor. Default: if None,
            uses the current device for the default tensor type

    Returns:
        Rotation matrix as tensor of shape (3, 3).
    """
    return random_rotations(1, dtype, device)[0]


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert a unit quaternion to a standard form: one in which the real
    part is non negative.

    Args:
        quaternions: Quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Standardized quaternions as tensor of shape (..., 4).
    """
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_raw_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions shape (..., 4).
    """
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return torch.stack((ow, ox, oy, oz), -1)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Multiply two quaternions representing rotations, returning the quaternion
    representing their composition, i.e. the versor with nonnegative real part.
    Usual torch rules for broadcasting apply.

    Args:
        a: Quaternions as tensor of shape (..., 4), real part first.
        b: Quaternions as tensor of shape (..., 4), real part first.

    Returns:
        The product of a and b, a tensor of quaternions of shape (..., 4).
    """
    ab = quaternion_raw_multiply(a, b)
    return standardize_quaternion(ab)


def quaternion_invert(quaternion: torch.Tensor) -> torch.Tensor:
    """
    Given a quaternion representing rotation, get the quaternion representing
    its inverse.

    Args:
        quaternion: Quaternions as tensor of shape (..., 4), with real part
            first, which must be versors (unit quaternions).

    Returns:
        The inverse, a tensor of quaternions of shape (..., 4).
    """

    scaling = torch.tensor([1, -1, -1, -1], device=quaternion.device)
    return quaternion * scaling


def quaternion_apply(quaternion: torch.Tensor, point: torch.Tensor) -> torch.Tensor:
    """
    Apply the rotation given by a quaternion to a 3D point.
    Usual torch rules for broadcasting apply.

    Args:
        quaternion: Tensor of quaternions, real part first, of shape (..., 4).
        point: Tensor of 3D points of shape (..., 3).

    Returns:
        Tensor of rotated points of shape (..., 3).
    """
    if point.size(-1) != 3:
        raise ValueError(f"Points are not in 3D, {point.shape}.")
    real_parts = point.new_zeros(point.shape[:-1] + (1,))
    point_as_quaternion = torch.cat((real_parts, point), -1)
    out = quaternion_raw_multiply(
        quaternion_raw_multiply(quaternion, point_as_quaternion),
        quaternion_invert(quaternion),
    )
    return out[..., 1:]


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def axis_angle_to_matrix_np(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix_np(axis_angle_to_quaternion_np(axis_angle))


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_axis_angle_np(matrix: np.ndarray) -> np.ndarray:
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle_np(matrix_to_quaternion_np(matrix))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = angles * 0.5
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def axis_angle_to_quaternion_np(axis_angle: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    NumPy implementation of axis-angle → quaternion conversion
    with the real part first, matching the PyTorch reference.

    Parameters
    ----------
    axis_angle : np.ndarray
        Array of shape (..., 3).  The vector’s magnitude is the
        rotation angle in radians; its direction is the rotation axis.
    eps : float, optional
        Threshold below which a series expansion is used to avoid
        numerical cancellation, by default 1e-6.

    Returns
    -------
    np.ndarray
        Quaternions of shape (..., 4) with (w, x, y, z) ordering.
    """
    axis_angle = np.asarray(axis_angle, dtype=np.float64)
    angles = np.linalg.norm(axis_angle, axis=-1, keepdims=True)          # (...,1)
    half_angles = 0.5 * angles                                           # (...,1)

    # Boolean mask for “small” angles
    small = np.abs(angles) < eps                                         # (...,1)

    sin_half_over_angle = np.empty_like(angles)

    # Regular case
    mask_reg = ~small
    if np.any(mask_reg):
        sin_half_over_angle[mask_reg] = (
            np.sin(half_angles[mask_reg]) / angles[mask_reg]
        )

    # Series expansion:  sin(θ/2)/θ ≈ ½ – θ²/48  (valid for θ≈0)
    if np.any(small):
        sin_half_over_angle[small] = 0.5 - (angles[small] ** 2) / 48.0

    quat = np.concatenate(
        [np.cos(half_angles), axis_angle * sin_half_over_angle], axis=-1
    )

    return quat


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def quaternion_to_axis_angle_np(quat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Quaternion(s) → axis-angle vector(s).

    * Input  : shape (..., 4) with (w, x, y, z) ordering.
    * Output : shape (..., 3) whose length is the rotation angle **≤ π**.

    The function:
    • normalises the quaternion (ignores scale),
    • uses a safe series expansion when the angle is tiny,
    • re-maps any angle > π to the equivalent rotation with
      angle (2π − θ) and flipped axis, so the returned magnitude
      is always in [0, π].
    """
    quat = np.asarray(quat, dtype=np.float64)

    # --- normalise (handles scale drift) ----------------------------
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    q = np.where(norm < eps, quat, quat / norm)          # unit-norm

    w, v = q[..., :1], q[..., 1:]                        # scalar & vector
    v_norm = np.linalg.norm(v, axis=-1, keepdims=True)

    half = np.arctan2(v_norm, w)                         # θ/2
    theta = 2.0 * half                                   # θ

    # sin(θ/2)/θ with numerical care
    small = np.abs(theta) < eps
    ratio = np.empty_like(theta)
    ratio[~small] = np.sin(half[~small]) / theta[~small]
    ratio[small] = 0.5 - (theta[small] ** 2) / 48.0      # series

    aa = v / ratio                                       # axis × angle

    # --- map angle to [0, π] for uniqueness -------------------------
    ang_mag = np.linalg.norm(aa, axis=-1)                # (...)
    flip = ang_mag > np.pi + 1e-8
    if np.any(flip):
        factor = (2 * np.pi - ang_mag[flip]) / ang_mag[flip]
        aa[flip] = -aa[flip] * factor[..., None]

    # identity → zero-vector
    aa[ang_mag < eps] = 0.0
    return aa


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """
    Converts 6D rotation representation by Zhou et al. [1] to rotation matrix
    using Gram--Schmidt orthogonalization per Section B of [1].
    Args:
        d6: 6D rotation representation, of size (*, 6)

    Returns:
        batch of rotation matrices of size (*, 3, 3)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """

    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def rotation_6d_to_matrix_np(d6: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Convert the 6-D rotation representation (Zhou et al. 2019) to a
    3 × 3 rotation matrix.

    *Input* – `d6`: shape (..., 6) with two stacked 3-vectors.
    *Output* – rotation matrix shape (..., 3, 3) whose **rows** are the
    orthonormal basis vectors of the rotation.

    A vector `v` is normalised as: `v / max(‖v‖, eps)` so the function
    stays well-defined even for nearly-zero inputs.
    """
    d6 = np.asarray(d6, dtype=np.float64)
    if d6.shape[-1] != 6:
        raise ValueError(f"Expected last dim = 6, got {d6.shape}.")

    a1, a2 = d6[..., :3], d6[..., 3:]

    def norm(v):                                     # ε-safe normalise
        return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), eps)

    b1 = norm(a1)
    # Gram–Schmidt step
    b2 = norm(a2 - (b1 * a2).sum(axis=-1, keepdims=True) * b1)
    b3 = np.cross(b1, b2, axis=-1)

    # Stack as ROWS to match the rest of the library
    R = np.stack((b1, b2, b3), axis=-2)              # (...,3,3)
    return R


# Convert rotation matrix back to rotation6d
def matrix_to_rotation_6d_np(R):
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = R.shape[:-2]
    return R[..., :2, :].reshape(batch_dim + (6,))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """
    Converts rotation matrices to 6D rotation representation by Zhou et al. [1]
    by dropping the last row. Note that 6D representation is not unique.
    Args:
        matrix: batch of rotation matrices of size (*, 3, 3)

    Returns:
        6D rotation representation, of size (*, 6)

    [1] Zhou, Y., Barnes, C., Lu, J., Yang, J., & Li, H.
    On the Continuity of Rotation Representations in Neural Networks.
    IEEE Conference on Computer Vision and Pattern Recognition, 2019.
    Retrieved from http://arxiv.org/abs/1812.07035
    """
    batch_dim = matrix.size()[:-2]
    return matrix[..., :2, :].clone().reshape(batch_dim + (6,))


def quaternion_to_rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
    return matrix_to_rotation_6d(quaternion_to_matrix(quaternion))


def rotation_6d_to_quaternion(d6: torch.Tensor) -> torch.Tensor:
    return matrix_to_quaternion(rotation_6d_to_matrix(d6))


def axis_angle_to_rotation_6d(aa: torch.Tensor) -> torch.Tensor:
    matrix = axis_angle_to_matrix(aa)
    return matrix_to_rotation_6d(matrix)


def rotation_6d_to_axis_angle(rot6d: torch.Tensor) -> torch.Tensor:
    matrix = rotation_6d_to_matrix(rot6d)
    return matrix_to_axis_angle(matrix)


def axis_angle_to_rotation_6d_np(aa: np.ndarray) -> np.ndarray:
    matrix = axis_angle_to_matrix_np(aa)
    return matrix_to_rotation_6d_np(matrix)

    
def rotation_6d_to_axis_angle_direct(d6):
    """
    Convert a 6D rotation representation (Zhou et al.) directly to an axis–angle representation.
    
    Args:
        d6: Tensor of shape (..., 6) representing the 6D rotation.
    
    Returns:
        Tensor of shape (..., 3) representing the axis–angle vector.
        The direction is the rotation axis, and the norm is the rotation angle in radians.
    """
    eps = 1e-7
    
    # 1) Convert 6D to orthonormal basis (b1, b2, b3) via Gram-Schmidt
    a1 = d6[..., :3]
    a2 = d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)                        # first basis vector
    proj = torch.sum(b1 * a2, dim=-1, keepdim=True)
    b2 = F.normalize(a2 - proj * b1, dim=-1)            # second basis vector
    b3 = torch.cross(b1, b2, dim=-1)                    # third basis vector
    
    # 2) Rotation angle via trace(R). Where R = [b1, b2, b3].
    #    trace(R) = b1.x + b2.y + b3.z
    trace = b1[..., 0] + b2[..., 1] + b3[..., 2]
    
    # clamp to valid range of acos
    cos_theta = (trace - 1.0) * 0.5  # => cos(theta)
    cos_theta_clamped = torch.clamp(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = torch.acos(cos_theta_clamped)               # => angle in [0, π]
    
    # 3) Axis via "vee(R - R^T)" = (1/(2sinθ))[b3.y - b2.z, b1.z - b3.x, b2.x - b1.y]
    #    We can compute these components directly without forming R-R^T:
    rx = b3[..., 1] - b2[..., 2]  # R[2,1] - R[1,2]
    ry = b1[..., 2] - b3[..., 0]  # R[0,2] - R[2,0]
    rz = b2[..., 0] - b1[..., 1]  # R[1,0] - R[0,1]
    skew_vec = torch.stack([rx, ry, rz], dim=-1)
    
    sin_theta = torch.sin(theta)
    # For small angles, sinθ ~ θ and (θ / 2 sinθ) ~ 1/2,
    # so use 0.5 as the fallback factor.
    factor = torch.where(
        (sin_theta.abs() > eps),
        theta / (2.0 * sin_theta),
        0.5 * torch.ones_like(theta)
    )
    
    axis_angle = factor.unsqueeze(-1) * skew_vec
    return axis_angle


def rotation_6d_to_axis_angle_direct_np(d6):
    """
    Convert a 6D rotation representation (Zhou et al.) directly to an axis–angle representation.

    Args:
        d6: ndarray of shape (..., 6) representing the 6D rotation.

    Returns:
        ndarray of shape (..., 3) representing the axis–angle vector.
        The direction is the rotation axis, and the norm is the rotation angle in radians.
    """
    eps = 1e-7

    # 1) Convert 6D to orthonormal basis (b1, b2, b3) via Gram-Schmidt
    a1 = d6[..., :3]
    a2 = d6[..., 3:]

    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - proj * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)

    # 2) Compute trace of rotation matrix R = [b1, b2, b3]
    trace = b1[..., 0] + b2[..., 1] + b3[..., 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta_clamped = np.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = np.arccos(cos_theta_clamped)

    # 3) Compute skew-symmetric vector (from R - R^T)
    rx = b3[..., 1] - b2[..., 2]
    ry = b1[..., 2] - b3[..., 0]
    rz = b2[..., 0] - b1[..., 1]
    skew_vec = np.stack([rx, ry, rz], axis=-1)

    sin_theta = np.sin(theta)
    factor = np.where(np.abs(sin_theta) > eps,
                      theta / (2.0 * sin_theta),
                      0.5)

    axis_angle = factor[..., np.newaxis] * skew_vec
    return axis_angle



def rotation_6d_to_axis_angle_np(d6):
    return rotation_6d_to_axis_angle_direct_np(d6)


def torch_rotation_matrix_from_vectors(vec1: torch.Tensor, vec2: torch.Tensor):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector of shape N,3
    :param vec2: A 3d "destination" vector of shape N,3
    :return mat: A transform matrix (Nx3x3) which when applied to vec1, aligns it with vec2.
    """
    a = vec1 / torch.norm(vec1, dim=-1, keepdim=True)
    b = vec2 / torch.norm(vec2, dim=-1, keepdim=True)
    
    v = torch.cross(a, b, dim=-1)
    c = torch.matmul(a.unsqueeze(1), b.unsqueeze(-1)).squeeze(-1)
    s = torch.norm(v, dim=-1, keepdim=True)
    kmat = torch.zeros(v.shape[0], 3, 3, device=v.device, dtype=v.dtype)
    kmat[:, 0, 1] = -v[:, 2]
    kmat[:, 0, 2] = v[:, 1]
    kmat[:, 1, 0] = v[:, 2]
    kmat[:, 1, 2] = -v[:, 0]
    kmat[:, 2, 0] = -v[:, 1]
    kmat[:, 2, 1] = v[:, 0]
    rot_mat = torch.eye(3, device=v.device, dtype=v.dtype).unsqueeze(0)
    rot_mat = rot_mat + kmat + torch.matmul(kmat, kmat) * ((1 - c) / (s ** 2)).unsqueeze(-1)
    return rot_mat


def batch_look_at_th(camera_position, look_at, camera_up_direction):
    r"""Generate transformation matrix for given camera parameters.
    Formula is :math:`\text{P_cam} = \text{P_world} * \text{transformation_mtx}`,
    with :math:`\text{P_world}` being the points coordinates padded with 1.
    Args:
        camera_position (torch.FloatTensor):
            camera positions of shape :math:`(\text{batch_size}, 3)`,
            it means where your cameras are
        look_at (torch.FloatTensor):
            where the camera is watching, of shape :math:`(\text{batch_size}, 3)`,
        camera_up_direction (torch.FloatTensor):
            camera up directions of shape :math:`(\text{batch_size}, 3)`,
            it means what are your camera up directions, generally [0, 1, 0]
    Returns:
        (torch.FloatTensor):
            The camera transformation matrix of shape :math:`(\text{batch_size}, 4, 3)`.
    """
    z_axis = (camera_position - look_at)
    z_axis /= z_axis.norm(dim=1, keepdim=True)
    x_axis = torch.cross(camera_up_direction, z_axis, dim=1)
    x_axis /= x_axis.norm(dim=1, keepdim=True)
    y_axis = torch.cross(z_axis, x_axis, dim=1)
    rot_part = torch.stack([x_axis, y_axis, z_axis], dim=2)
    trans_part = (-camera_position.unsqueeze(1) @ rot_part)
    trans_part = trans_part.permute(0,2,1)
    return rot_part, trans_part


def rotation_about_x(angle: float) -> torch.Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[1, 0, 0, 0], [0, cos, -sin, 0], [0, sin, cos, 0], [0, 0, 0, 1]])


def rotation_about_y(angle: float) -> torch.Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[cos, 0, sin, 0], [0, 1, 0, 0], [-sin, 0, cos, 0], [0, 0, 0, 1]])


def rotation_about_z(angle: float) -> torch.Tensor:
    cos = math.cos(angle)
    sin = math.sin(angle)
    return torch.tensor([[cos, -sin, 0, 0], [sin, cos, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])



def pitch_yaw_to_axis_angle_vector(pitch, yaw):
    """
    Convert pitch and yaw to axis-angle vector.
    
    :param pitch: Pitch angle (in radians).
    :param yaw: Yaw angle (in radians).
    :return: Axis-angle vector (3D vector where direction is axis, magnitude is rotation angle).
    """
    # Rotation matrix for pitch (rotation around x-axis)
    R_pitch = np.array([
        [1, 0, 0],
        [0, np.cos(pitch), -np.sin(pitch)],
        [0, np.sin(pitch), np.cos(pitch)]
    ])

    # Rotation matrix for yaw (rotation around y-axis)
    R_yaw = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)]
    ])

    # Combined rotation matrix (yaw first, then pitch)
    R_combined = np.dot(R_yaw, R_pitch)

    # Compute the angle of rotation (magnitude of axis-angle vector)
    angle = np.arccos((np.trace(R_combined) - 1) / 2)

    # If there is no rotation, return a zero vector
    if np.isclose(angle, 0):
        return np.zeros(3)

    # Compute the axis of rotation (normalized)
    axis = np.array([
        R_combined[2, 1] - R_combined[1, 2],
        R_combined[0, 2] - R_combined[2, 0],
        R_combined[1, 0] - R_combined[0, 1]
    ]) / (2 * np.sin(angle))

    # Return the axis-angle vector (axis * angle)
    return axis * angle

def axis_angle_vector_to_pitch_yaw(axis_angle_vector):
    """
    Convert axis-angle vector back to pitch and yaw.
    
    :param axis_angle_vector: 3D axis-angle vector.
    :return: Pitch and yaw angles (in radians).
    """
    # Compute the rotation angle (magnitude of the axis-angle vector)
    angle = np.linalg.norm(axis_angle_vector)
    
    # If the angle is zero, return zero pitch and yaw
    if np.isclose(angle, 0):
        return 0.0, 0.0

    # Compute the axis of rotation (normalize the axis-angle vector)
    axis = axis_angle_vector / angle

    # Use the Rodrigues' formula to get the rotation matrix
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    # Extract yaw (rotation around y-axis)
    yaw = np.arctan2(R[0, 2], R[0, 0])

    # Extract pitch (rotation around x-axis)
    pitch = np.arcsin(-R[0, 1])

    return pitch, yaw
