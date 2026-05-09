import math
from typing import List, Optional, Tuple, Union

import torch
import numpy as np
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.cameras import _R, _T, CamerasBase
from pytorch3d.transforms import Transform3d

from .fisheye624 import ArctanProjection


class FishEyeCamera624Pytorch3D(CamerasBase, ArctanProjection):
    """
    A fisheye camera model that uses ArctanProjection for projection,
    """

    _FIELDS = (
        "focal_length",
        "principal_point",
        "R",
        "T",
        "params",
        "world_coordinates",
        "device",
        "image_size",
    )

    def __init__(
        self,
        focal_length,
        principal_point,
        params,
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        world_coordinates: bool = False,
        device: Device = "cpu",
        image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    ) -> None:
        """
        Initializes the FishEyeCameras class with the given parameters.

        Args:
            focal_length: Focal length tensor of shape (N, 2), where N is the number of cameras.
            principal_point: Principal point tensor of shape (N, 2).
            params: Distortion parameters tensor of shape (N, 12).
            R: Rotation matrices of shape (N, 3, 3).
            T: Translation vectors of shape (N, 3).
            world_coordinates: Whether to use world coordinates.
            device: The device to run the computations on.
            image_size: Image size tensor of shape (N, 2).
        """
        super().__init__(
            device=device,
            R=R,
            T=T,
            image_size=image_size,
        )
        if image_size is not None:
            if (self.image_size < 1).any():
                raise ValueError("Image_size provided has invalid values")
        else:
            self.image_size = None

        self.device = device
        self.focal_length = focal_length.to(self.device)
        self.principal_point = principal_point.to(self.device)
        self.params = params.to(self.device)
        self.R = R.to(self.device)
        self.T = T.to(self.device)
        self.world_coordinates = world_coordinates

        self.k1 = self.params[:, 0].view(-1, 1, 1)
        self.k2 = self.params[:, 1].view(-1, 1, 1)
        self.k3 = self.params[:, 2].view(-1, 1, 1)
        self.k4 = self.params[:, 3].view(-1, 1, 1)
        self.k5 = self.params[:, 4].view(-1, 1, 1)
        self.k6 = self.params[:, 5].view(-1, 1, 1)
        self.p1 = self.params[:, 6].view(-1, 1, 1)
        self.p2 = self.params[:, 7].view(-1, 1, 1)
        self.s1 = self.params[:, 8].view(-1, 1, 1)
        self.s2 = self.params[:, 9].view(-1, 1, 1)
        self.s3 = self.params[:, 10].view(-1, 1, 1)
        self.s4 = self.params[:, 11].view(-1, 1, 1)

    def evaluate(self, p):
        k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
        p1, p2 = self.p1, self.p2
        s1, s2, s3, s4 = self.s1, self.s2, self.s3, self.s4

        # radial component
        r2 = (p * p).sum(dim=-1, keepdims=True)
        r2 = torch.clip(r2, -np.pi**2, np.pi**2)
            
        r4 = r2 * r2
        r6 = r2 * r4
        r8 = r4 * r4
        r10 = r4 * r6
        r12 = r6 * r6
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
        uv = p * radial

        # tangential component
        x, y = uv[..., 0:1], uv[..., 1:2]
        x2 = x * x
        y2 = y * y
        xy = x * y
        r2 = x2 + y2
        x = x + 2 * p2 * xy + p1 * (r2 + 2 * x2)
        y = y + 2 * p1 * xy + p2 * (r2 + 2 * y2)

        # thin prism
        r4 = r2 * r2
        x = x + s1 * r2 + s2 * r4
        y = y + s3 * r2 + s4 * r4

        xy_dist = torch.cat((x, y), dim=-1)

        return xy_dist

    def transform_points(
        self, points, eps: Optional[float] = None, **kwargs
    ) -> torch.Tensor:
        """
        Projects 3D points from world or camera coordinates to image coordinates.

        Args:
            points: Tensor of shape (..., 3).
            eps: Small value to avoid division by zero.

        Returns:
            Tensor of projected points of shape (..., 3).
        """
        # Transform points to camera coordinates if in world coordinates
        if self.world_coordinates:
            world_to_view_transform = self.get_world_to_view_transform(
                R=self.R, T=self.T
            )
            points_cam = world_to_view_transform.transform_points(
                points.to(self.device), eps=eps
            )
        else:
            points_cam = points.to(self.device)

        xyd = self.project3(points_cam)
        xy_dist = self.evaluate(xyd[..., :2])

        batched_depth = xyd[..., 2:]  # Depth

        uv = xy_dist * self.focal_length.unsqueeze(1) + self.principal_point.unsqueeze(1)

        ndc_uv = 2.0 * uv / self.image_size.flip(1).unsqueeze(1)  - 1.0  # Flip to get (W, H)
        projected_points = torch.cat([ndc_uv, batched_depth], dim=-1) # (N, P, 3)

        return projected_points

    def unproject_points(
        self,
        xy_depth: torch.Tensor,
        world_coordinates: bool = True,
        scaled_depth_input: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Unprojects 2D points with depth back to 3D points in camera or world coordinates.

        Args:
            xy_depth: Tensor of shape (..., 3).
            world_coordinates: Whether to transform points to world coordinates.

        Returns:
            Tensor of unprojected 3D points of shape (..., 3).
        """
        xy_depth = xy_depth.to(self.device)
        batch_size = self.params.shape[0]
        original_shape = xy_depth.shape

        from_ndc = kwargs.get("from_ndc", False)

        assert scaled_depth_input, "scaled_depth_input must be True for FishEyeCameras"

        unprojected_points_list = []
        # Unproject points for each camera in the batch
        for i in range(batch_size):
            if from_ndc:
                p_ndc = xy_depth[i]
        
                x_ndc = p_ndc[..., 0]  # (P,)
                y_ndc = p_ndc[..., 1]  # (P,)
                d = p_ndc[..., 2:]  # (P,)

                # Convert NDC coordinates back to image (pixel) coordinates
                # Formula: u = (x_ndc + 1) * W / 2, v = (-y_ndc + 1) * H / 2
                u = (x_ndc + 1.0) * self.image_size[i, 1] / 2.0  # (P,)
                v = (y_ndc + 1.0) * self.image_size[i, 0] / 2.0  # (P,)
                uv = torch.stack([u, v], dim=-1)  # (P, 2)
            else:
                uv = xy_depth[i, ..., :2]
                d = xy_depth[i, ..., 2:]
                
            # Normalize image coordinates by subtracting principal point and dividing by focal length
            q = (uv - self.principal_point[i]) / self.focal_length[i]  # (P, 2)
            q = self.distortion_models[i].inverse_evaluate(q)  # (P, 2)

            uvd  = torch.cat([q, d], dim=-1)  # (P, 3)
            points_cam = self.unproject3(uvd)

            unprojected_points_list.append(points_cam)
        
        points_cam  = torch.stack(unprojected_points_list, dim=0).view(original_shape)
        
        if world_coordinates:
            # Transform points from camera to world coordinates
            view_to_world_transform = self.get_world_to_view_transform(R=self.R, T=self.T).inverse()  # (N, 3, 3) and (N, 3)
            points_world = view_to_world_transform.transform_points(
                points_cam, eps=kwargs.get('eps', None)
            )  # (N, P, 3)
            return points_world
        else:
            return points_cam

    def transform_points_screen(self, points, **kwargs) -> torch.Tensor:
        ndc_d = self.transform_points(points, **kwargs)
        ndcxy = ndc_d[..., :2]

        x_ndc = ndcxy[..., 0]  # (P,)
        y_ndc = ndcxy[..., 1]
        d = ndc_d[..., 2]

        u = (-x_ndc + 1.0) * self.image_size[:, 1] / 2.0  # (P,)
        v = (-y_ndc + 1.0) * self.image_size[:, 0] / 2.0  # (P,)

        return torch.stack([u, v, d], dim=-1)

    def transform_points_screen_cv2(self, points, **kwargs) -> torch.Tensor:
        ndc_d = self.transform_points(points, **kwargs)

        ndcxy = ndc_d[..., :2]

        x_ndc = ndcxy[..., 0]  # (P,)
        y_ndc = ndcxy[..., 1]
        d = ndc_d[..., 2]
        
        u = (x_ndc + 1.0) * self.image_size[:, 1:] / 2.0  # (P,)
        v = (y_ndc + 1.0) * self.image_size[:, :1] / 2.0  # (P,)

        return torch.stack([u, v, d], dim=-1)

    def in_ndc(self):
        """
        Indicates that the camera coordinates are not in normalized device coordinates.
        """
        return True

    def is_perspective(self):
        """
        Indicates that the camera model is not a perspective camera.
        """
        return False


    def get_ndc_camera_transform(self, **kwargs) -> Transform3d:
        """
        Returns a Transform3d that maps projected 2D points (in image coordinates)
        to NDC coordinates.
        """
        if self.in_ndc():
            return Transform3d(device=self.device, dtype=torch.float32)

    def undistort_kpts2d(self, keypoints, K_new=None) -> torch.Tensor:
        keypoints_norm = torch.empty_like(keypoints, dtype=torch.float32)
        
        if K_new is None:
            K_new = torch.eye(3, device=keypoints.device).unsqueeze(0).repeat(keypoints.shape[0], 1, 1)
            
            K_new[..., 0, 0] = self.focal_length[:, 0]
            K_new[..., 1, 1] = self.focal_length[:, 1]
            K_new[..., 0, 2] = self.principal_point[:, 0]
            K_new[..., 1, 2] = self.principal_point[:, 1]

        if K_new.dim() == 3 and keypoints.dim() == 3:
            K_new = K_new.unsqueeze(1)

        keypoints_norm[..., 0] = (keypoints[..., 0] - K_new[..., 0, 2]) / K_new[..., 0, 0]
        keypoints_norm[..., 1] = (keypoints[..., 1] - K_new[..., 1, 2]) / K_new[..., 1, 1]
        
        # Apply the inverse distortion mapping to get the "ideal" undistorted normalized coordinates.
        # (Note: Your distortion model is non-linear so this step is key for alignment)
        keypoints_undist_norm = self.inverse_evaluate(keypoints_norm)
        
        # Reproject the undistorted normalized coordinates with K_new (or any new desired intrinsics).
        keypoints_undist = torch.empty_like(keypoints, dtype=torch.float32)
        keypoints_undist[..., 0] = keypoints_undist_norm[..., 0] * K_new[..., 0, 0] + K_new[..., 0, 2]
        keypoints_undist[..., 1] = keypoints_undist_norm[..., 1] * K_new[..., 1, 1] + K_new[..., 1, 2]
        
        return keypoints_undist

    def inverse_evaluate(self, uv, tol=1e-6, max_iter=15):
        k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
        p1, p2 = self.p1, self.p2
        s1, s2, s3, s4 = self.s1, self.s2, self.s3, self.s4

        p = uv.clone()

        for idx in range(max_iter):
            # radial component
            x, y = p[..., 0:1], p[..., 1:2]
            r2 = x * x + y * y
            r4 = r2 * r2
            r6 = r2 * r4
            r8 = r4 * r4
            r10 = r4 * r6
            r12 = r6 * r6
            radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12

            # Compute partial derivatives of the radial factor.
            dradial_dx = 2 * k1 * x + 4 * k2 * r2 * x + 6 * k3 * r4 * x + 8 * k4 * r6 * x + 10 * k5 * r8 * x + 12 * k6 * r10 * x
            dradial_dy = 2 * k1 * y + 4 * k2 * r2 * y + 6 * k3 * r4 * y + 8 * k4 * r6 * y + 10 * k5 * r8 * y + 12 * k6 * r10 * y

            # tangential component
            delta_x = 2 * p2 * x * y + p1 * (r2 + 2 * x * x)
            delta_y = 2 * p1 * x * y + p2 * (r2 + 2 * y * y)

            # Derivatives of the tangential component.
            ddelta_x_dx = 2 * p2 * y + 6 * p1 * x
            ddelta_x_dy = 2 * p2 * x + 2 * p1 * y
            ddelta_y_dx = 2 * p1 * y + 2 * p2 * x
            ddelta_y_dy = 2 * p1 * x + 6 * p2 * y

            # thin prism
            prism_x = s1 * r2 + s2 * r4
            prism_y = s3 * r2 + s4 * r4

            # Derivatives of the thin prism terms.
            dprism_x_dx = 2 * s1 * x + 4 * s2 * r2 * x
            dprism_x_dy = 2 * s1 * y + 4 * s2 * r2 * y
            dprism_y_dx = 2 * s3 * x + 4 * s4 * r2 * x
            dprism_y_dy = 2 * s3 * y + 4 * s4 * r2 * y

            # Intermediate functions representing the “nonradial” part.
            # Let g = x + delta_x + prism_x and h = y + delta_y + prism_y.
            g = x + delta_x + prism_x
            h = y + delta_y + prism_y
            dg_dx = 1 + ddelta_x_dx + dprism_x_dx
            dg_dy = ddelta_x_dy + dprism_x_dy
            dh_dx = ddelta_y_dx + dprism_y_dx
            dh_dy = 1 + ddelta_y_dy + dprism_y_dy

            # The distortion function is defined as:
            #   F(p) = [ g * radial,  h * radial ]
            F1 = g * radial
            F2 = h * radial

            # f(p) = F(p) - uv should be zero at the correct undistorted point.
            error1 = F1 - uv[..., 0:1]
            error2 = F2 - uv[..., 1:2]

            # Stack the error components.
            error = torch.cat([error1, error2], axis=-1)

            # Compute the Jacobian matrix J = dF/dp.
            # For the x-coordinate:
            dF1_dx = dg_dx * radial + g * dradial_dx
            dF1_dy = dg_dy * radial + g * dradial_dy
            # For the y-coordinate:
            dF2_dx = dh_dx * radial + h * dradial_dx
            dF2_dy = dh_dy * radial + h * dradial_dy

            # Assemble the Jacobian (last two dimensions: 2x2).
            J = torch.stack([torch.cat([dF1_dx, dF1_dy], axis=-1),
                             torch.cat([dF2_dx, dF2_dy], axis=-1)], axis=-2)

            # Solve for the Newton update: delta such that J * delta = -error.
            delta = -torch.linalg.solve(J, error.unsqueeze(-1)).squeeze(-1)

            # Update the current guess.
            p = p + delta

            # Optionally, handle pathological cases.
            err_norm = torch.linalg.norm(delta, dim=-1)
            invalid_mask = (torch.isnan(delta).any(dim=-1)) | (torch.isinf(delta).any(dim=-1)) | (err_norm > 1e8)
            if invalid_mask.any():
                p[invalid_mask] = uv[invalid_mask]  # reset invalid points
                delta[invalid_mask] = 0
            if (err_norm < tol).all():
                break
            
        return p

    def uvd_to_camera(self, uvd):
        uv = uvd[..., :2]
        d = uvd[..., 2]

        q = (uv - self.principal_point.unsqueeze(1)) / self.focal_length.unsqueeze(1)  

        qd = torch.cat([q, d.unsqueeze(-1)], dim=-1)

        return self.unproject3(qd)
