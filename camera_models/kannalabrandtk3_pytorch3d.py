import math
from typing import List, Optional, Tuple, Union

import torch
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.cameras import _R, _T, CamerasBase
from pytorch3d.transforms import Transform3d


from .projections import PerspectiveProjection


class KannalaBrandtK3CameraPytorch3D(CamerasBase, PerspectiveProjection):
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
        Initializes the Rational8Camera class with the given parameters.

        Args:
            focal_length: Focal length tensor of shape (N, 2), where N is the number of cameras.
            principal_point: Principal point tensor of shape (N, 2).
            params: Distortion parameters tensor of shape (N, 4).
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

        self.k1 = self.params[:, 0].view(-1, 1)
        self.k2 = self.params[:, 1].view(-1, 1)
        self.k3 = self.params[:, 2].view(-1, 1)
        self.k4 = self.params[:, 3].view(-1, 1)

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

        batch_size = self.params.shape[0]
        
        r3 = torch.linalg.norm(points_cam, axis=-1)
        s = torch.sign(points_cam[..., 2]) # Z 
        batched_depth = (r3 * s).unsqueeze(-1)  # (N, P, 1)

        # normalized pinhole
        xy_norm = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
        xy_norm_dist = self.evaluate(xy_norm)  # (N, P, 2)

        uv = xy_norm_dist * self.focal_length.unsqueeze(1) + self.principal_point.unsqueeze(1)

        ndc_uv = 2.0 * uv / self.image_size.flip(1).unsqueeze(1)  - 1.0  # Flip to get (W, H)
        projected_points = torch.cat([ndc_uv, batched_depth], dim=-1) # (N, P, 3)

        return projected_points

    def evaluate(self, xy_norm):
        r = torch.linalg.norm(xy_norm, axis=-1)
        theta = torch.arctan(r)

        theta_d = theta * (
                1
                + self.k1 * (theta ** 2) # theta^2
                + self.k2 * (theta ** 4) # theta^4
                + self.k3 * (theta ** 6) # theta^6
                + self.k4 * (theta ** 8) # theta^8
        )

        scale = torch.where(r > 1e-8, theta_d / r, torch.ones_like(r))

        xy_norm_dist = xy_norm * scale[..., None]

        return xy_norm_dist


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
        B = points.shape[0]

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


    def inverse_evaluate(self, uv, tol=1e-7, max_iter=10):
        xy = uv.clone() 

        for _ in range(max_iter):
            x, y = xy[..., 0], xy[..., 1]
            r = torch.linalg.norm(xy, dim=-1)
            theta = torch.arctan(r)
            theta2 = theta**2
            theta4 = theta2**2
            theta6 = theta2 * theta4
            theta8 = theta4**2

            theta_d = theta * (1 + self.k1*theta2 + self.k2*theta4 + self.k3*theta6 + self.k4*theta8)

            scale = torch.where(r > 1e-8, theta_d / r, torch.ones_like(r))

            F_xy = xy * scale[..., None]

            error = F_xy - uv
            if torch.max(torch.linalg.norm(error, dim=-1)) < tol:
                break

            dtheta_dr = 1 / (1 + r**2)
            dtheta_d_dtheta = (1 + 3*self.k1*theta2 + 5*self.k2*theta4 + 7*self.k3*theta6 + 9*self.k4*theta8)
            dscale_dr = (dtheta_d_dtheta * dtheta_dr * r - theta_d) / (r**2 + 1e-8)

            J11 = scale + x**2 * dscale_dr / (r + 1e-8)
            J22 = scale + y**2 * dscale_dr / (r + 1e-8)
            J12 = x * y * dscale_dr / (r + 1e-8)
            J21 = J12

            det = J11 * J22 - J12 * J12
            singular = torch.abs(det) < 1e-12

            delta = torch.zeros_like(xy)
            good = ~singular
            if torch.any(good):
                delta_x = (error[..., 0][good] * J22[good] - error[..., 1][good] * J12[good]) / det[good]
                delta_y = (error[..., 1][good] * J11[good] - error[..., 0][good] * J12[good]) / det[good]
                update = torch.stack([delta_x, delta_y], dim=-1)
                delta[good] = update

            xy = xy - delta

            # Optionally, handle pathological cases.
            err_norm = torch.linalg.norm(delta, dim=-1)
            invalid_mask = (torch.isnan(delta).any(dim=-1)) | (torch.isinf(delta).any(dim=-1)) | (err_norm > 1e8)
            invalid_mask |= singular

            if invalid_mask.any():
                xy[invalid_mask] = uv[invalid_mask]  # reset invalid points
                delta[invalid_mask] = 0

            if (err_norm < tol).all():
                break
            
        return xy
        
    def distort_kpts2d(self, uv) -> torch.Tensor:
        xy_norm = (uv - self.principal_point.unsqueeze(1)) / self.focal_length.unsqueeze(1)  # (N, P, 2) 
        xy_dist = self.evaluate(xy_norm)  # (N, P, 2)
        uv_dist = xy_dist * self.focal_length.unsqueeze(1) + self.principal_point.unsqueeze(1)

        return uv_dist
    
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

    def uvd_to_camera(self, uvd):
        uv = uvd[..., :2]
        d = uvd[..., 2]

        q = (uv - self.principal_point.unsqueeze(1)) / self.focal_length.unsqueeze(1)   # Normalized (X/Z, Y/Z) coordinates

        denominator = q[..., 0]**2 + q[..., 1]**2 + 1
        z = torch.sign(d) * torch.sqrt(d**2 / denominator)

        x = q[..., 0] * z 
        y = q[..., 1] * z  

        return torch.stack([x, y, z], dim=-1)
