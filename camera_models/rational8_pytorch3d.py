import math
from typing import List, Optional, Tuple, Union

import torch
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.cameras import _R, _T, CamerasBase
from pytorch3d.transforms import Transform3d
from .projections import PerspectiveProjection


class Rational8CameraPytorch3D(CamerasBase, PerspectiveProjection):
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
            params: Distortion parameters tensor of shape (N, 8).
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
        self.p1 = self.params[:, 2].view(-1, 1)
        self.p2 = self.params[:, 3].view(-1, 1)
        self.k3 = self.params[:, 4].view(-1, 1)
        self.k4 = self.params[:, 5].view(-1, 1)
        self.k5 = self.params[:, 6].view(-1, 1)
        self.k6 = self.params[:, 7].view(-1, 1)


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
        
        # normalized pinhole
        xy = points_cam[..., :2] / points_cam[..., 2].unsqueeze(-1)
        
        r3 = torch.linalg.norm(points_cam, axis=-1)
        s = torch.sign(points_cam[..., 2]) # Z 
        batched_depth = (r3 * s).unsqueeze(-1)  # (N, P, 1)

        x, y = xy[..., 0], xy[..., 1]

        r2 = torch.clamp(x * x + y * y, 0, 1e10)      # (N, P)
        r4 = torch.clamp(r2 * r2, 0, 1e11)              # (N, P)
        r6 = torch.clamp(r2 * r4, 0, 1e12)              # (N, P)

        # Rational radial
        num = 1 + self.k1*r2 + self.k2*r4 + self.k3*r6
        den = 1 + self.k4*r2 + self.k5*r4 + self.k6*r6
        r_dist = num / den

        # Tangential
        xy_dist_x = x*r_dist + 2*self.p1*x*y + self.p2*(r2 + 2*x*x)
        xy_dist_y = y*r_dist + 2*self.p2*x*y + self.p1*(r2 + 2*y*y)

        xy_dist = torch.stack([xy_dist_x, xy_dist_y], dim=-1)  # (N, P, 2)

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
            xy_dist = (uv - self.principal_point[i]) / self.focal_length[i]  # (P, 2)
            xy_undist = self.distortion_models[i].inverse_evaluate(xy_dist)  # (P, 2)

            x_u = xy_undist[..., 0]
            y_u = xy_undist[..., 1]

            dir_3d = torch.stack([x_u, y_u, torch.ones_like(x_u)], axis=-1)
            norm_dir = torch.linalg.norm(dir_3d, axis=-1)

            sign_z = torch.sign(d)
            mag = torch.abs(d)
            scale = mag / torch.maximum(norm_dir, 1e-12)
            points_cam = dir_3d * scale[..., None]
            points_cam[..., 2] *= sign_z
            
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

    def inverse_evaluate(self, uv, tol=1e-7, max_iter=10):
        xy = uv.clone() 

        for idx in range(max_iter):
            x, y = xy[..., 0], xy[..., 1]

            r2 = torch.clip(x * x + y * y, 0, 1e10)
            r4 = torch.clip(r2 * r2, 0, 1e11)
            r6 = torch.clip(r4 * r2, 0, 1e12)

            num = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
            den = 1 + self.k4 * r2 + self.k5 * r4 + self.k6 * r6
            r_dist = num / den

            F1 = x * r_dist + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
            F2 = y * r_dist + 2 * self.p2 * x * y + self.p1 * (r2 + 2 * y * y)
            F_xy = torch.stack([F1, F2], dim=-1)


            error = F_xy - uv
            error_norm = torch.linalg.norm(error, axis=-1)
            if torch.all(error_norm < tol):
                break

            common = (2.0 * ( (den * (self.k1 + 4 * self.k2 * r2 + 6 * self.k3 * r4)
                            - num * (self.k4 + 4 * self.k5 * r2 + 6 * self.k6 * r4) ) ))
            common = common / (den * den)
            dr_dx = common * x
            dr_dy = common * y


            J11 = r_dist + x * dr_dx + 2 * self.p1 * y + 6 * self.p2 * x
            J12 = x * dr_dy + 2 * self.p1 * x + 2 * self.p2 * y
            J21 = y * dr_dx + 2 * self.p2 * y + 2 * self.p1 * x
            J22 = r_dist + y * dr_dy + 2 * self.p2 * x + 6 * self.p1 * y

            det = J11 * J22 - J12 * J21

            delta_x = (error[..., 0] * J22 - error[..., 1] * J12) / det
            delta_y = (error[..., 1] * J11 - error[..., 0] * J21) / det
            delta = torch.stack([delta_x, delta_y], axis=-1)

            xy = xy - delta
        
            err_norm = torch.linalg.norm(delta, dim=-1)
            invalid_mask = (torch.isnan(delta).any(dim=-1)) | (torch.isinf(delta).any(dim=-1)) | (err_norm > 1e8)
            if invalid_mask.any():
                xy[invalid_mask] = uv[invalid_mask]  # reset invalid points
                delta[invalid_mask] = 0
            if (err_norm < tol).all():
                break

        return xy
    
    def distort_kpts2d(self, uv) -> torch.Tensor:
        xy = (uv - self.principal_point.unsqueeze(1)) / self.focal_length.unsqueeze(1)  # (N, P, 2) 

        x, y = xy[..., 0], xy[..., 1]

        r2 = torch.clamp(x * x + y * y, 0, 1e10)      # (N, P)
        r4 = torch.clamp(r2 * r2, 0, 1e11)              # (N, P)
        r6 = torch.clamp(r2 * r4, 0, 1e12)              # (N, P)

        # Rational radial
        num = 1 + self.k1*r2 + self.k2*r4 + self.k3*r6
        den = 1 + self.k4*r2 + self.k5*r4 + self.k6*r6
        r_dist = num / den

        # Tangential
        xy_dist_x = x*r_dist + 2*self.p1*x*y + self.p2*(r2 + 2*x*x)
        xy_dist_y = y*r_dist + 2*self.p2*x*y + self.p1*(r2 + 2*y*y)

        xy_dist = torch.stack([xy_dist_x, xy_dist_y], dim=-1)  # (N, P, 2)

        uv_dist = xy_dist * self.focal_length.unsqueeze(1) + self.principal_point.unsqueeze(1)

        return uv_dist

    def uvd_to_camera(self, uvd):
        uv = uvd[..., :2]
        d = uvd[..., 2]

        q = (uv - self.principal_point.unsqueeze(1)) / self.focal_length.unsqueeze(1)   # Normalized (X/Z, Y/Z) coordinates

        denominator = q[..., 0]**2 + q[..., 1]**2 + 1
        z = torch.sign(d) * torch.sqrt(d**2 / denominator)

        x = q[..., 0] * z 
        y = q[..., 1] * z  

        return torch.stack([x, y, z], dim=-1)
