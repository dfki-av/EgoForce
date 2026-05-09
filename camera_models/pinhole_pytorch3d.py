import math
from typing import List, Optional, Tuple, Union

import torch
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.cameras import _R, _T, CamerasBase, PerspectiveCameras
from pytorch3d.transforms import Transform3d

from .projections import PerspectiveProjection


class PinholeCameraPytorch3D(PerspectiveCameras, PerspectiveProjection):
    _FIELDS = (
        "focal_length",
        "principal_point",
        "R",
        "T",
        "world_coordinates",
        "device",
        "image_size",
    )

    def __init__(
        self,
        focal_length,
        principal_point,
        R: torch.Tensor = _R,
        T: torch.Tensor = _T,
        world_coordinates: bool = False,
        device: Device = "cpu",
        image_size: Optional[Union[List, Tuple, torch.Tensor]] = None,
    ) -> None:
        """
        Initializes the PinholeCamera class with the given parameters.

        Args:
            focal_length: Focal length tensor of shape (N, 2), where N is the number of cameras.
            principal_point: Principal point tensor of shape (N, 2).
            R: Rotation matrices of shape (N, 3, 3).
            T: Translation vectors of shape (N, 3).
            world_coordinates: Whether to use world coordinates.
            device: The device to run the computations on.
            image_size: Image size tensor of shape (N, 2).
        """
        super().__init__(
            device=device,
            focal_length=focal_length,
            principal_point=principal_point,
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
        self.R = R.to(self.device)
        self.T = T.to(self.device)
        self.world_coordinates = world_coordinates

    def transform_points_screen_cv2(self, points, **kwargs) -> torch.Tensor:
        B = points.shape[0]

        # if self.focal_length.shape[0] == 1:
        #     ndc_d = []
        #     for bdx in range(B):
        #         ndc_d_i = self.transform_points(points[bdx:bdx+1], **kwargs)
        #         ndc_d.append(ndc_d_i)
        #     ndc_d = torch.cat(ndc_d, dim=0)
        # else:
        ndc_d = self.transform_points(points, **kwargs)

        ndcxy = ndc_d[..., :2]

        u = ndcxy[..., 0]  # (P,)
        v = ndcxy[..., 1]
        d = ndc_d[..., 2]

        return torch.stack([u, v, d], dim=-1)

    def undistort_kpts2d(self, pts2d, K=None) -> torch.Tensor:
        return pts2d

    def uvd_to_camera(self, uvd):
        uv = uvd[..., :2]
        d = uvd[..., 2]

        q = (uv - self.principal_point.unsqueeze(1)) / self.focal_length.unsqueeze(1)   # Normalized (X/Z, Y/Z) coordinates

        denominator = q[..., 0]**2 + q[..., 1]**2 + 1
        z = torch.sign(d) * torch.sqrt(d**2 / denominator)

        x = q[..., 0] * z 
        y = q[..., 1] * z  

        return torch.stack([x, y, z], dim=-1)
