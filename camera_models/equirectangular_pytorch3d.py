import math
from typing import List, Optional, Tuple, Union

import torch
from pytorch3d.common.datatypes import Device
from pytorch3d.renderer.cameras import _R, _T, CamerasBase
from pytorch3d.transforms import Transform3d


class EquirectangularCameraPytorch3D(CamerasBase):
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
		self.R = R.to(self.device)
		self.T = T.to(self.device)
		self.world_coordinates = world_coordinates

	def evaluate(self, points_cam: torch.Tensor) -> torch.Tensor:
		x = points_cam[..., 0:1]
		y = points_cam[..., 1:2]
		z = points_cam[..., 2:3]

		eps = 1e-12
		r = torch.linalg.norm(points_cam, dim=-1, keepdim=True)
		yn = torch.clamp(y / torch.clamp(r, min=eps), min=-1.0, max=1.0)

		lon = torch.atan2(x, z)
		lat = torch.asin(yn)
		return torch.cat([lon, lat], dim=-1)

	def inverse_evaluate(self, q: torch.Tensor) -> torch.Tensor:
		lon = q[..., 0:1]
		lat = q[..., 1:2]

		cos_lat = torch.cos(lat)
		x = cos_lat * torch.sin(lon)
		y = torch.sin(lat)
		z = cos_lat * torch.cos(lon)

		rays = torch.cat([x, y, z], dim=-1)
		rays = rays / torch.clamp(torch.linalg.norm(rays, dim=-1, keepdim=True), min=1e-12)
		return rays

	def transform_points(
		self, points, eps: Optional[float] = None, **kwargs
	) -> torch.Tensor:
		if self.world_coordinates:
			world_to_view_transform = self.get_world_to_view_transform(R=self.R, T=self.T)
			points_cam = world_to_view_transform.transform_points(points.to(self.device), eps=eps)
		else:
			points_cam = points.to(self.device)

		r3 = torch.linalg.norm(points_cam, dim=-1)
		s = torch.sign(points_cam[..., 2])
		batched_depth = (r3 * s).unsqueeze(-1)

		q = self.evaluate(points_cam)
		uv = q * self.focal_length.unsqueeze(1) + self.principal_point.unsqueeze(1)

		ndc_uv = 2.0 * uv / self.image_size.flip(1).unsqueeze(1) - 1.0
		projected_points = torch.cat([ndc_uv, batched_depth], dim=-1)
		return projected_points

	def unproject_points(
		self,
		xy_depth: torch.Tensor,
		world_coordinates: bool = True,
		scaled_depth_input: bool = True,
		**kwargs,
	) -> torch.Tensor:
		xy_depth = xy_depth.to(self.device)
		from_ndc = kwargs.get("from_ndc", False)

		assert scaled_depth_input, "scaled_depth_input must be True for EquirectangularCameraPytorch3D"

		if from_ndc:
			uv = (xy_depth[..., :2] + 1.0) * self.image_size.flip(1).unsqueeze(1) / 2.0
			d = xy_depth[..., 2:3]
		else:
			uv = xy_depth[..., :2]
			d = xy_depth[..., 2:3]

		q = (uv - self.principal_point.unsqueeze(1)) / self.focal_length.unsqueeze(1)
		rays = self.inverse_evaluate(q)
		points_cam = rays * torch.abs(d)

		if world_coordinates:
			view_to_world_transform = self.get_world_to_view_transform(R=self.R, T=self.T).inverse()
			return view_to_world_transform.transform_points(points_cam, eps=kwargs.get("eps", None))

		return points_cam

	def transform_points_screen(self, points, **kwargs) -> torch.Tensor:
		ndc_d = self.transform_points(points, **kwargs)
		ndcxy = ndc_d[..., :2]

		x_ndc = ndcxy[..., 0]
		y_ndc = ndcxy[..., 1]
		d = ndc_d[..., 2]

		u = (-x_ndc + 1.0) * self.image_size[:, 1] / 2.0
		v = (-y_ndc + 1.0) * self.image_size[:, 0] / 2.0

		return torch.stack([u, v, d], dim=-1)

	def transform_points_screen_cv2(self, points, **kwargs) -> torch.Tensor:
		ndc_d = self.transform_points(points, **kwargs)
		ndcxy = ndc_d[..., :2]

		x_ndc = ndcxy[..., 0]
		y_ndc = ndcxy[..., 1]
		d = ndc_d[..., 2]

		u = (x_ndc + 1.0) * self.image_size[:, 1:] / 2.0
		v = (y_ndc + 1.0) * self.image_size[:, :1] / 2.0

		return torch.stack([u, v, d], dim=-1)

	def in_ndc(self):
		return True

	def is_perspective(self):
		return False

	def get_ndc_camera_transform(self, **kwargs) -> Transform3d:
		if self.in_ndc():
			return Transform3d(device=self.device, dtype=torch.float32)

	def undistort_kpts2d(self, keypoints, K_new=None) -> torch.Tensor:
		return keypoints

	def uvd_to_camera(self, uvd):
		uv = uvd[..., :2]
		d = uvd[..., 2:3]

		q = (uv - self.principal_point.unsqueeze(1)) / self.focal_length.unsqueeze(1)
		rays = self.inverse_evaluate(q)
		return rays * torch.abs(d)

