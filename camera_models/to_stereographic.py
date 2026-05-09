import numpy as np
import torch
from typing import Any, Callable, Optional, Tuple, Union

from .to_pinhole_camera import make_project_from_src_cam, warp_by_calibration_batched

Array = Union[np.ndarray, torch.Tensor]


def _is_torch(x: Array) -> bool:
	return isinstance(x, torch.Tensor)


def _backend_intrinsics(param: Array, like: Array) -> Array:
	if _is_torch(like):
		if isinstance(param, torch.Tensor):
			out = param.to(dtype=like.dtype, device=like.device)
		else:
			out = torch.tensor(param, dtype=like.dtype, device=like.device)
	else:
		out = np.asarray(param, dtype=like.dtype)

	if out.ndim == 2:
		# (B,2) -> (B,1,...,1,2) depending on target dimensionality.
		target_ndim = like.ndim
		out = out.reshape((out.shape[0],) + (1,) * (target_ndim - 2) + (2,))

	return out


def _safe_divisor(x: Array, eps: float) -> Array:
	if _is_torch(x):
		return torch.where(x >= 0, torch.clamp(x, min=eps), torch.clamp(x, max=-eps))
	return np.where(x >= 0, np.maximum(x, eps), np.minimum(x, -eps))


def make_unproject_stereographic(f: Array, c: Array) -> Callable[[Array], Array]:
	"""
	Create an unprojector for a stereographic destination camera.

	Input uv in pixels is first normalized by intrinsics:
	  q = (uv - c) / f

	Then inverted with unit-sphere stereographic inverse:
	  x = 4*qx / (||q||^2 + 4)
	  y = 4*qy / (||q||^2 + 4)
	  z = (4 - ||q||^2) / (||q||^2 + 4)
	"""

	def unproject_uv(uv_px: Array) -> Array:
		intr_f = _backend_intrinsics(f, uv_px)
		intr_c = _backend_intrinsics(c, uv_px)

		q = (uv_px - intr_c) / intr_f
		q2 = (q * q).sum(axis=-1, keepdims=True)

		denom = q2 + 4.0
		x = 4.0 * q[..., 0:1] / denom
		y = 4.0 * q[..., 1:2] / denom
		z = (4.0 - q2) / denom

		if _is_torch(uv_px):
			rays = torch.cat([x, y, z], dim=-1)
			nrm = torch.clamp(torch.linalg.norm(rays, dim=-1, keepdim=True), min=torch.finfo(rays.dtype).eps)
			return rays / nrm

		rays = np.concatenate([x, y, z], axis=-1)
		nrm = np.linalg.norm(rays, axis=-1, keepdims=True)
		nrm = np.maximum(nrm, np.finfo(rays.dtype).eps)
		return rays / nrm

	return unproject_uv


def make_project_stereographic(f: Array, c: Array) -> Callable[..., Union[Array, Tuple[Array, Array]]]:
	"""
	Create a stereographic projector from rays to pixel coordinates.

	For a ray r = (x,y,z) with norm ||r||, stereographic normalized coordinate is:
	  q = 2 * (x,y) / (||r|| + z)
	"""

	def project_ray(
		rays: Array,
		return_valid_mask: bool = False,
		invalid_policy: str = "nan",
	) -> Union[Array, Tuple[Array, Array]]:
		if rays.shape[-1] != 3:
			raise ValueError("rays must have last dimension 3")

		lib = torch if _is_torch(rays) else np
		eps = torch.finfo(rays.dtype).eps if _is_torch(rays) else np.finfo(rays.dtype).eps

		intr_f = _backend_intrinsics(f, rays[..., :2])
		intr_c = _backend_intrinsics(c, rays[..., :2])

		x = rays[..., 0:1]
		y = rays[..., 1:2]
		z = rays[..., 2:3]
		r = torch.linalg.norm(rays, dim=-1, keepdim=True) if _is_torch(rays) else np.linalg.norm(rays, axis=-1, keepdims=True)
		denom = r + z
		denom_safe = _safe_divisor(denom, eps)

		qx = 2.0 * x / denom_safe
		qy = 2.0 * y / denom_safe

		if _is_torch(rays):
			q = torch.cat([qx, qy], dim=-1)
			uv = q * intr_f + intr_c
			finite = torch.isfinite(uv).all(dim=-1) & torch.isfinite(r).all(dim=-1)
			valid = finite & (denom.abs()[..., 0] > eps)
			if invalid_policy == "nan":
				uv = torch.where(valid[..., None], uv, torch.full_like(uv, float("nan")))
			elif invalid_policy == "clamp":
				uv = torch.where(valid[..., None], uv, uv.clamp(-1e9, 1e9))
			else:
				raise ValueError("invalid_policy must be 'nan' or 'clamp'")
		else:
			q = np.concatenate([qx, qy], axis=-1)
			uv = q * intr_f + intr_c
			finite = np.isfinite(uv).all(axis=-1) & np.isfinite(r).all(axis=-1)
			valid = finite & (np.abs(denom)[..., 0] > eps)
			if invalid_policy == "nan":
				uv = np.where(valid[..., None], uv, np.nan)
			elif invalid_policy == "clamp":
				uv = np.where(valid[..., None], uv, np.clip(uv, -1e9, 1e9))
			else:
				raise ValueError("invalid_policy must be 'nan' or 'clamp'")

		if return_valid_mask:
			return uv, valid
		return uv

	return project_ray


class ToStereographicCamera:
	def __init__(self, src_cam, focal_length, principal_point, image_size) -> None:
		self.unproj_dst = make_unproject_stereographic(focal_length, principal_point)
		self.proj_src = make_project_from_src_cam(src_cam)
		self.proj_dst = make_project_stereographic(focal_length, principal_point)

		self.image_W = image_size[0]
		self.image_H = image_size[1]
		self.src_cam = src_cam

		self.f_dst = focal_length
		self.c_dst = principal_point

	def __call__(self, src_img, R=None) -> Any:
		H_out, W_out = self.image_H, self.image_W

		out = warp_by_calibration_batched(
			src=src_img,
			dst_size=(H_out, W_out),
			unproject_dst=self.unproj_dst,
			project_src=self.proj_src,
			interpolation="bilinear",
			rotation_R=R,
			fill_value=0.0,
			return_valid_mask=False,
		)
		return out

	def transform_keypoints_2d(
		self,
		keypoints: Union[np.ndarray, torch.Tensor],
		R: Optional[Union[np.ndarray, torch.Tensor]] = None,
		return_valid_mask: bool = False,
		invalid_policy: str = "nan",
	):
		"""
		Map 2D points from source camera pixels to destination (stereographic) pixels.
		"""
		squeeze_batch = False
		if keypoints.ndim == 2:
			B = 1
			kps = keypoints[None, ...]
			squeeze_batch = True
		elif keypoints.ndim == 3:
			B = keypoints.shape[0]
			kps = keypoints
		else:
			raise ValueError("keypoints must be (N,2) or (B,N,2)")

		if isinstance(kps, torch.Tensor):
			f_src = self.src_cam.f
			c_src = self.src_cam.c
			if not isinstance(f_src, torch.Tensor):
				f_src = torch.tensor(f_src, dtype=kps.dtype, device=kps.device)
			if not isinstance(c_src, torch.Tensor):
				c_src = torch.tensor(c_src, dtype=kps.dtype, device=kps.device)

			if f_src.ndim == 1:
				f_src = f_src.view(1, 1, 2).expand(B, 1, 2)
			else:
				f_src = f_src.view(B, 1, 2)
			if c_src.ndim == 1:
				c_src = c_src.view(1, 1, 2).expand(B, 1, 2)
			else:
				c_src = c_src.view(B, 1, 2)

			kpsf = kps.float()
		else:
			f_src = np.asarray(self.src_cam.f, dtype=np.float32)
			c_src = np.asarray(self.src_cam.c, dtype=np.float32)

			if f_src.ndim == 1:
				f_src = f_src.reshape(1, 1, 2).repeat(B, axis=0)
			else:
				f_src = f_src.reshape(B, 1, 2)
			if c_src.ndim == 1:
				c_src = c_src.reshape(1, 1, 2).repeat(B, axis=0)
			else:
				c_src = c_src.reshape(B, 1, 2)

			kpsf = kps.astype(np.float32, copy=False)

		q = (kpsf - c_src) / f_src
		q = self.src_cam.distortion_model.inverse_evaluate(q)
		rays_src = self.src_cam.unproject(q)

		if rays_src.ndim == 2:
			if isinstance(rays_src, torch.Tensor):
				rays_src = rays_src[None, ...].expand(B, -1, -1)
			else:
				rays_src = np.repeat(rays_src[None, ...], B, axis=0)

		if R is not None:
			if isinstance(rays_src, torch.Tensor):
				Rt = R if isinstance(R, torch.Tensor) else torch.tensor(R, dtype=rays_src.dtype, device=rays_src.device)
				if Rt.ndim == 2:
					Rt = Rt.expand(B, -1, -1)
				rays_dst = torch.matmul(rays_src, Rt)
			else:
				Rt = np.asarray(R, dtype=rays_src.dtype)
				if Rt.ndim == 2:
					Rt = np.broadcast_to(Rt, (B, 3, 3))
				rays_dst = np.matmul(rays_src, Rt)
		else:
			rays_dst = rays_src

		kp_dst, valid = self.proj_dst(
			rays_dst,
			return_valid_mask=True,
			invalid_policy=invalid_policy,
		)

		if squeeze_batch:
			kp_dst = kp_dst[0]
			valid = valid[0]

		if return_valid_mask:
			return kp_dst, valid
		return kp_dst
