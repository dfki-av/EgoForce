import numpy as np
import torch
from typing import Any, Callable, Optional, Tuple, Union, Literal

Array = Union[np.ndarray, torch.Tensor]
Interp = Literal["bilinear", "nearest"]

# ----------------------------- #
# Helpers: framework unification
# ----------------------------- #

def _is_torch(x: Array) -> bool:
    return isinstance(x, torch.Tensor)

def _lib(x: Array):
    return torch if _is_torch(x) else np

def _as_float(x: Array) -> Array:
    if _is_torch(x):
        return x.float()
    return x.astype(np.float32, copy=False)

def _to_bhwc(img: Array):
    """
    Normalize layout to (B,H,W,C). Returns (bhwc, invert_fn).
    Accepts: (H,W), (H,W,C), (C,H,W), (B,H,W), (B,H,W,C), (B,C,H,W)
    """
    torch_mode = _is_torch(img)

    def torch_tensor(x): return x if torch_mode else None  # just to silence linters

    if img.ndim == 2:
        H, W = img.shape
        x = img[..., None]       # (H,W,1)
        x = x[None, ...]         # (1,H,W,1)
        def inv(y):
            y = y[0]             # (H,W,1)
            return y[..., 0]     # (H,W)
        return x, inv

    if img.ndim == 3:
        a, b, c = img.shape
        # Heuristic: treat as CHW if first dim is small (1/3/4) and last dim not small
        if a in (1, 3, 4) and c not in (1, 2, 3, 4):
            # (C,H,W) -> (1,H,W,C)
            if _is_torch(img):
                x = img.permute(1, 2, 0).contiguous()[None, ...]
            else:
                x = np.transpose(img, (1, 2, 0))[None, ...]
            def inv(y):
                y = y[0]
                return y.permute(2, 0, 1).contiguous() if _is_torch(y) else np.transpose(y, (2, 0, 1))
            return x, inv
        else:
            # assume HWC -> (1,H,W,C)
            x = img[None, ...]
            def inv(y):
                return y[0]
            return x, inv

    if img.ndim == 4:
        B, A, H, W = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
        # If last dim small -> already BHWC
        if img.shape[-1] in (1, 2, 3, 4):
            x = img  # (B,H,W,C)
            def inv(y): return y
            return x, inv
        # Else assume BCHW
        if _is_torch(img):
            x = img.permute(0, 2, 3, 1).contiguous()  # (B,H,W,C)
        else:
            x = np.transpose(img, (0, 2, 3, 1))
        def inv(y):
            return y.permute(0, 3, 1, 2).contiguous() if _is_torch(y) else np.transpose(y, (0, 3, 1, 2))
        return x, inv

    raise ValueError("Unsupported image shape; expected 2D/3D/4D tensor/ndarray.")

def _meshgrid_xy_batched(B: int, H: int, W: int, like: Array):
    if _is_torch(like):
        device = like.device
        xs = torch.arange(W, dtype=torch.float32, device=device)
        ys = torch.arange(H, dtype=torch.float32, device=device)
        X = xs[None, None, :].expand(B, H, W)
        Y = ys[None, :, None].expand(B, H, W)
        return X, Y
    else:
        xs = np.arange(W, dtype=np.float32)
        ys = np.arange(H, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys)  # (H,W)
        X = np.broadcast_to(X, (B, H, W)).copy()
        Y = np.broadcast_to(Y, (B, H, W)).copy()
        return X, Y

def _zeros_like_bhwc(B: int, H: int, W: int, C: int, like: Array, fill_value=0) -> Array:
    if _is_torch(like):
        dtype = like.dtype if like.dtype.is_floating_point else torch.float32
        return torch.full((B, H, W, C), fill_value, dtype=dtype, device=like.device)
    else:
        dtype = like.dtype if np.issubdtype(like.dtype, np.floating) else np.float32
        return np.full((B, H, W, C), fill_value, dtype=dtype)

# ----------------------------- #
# Batched samplers
# ----------------------------- #

def _sample_nearest_batched(src_bhwc: Array, x: Array, y: Array, valid: Array, fill_value=0) -> Array:
    B, Hs, Ws, C = src_bhwc.shape
    lib = _lib(src_bhwc)

    if _is_torch(x):
        xi = (x + 0.5).floor().to(torch.int64).clamp(0, Ws - 1)
        yi = (y + 0.5).floor().to(torch.int64).clamp(0, Hs - 1)
        b = torch.arange(B, device=src_bhwc.device).view(B, 1, 1).expand_as(xi)
        out = _zeros_like_bhwc(B, valid.shape[1], valid.shape[2], C, like=src_bhwc, fill_value=fill_value)
        gathered = src_bhwc[b, yi, xi]  # (B,Hout,Wout,C)
        out[valid] = gathered[valid]
        # cast back if integers
        if src_bhwc.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            out = out.round().to(src_bhwc.dtype)
        return out
    else:
        xi = np.floor(x + 0.5).astype(np.int64)
        yi = np.floor(y + 0.5).astype(np.int64)
        xi = np.clip(xi, 0, Ws - 1)
        yi = np.clip(yi, 0, Hs - 1)
        b = np.arange(B).reshape(B, 1, 1)
        out = _zeros_like_bhwc(B, valid.shape[1], valid.shape[2], C, like=src_bhwc, fill_value=fill_value)
        gathered = src_bhwc[b, yi, xi]
        out[valid] = gathered[valid]
        if src_bhwc.dtype.kind in "iu":
            out = np.rint(out).astype(src_bhwc.dtype, copy=False)
        return out

def _sample_bilinear_batched(src_bhwc: Array, x: Array, y: Array, valid: Array, fill_value=0) -> Array:
    B, Hs, Ws, C = src_bhwc.shape
    lib = _lib(src_bhwc)

    if _is_torch(x):
        x0 = torch.floor(x); y0 = torch.floor(y)
        x1 = x0 + 1.0;       y1 = y0 + 1.0

        x0i = x0.clamp(0, Ws - 1).to(torch.int64)
        y0i = y0.clamp(0, Hs - 1).to(torch.int64)
        x1i = x1.clamp(0, Ws - 1).to(torch.int64)
        y1i = y1.clamp(0, Hs - 1).to(torch.int64)

        b = torch.arange(B, device=src_bhwc.device).view(B, 1, 1).expand_as(x0i)

        Ia = src_bhwc[b, y0i, x0i]
        Ib = src_bhwc[b, y0i, x1i]
        Ic = src_bhwc[b, y1i, x0i]
        Id = src_bhwc[b, y1i, x1i]

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        out = wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id

        # constant border via valid mask
        fv = torch.tensor(fill_value, dtype=out.dtype, device=out.device)
        out = torch.where(valid[..., None], out, fv)

        # cast back if integer source
        if src_bhwc.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            out = out.clamp(0, 255).round().to(src_bhwc.dtype)
        return out

    else:
        x0 = np.floor(x); y0 = np.floor(y)
        x1 = x0 + 1.0;    y1 = y0 + 1.0

        x0i = np.clip(x0, 0, Ws - 1).astype(np.int64)
        y0i = np.clip(y0, 0, Hs - 1).astype(np.int64)
        x1i = np.clip(x1, 0, Ws - 1).astype(np.int64)
        y1i = np.clip(y1, 0, Hs - 1).astype(np.int64)

        b = np.arange(B).reshape(B, 1, 1)

        Ia = src_bhwc[b, y0i, x0i]
        Ib = src_bhwc[b, y0i, x1i]
        Ic = src_bhwc[b, y1i, x0i]
        Id = src_bhwc[b, y1i, x1i]

        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)

        out = wa[..., None] * Ia + wb[..., None] * Ib + wc[..., None] * Ic + wd[..., None] * Id

        out = np.where(valid[..., None], out, fill_value).astype(out.dtype, copy=False)
        if src_bhwc.dtype.kind in "iu":
            out = np.clip(out, 0, 255).round().astype(src_bhwc.dtype, copy=False)
        return out

# ---------------------------------------------- #
# Core: batched distortByCalibration (+ rotation)
# ---------------------------------------------- #

def warp_by_calibration_batched(
    src: Array,
    dst_size: Tuple[int, int],
    unproject_dst: Callable[[Array], Array],
    project_src: Callable[[Array], Array],
    interpolation: Interp = "bilinear",
    rotation_R: Optional[Array] = None,  # (3,3) or (B,3,3)
    fill_value: float = 0.0,
    return_valid_mask: bool = False,
) -> Union[Array, Tuple[Array, Array]]:
    """
    Batched equivalent of Meta's distortByCalibration (+ rotation variant).

    src            : (H,W)[,C], (C,H,W), (B,H,W)[,C], (B,C,H,W), (B,H,W,C)  | np or torch
    dst_size       : (H_out, W_out)
    unproject_dst  : (B,H_out,W_out,2) -> (B,H_out,W_out,3)  rays in dst cam
    project_src    : (B,H_out,W_out,3) -> (B,H_out,W_out,2)  source pixel coords
    rotation_R     : None, (3,3), or (B,3,3)
    interpolation  : "bilinear" (default) or "nearest"
    fill_value     : scalar border value
    return_valid_mask : if True, also returns (B,H_out,W_out) bool

    Returns       : warped image with original layout; optional valid mask.
    """
    src_bhwc, invert_layout = _to_bhwc(src)
    B, Hs, Ws, C = src_bhwc.shape
    H_out, W_out = dst_size

    # 1) dst grid
    X, Y = _meshgrid_xy_batched(B, H_out, W_out, like=src_bhwc)

    # 2) unproject in dst camera to 3D rays
    uv_dst = (_lib(src_bhwc).stack([X, Y], axis=-1)
              if _is_torch(src_bhwc) else np.stack([X, Y], axis=-1))  # (B,H_out,W_out,2)
    rays = unproject_dst(uv_dst)                                     # (B,H_out,W_out,3)

    # 3) optional 3D rotation
    if rotation_R is not None:
        R = rotation_R
        if _is_torch(src_bhwc):
            if not isinstance(R, torch.Tensor):
                R = torch.tensor(R, dtype=rays.dtype, device=rays.device)
            if R.ndim == 2:
                R = R.expand(B, -1, -1)
            rays = torch.matmul(rays, R.transpose(-1, -2))  # (...,3)
        else:
            R = np.asarray(R, dtype=rays.dtype)
            if R.ndim == 2:
                R = np.broadcast_to(R, (B, 3, 3))
            rays = np.matmul(rays, np.swapaxes(R, -1, -2))

    # 4) project through source camera to source pixel coords
    uv_src = project_src(rays)  # (B,H_out,W_out,2)

    # 5) validity (margin=0.5, matching C++ inBounds(...,0.5f))
    margin = 0.5
    u = _as_float(uv_src[..., 0])
    v = _as_float(uv_src[..., 1])
    if _is_torch(src_bhwc):
        valid = (u >= margin) & (u <= (Ws - 1 - margin)) & (v >= margin) & (v <= (Hs - 1 - margin))
    else:
        valid = (u >= margin) & (u <= (Ws - 1 - margin)) & (v >= margin) & (v <= (Hs - 1 - margin))

    # 6) sample
    srcf = _as_float(src_bhwc)
    if interpolation == "nearest":
        dst_bhwc = _sample_nearest_batched(srcf, u, v, valid, fill_value=fill_value)
    elif interpolation == "bilinear":
        dst_bhwc = _sample_bilinear_batched(srcf, u, v, valid, fill_value=fill_value)
    else:
        raise ValueError("interpolation must be 'bilinear' or 'nearest'")

    # 7) cast back to original dtype if it was floating
    if _is_torch(src) and src.dtype.is_floating_point:
        dst_bhwc = dst_bhwc.to(src.dtype)
    if not _is_torch(src) and np.issubdtype(src.dtype, np.floating):
        dst_bhwc = dst_bhwc.astype(src.dtype, copy=False)

    out = invert_layout(dst_bhwc)
    if return_valid_mask:
        return out, valid
    return out

# ------------------------------------------------------------------ #
# Dest = PINHOLE: unproject helper (supports batch)
# ------------------------------------------------------------------ #

def make_unproject_pinhole(f: Array, c: Array) -> Callable[[Array], Array]:
    """
    Create unprojector for a pinhole dest camera.
    f : (...,2)  [fx, fy] (shared or per-batch)
    c : (...,2)  [cx, cy] (shared or per-batch)

    Returns: uv_px (B,H,W,2) -> rays (B,H,W,3) with unit length.
    """
    def unproject_uv(uv_px: Array) -> Array:
        lib = _lib(uv_px)
        # make f,c arrays/broadcastable on the same backend
        if _is_torch(uv_px):
            _f = f if isinstance(f, torch.Tensor) else torch.tensor(f, dtype=uv_px.dtype, device=uv_px.device)
            _c = c if isinstance(c, torch.Tensor) else torch.tensor(c, dtype=uv_px.dtype, device=uv_px.device)
        else:
            _f = np.asarray(f, dtype=uv_px.dtype)
            _c = np.asarray(c, dtype=uv_px.dtype)

        q = (uv_px - _c) / _f                        # (B,H,W,2)
        qx, qy = q[..., 0:1], q[..., 1:2]
        ones = (torch.ones_like(qx) if _is_torch(uv_px) else np.ones_like(qx))
        rays = _lib(uv_px).concat([qx, qy, ones], dim=-1) if _is_torch(uv_px) else np.concatenate([qx, qy, ones], axis=-1)
        # Normalize (safer for arbitrary projectors)
        if _is_torch(uv_px):
            nrm = torch.clamp(torch.linalg.norm(rays, dim=-1, keepdim=True), min=torch.finfo(rays.dtype).eps)
            return rays / nrm
        else:
            nrm = np.linalg.norm(rays, axis=-1, keepdims=True)
            nrm = np.maximum(nrm, np.finfo(rays.dtype).eps)
            return rays / nrm
    return unproject_uv

# ------------------------------------------------------------------ #
# Source = ANY CAMERA: projector helper
# ------------------------------------------------------------------ #

def make_project_from_src_cam(src_cam) -> Callable[[Array], Array]:
    """
    Wrap src_cam.camera_to_uv to accept (B,H,W,3) tensors/ndarrays.
    """
    def project_ray(rays: Array) -> Array:
        # Most camera implementations broadcast; if not, flatten & restore:
        shape = rays.shape
        if shape[-1] != 3:
            raise ValueError("rays must have last dimension 3")
        flat = rays.reshape(-1, 3)
        uv = src_cam.camera_to_uv(flat)  # expects (...,3)->(...,2)
        uv = uv.reshape(*shape[:-1], 2)
        return uv
    return project_ray


class ToPinholeCamera:
    def __init__(self, src_cam, focal_length, principal_point, image_size) -> None:
        self.unproj_dst = make_unproject_pinhole(focal_length, principal_point)    # pinhole unprojector
        self.proj_src   = make_project_from_src_cam(src_cam)      # generic projector

        self.image_W = image_size[0]
        self.image_H = image_size[1]
        self.src_cam = src_cam

        self.f_dst = focal_length
        self.c_dst = principal_point

    def __call__(self, src_img, R=None) -> Any:
        H_out, W_out = self.image_H, self.image_W

        out = warp_by_calibration_batched(
            src=src_img,                           # NumPy or Torch (CPU/GPU)
            dst_size=(H_out, W_out),
            unproject_dst=self.unproj_dst,              # pinhole
            project_src=self.proj_src,                  # anything with camera_to_uv
            interpolation="bilinear",              # or "nearest" for labels
            rotation_R=R,                  # optional
            fill_value=0.0,
            return_valid_mask=False,
        )

        return out

    def transform_keypoints_2d(
        self,
        keypoints: Union[np.ndarray, torch.Tensor],   # (N,2) or (B,N,2)
        R: Optional[Union[np.ndarray, torch.Tensor]] = None,
        return_valid_mask: bool = False,
        invalid_policy: str = "nan",                  # "nan" or "clamp"
    ):
        """
        Map 2D points from SOURCE camera pixels to DEST (pinhole) pixels.

        Pipeline per point:
          1) q_src = inverse_distort( (uv_src - c_src) / f_src )
          2) ray_src = src_cam.unproject(q_src)                  # 3D unit ray in src frame
          3) ray_dst = ray_src @ R        if R is given
                        (IMPORTANT: Use the SAME R as for image warp, which maps dst→src.
                         Here we need src→dst, and with row-vector convention ray_dst = ray_src @ R.)
          4) uv_dst = pinhole_project(ray_dst; f_dst, c_dst)

        Args:
          keypoints       : (N,2) or (B,N,2), np or torch
          R               : None, (3,3) or (B,3,3). Use the SAME R you pass to image warping.
                            (We multiply row-vectors by R, which converts src→dst when the image warp used dst→src.)
          return_valid_mask: whether to also return (B,N) or (N,) boolean mask (z>0 & finite).
          invalid_policy  : "nan" (fill invalid with NaN) or "clamp" (large finite fallback).

        Returns:
          kp_dst          : (N,2) or (B,N,2) in the same backend/dtype as input.
          (optional) mask : validity mask.
        """
        lib = torch if isinstance(keypoints, torch.Tensor) else np

        # ---- normalize shapes to (B,N,2) ----
        squeeze_batch = False
        if keypoints.ndim == 2:
            B = 1
            N = keypoints.shape[0]
            kps = keypoints[None, ...]   # (1,N,2)
            squeeze_batch = True
        elif keypoints.ndim == 3:
            B, N, _ = keypoints.shape
            kps = keypoints
        else:
            raise ValueError("keypoints must be (N,2) or (B,N,2)")

        # ---- move src intrinsics to correct backend & shape ----
        # src f,c may be (2,) or (B,2). Broadcast to (B,1,2).
        if isinstance(keypoints, torch.Tensor):
            f_src = self.src_cam.f
            c_src = self.src_cam.c
            if not isinstance(f_src, torch.Tensor): f_src = torch.tensor(f_src, dtype=keypoints.dtype, device=keypoints.device)
            if not isinstance(c_src, torch.Tensor): c_src = torch.tensor(c_src, dtype=keypoints.dtype, device=keypoints.device)
            if f_src.ndim == 1: f_src = f_src.view(1, 1, 2).expand(B, 1, 2)
            else:               f_src = f_src.view(B, 1, 2)
            if c_src.ndim == 1: c_src = c_src.view(1, 1, 2).expand(B, 1, 2)
            else:               c_src = c_src.view(B, 1, 2)
        else:
            f_src = np.asarray(self.src_cam.f, dtype=keypoints.dtype if hasattr(keypoints, "dtype") else np.float32)
            c_src = np.asarray(self.src_cam.c, dtype=keypoints.dtype if hasattr(keypoints, "dtype") else np.float32)
            if f_src.ndim == 1: f_src = f_src.reshape(1, 1, 2).repeat(B, axis=0)
            else:               f_src = f_src.reshape(B, 1, 2)
            if c_src.ndim == 1: c_src = c_src.reshape(1, 1, 2).repeat(B, axis=0)
            else:               c_src = c_src.reshape(B, 1, 2)

        # ---- normalize dtype ----
        kpsf = kps.float() if isinstance(kps, torch.Tensor) else kps.astype(np.float32, copy=False)

        # 1) undistort & normalize (source)
        q = (kpsf - c_src) / f_src                        # (B,N,2)
        q = self.src_cam.distortion_model.inverse_evaluate(q)

        # 2) source rays
        #    src_cam.unproject expects (...,2) -> (...,3)
        rays_src = self.src_cam.unproject(q)              # (B,N,3) or (N,3) depending on impl
        # ensure shape (B,N,3)
        if rays_src.ndim == 2:
            rays_src = rays_src[None, ...].repeat(B, 1, 1)

        # 3) rotate rays from src->dst if R is provided
        if R is not None:
            if isinstance(keypoints, torch.Tensor):
                Rt = R
                if not isinstance(Rt, torch.Tensor):
                    Rt = torch.tensor(Rt, dtype=rays_src.dtype, device=rays_src.device)
                if Rt.ndim == 2:
                    Rt = Rt.expand(B, -1, -1)
                # row-vector convention: rays_dst = rays_src @ R
                rays_dst = torch.matmul(rays_src, Rt)
            else:
                Rt = np.asarray(R, dtype=rays_src.dtype)
                if Rt.ndim == 2:
                    Rt = np.broadcast_to(Rt, (B, 3, 3))
                rays_dst = np.matmul(rays_src, Rt)
        else:
            rays_dst = rays_src

        # 4) pinhole projection to dest intrinsics
        # dst f,c can be (2,) or (B,2) → (B,1,2)
        if isinstance(keypoints, torch.Tensor):
            f_dst = self.f_dst if isinstance(self.f_dst, torch.Tensor) else torch.tensor(self.f_dst, dtype=rays_dst.dtype, device=rays_dst.device)
            c_dst = self.c_dst if isinstance(self.c_dst, torch.Tensor) else torch.tensor(self.c_dst, dtype=rays_dst.dtype, device=rays_dst.device)
            if f_dst.ndim == 1: f_dst = f_dst.view(1, 1, 2).expand(B, 1, 2)
            else:               f_dst = f_dst.view(B, 1, 2)
            if c_dst.ndim == 1: c_dst = c_dst.view(1, 1, 2).expand(B, 1, 2)
            else:               c_dst = c_dst.view(B, 1, 2)

            x = rays_dst[..., 0:1]
            y = rays_dst[..., 1:1+1]
            z = rays_dst[..., 2:3]
            eps = torch.finfo(rays_dst.dtype).eps
            z_safe = torch.where(z >= 0, torch.clamp(z, min=eps), torch.clamp(z, max=-eps))
            u = f_dst[..., 0:1] * (x / z_safe) + c_dst[..., 0:1]
            v = f_dst[..., 1:2] * (y / z_safe) + c_dst[..., 1:2]
            kp_dst = torch.cat([u, v], dim=-1)

            valid = torch.isfinite(kp_dst).all(dim=-1) & (z.abs() > eps)[..., 0]
            if invalid_policy == "nan":
                kp_dst = torch.where(valid[..., None], kp_dst, torch.full_like(kp_dst, float('nan')))
            elif invalid_policy == "clamp":
                # clamp very large values
                kp_dst = torch.where(valid[..., None], kp_dst, kp_dst.clamp(-1e9, 1e9))
            else:
                raise ValueError("invalid_policy must be 'nan' or 'clamp'")

        else:
            f_dst = np.asarray(self.f_dst, dtype=rays_dst.dtype)
            c_dst = np.asarray(self.c_dst, dtype=rays_dst.dtype)
            if f_dst.ndim == 1: f_dst = f_dst.reshape(1, 1, 2).repeat(B, axis=0)
            else:               f_dst = f_dst.reshape(B, 1, 2)
            if c_dst.ndim == 1: c_dst = c_dst.reshape(1, 1, 2).repeat(B, axis=0)
            else:               c_dst = c_dst.reshape(B, 1, 2)

            x = rays_dst[..., 0:1]
            y = rays_dst[..., 1:2]
            z = rays_dst[..., 2:3]
            eps = np.finfo(rays_dst.dtype).eps
            z_safe = np.where(z >= 0, np.maximum(z, eps), np.minimum(z, -eps))
            u = f_dst[..., 0:1] * (x / z_safe) + c_dst[..., 0:1]
            v = f_dst[..., 1:2] * (y / z_safe) + c_dst[..., 1:2]
            kp_dst = np.concatenate([u, v], axis=-1)

            valid = np.isfinite(kp_dst).all(axis=-1) & (np.abs(z)[..., 0] > eps)
            if invalid_policy == "nan":
                kp_dst = np.where(valid[..., None], kp_dst, np.nan)
            elif invalid_policy == "clamp":
                kp_dst = np.where(valid[..., None], kp_dst, np.clip(kp_dst, -1e9, 1e9))
            else:
                raise ValueError("invalid_policy must be 'nan' or 'clamp'")

        # ---- restore original batch shape ----
        if squeeze_batch:
            kp_dst = kp_dst[0]
            valid = valid[0]

        return (kp_dst, valid) if return_valid_mask else kp_dst
