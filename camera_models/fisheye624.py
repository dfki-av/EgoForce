import numpy as np
import torch
import math
from typing import Sequence
from .projections import ArctanProjection

class OVR624Distortion:
    """
    OVRFisheye624 model, with 6 radial, 2 tangential coeffs and 4 coeffs to model thin-prism.
    """
    def stack(self, p, axis=0):
        assert len(p) > 0
        return torch.stack(p, dim=axis) if isinstance(p[0], torch.Tensor) else np.stack(p, axis=axis)

    def __init__(self, params, *args, **kwargs): 
        assert len(params) == 12, "Distortion parameters must be a 12D vector"           

        self.is_distorted = torch.count_nonzero(params) if isinstance(params, torch.Tensor) else np.count_nonzero(params)

        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6, self.p1, self.p2, self.s1, self.s2, self.s3, self.s4 = params
        
    def evaluate(self, p):
        k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
        p1, p2 = self.p1, self.p2
        s1, s2, s3, s4 = self.s1, self.s2, self.s3, self.s4

        lib = torch if isinstance(p, torch.Tensor) else np            

        # radial component
        r2 = (p * p).sum(axis=-1, keepdims=True)
        r2 = lib.clip(r2, -np.pi**2, np.pi**2)

        r4 = r2 * r2
        r6 = r2 * r4
        r8 = r4 * r4
        r10 = r4 * r6
        r12 = r6 * r6
        radial = 1 + k1 * r2 + k2 * r4 + k3 * r6 + k4 * r8 + k5 * r10 + k6 * r12
        uv = p * radial

        # tangential component
        x, y = uv[..., 0], uv[..., 1]
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

        return self.stack((x, y), axis=-1)

    def inverse_evaluate(self, uv, tol=1e-6, max_iter=15):
        # Unpack distortion coefficients
        k1, k2, k3, k4, k5, k6 = self.k1, self.k2, self.k3, self.k4, self.k5, self.k6
        p1, p2 = self.p1, self.p2
        s1, s2, s3, s4 = self.s1, self.s2, self.s3, self.s4

        lib = torch if isinstance(uv, torch.Tensor) else np
        
        # Initialize with distorted points as an initial guess
        p = uv.clone() if lib is torch else uv.copy()

        for idx in range(max_iter):
            # radial component
            x, y = p[..., 0], p[..., 1]
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
            error1 = F1 - uv[..., 0]
            error2 = F2 - uv[..., 1]
            # Stack the error components.
            error = lib.stack([error1, error2], axis=-1)

            # Compute the Jacobian matrix J = dF/dp.
            # For the x-coordinate:
            dF1_dx = dg_dx * radial + g * dradial_dx
            dF1_dy = dg_dy * radial + g * dradial_dy
            # For the y-coordinate:
            dF2_dx = dh_dx * radial + h * dradial_dx
            dF2_dy = dh_dy * radial + h * dradial_dy

            # Assemble the Jacobian (last two dimensions: 2x2).
            J = lib.stack([lib.stack([dF1_dx, dF1_dy], axis=-1),
                        lib.stack([dF2_dx, dF2_dy], axis=-1)], axis=-2)

            # Solve for the Newton update: delta such that J * delta = -error.
            if lib is torch:
                delta = -torch.linalg.solve(J, error.unsqueeze(-1)).squeeze(-1)
            else:
                # np.linalg.solve is applied over the last two dims in a vectorized manner.
                delta = -np.linalg.solve(J, error[..., None])[..., 0]

            # Update the current guess.
            p = p + delta

            # Optionally, handle pathological cases.
            if lib is torch:
                err_norm = torch.linalg.norm(delta, dim=-1)
                invalid_mask = (torch.isnan(delta).any(dim=-1)) | (torch.isinf(delta).any(dim=-1)) | (err_norm > 1e8)
                if invalid_mask.any():
                    p[invalid_mask] = uv[invalid_mask]  # reset invalid points
                    delta[invalid_mask] = 0
                if (err_norm < tol).all():
                    break
            else:
                err_norm = np.linalg.norm(delta, axis=-1)
                invalid_mask = (np.isnan(delta).any(axis=-1)) | (np.isinf(delta).any(axis=-1)) | (err_norm > 1e8)
                if np.any(invalid_mask):
                    p[invalid_mask] = uv[invalid_mask]
                    delta[invalid_mask] = 0
                if np.all(err_norm < tol):
                    break
        
        return p


class OVR624CameraModel(ArctanProjection):
    model_fov_limit = math.pi / 2
    TYPE = 'fisheye624'
    TYPE_ID = 3

    def __init__(self, f, c, params: Sequence[float], width: int, height: int):
        assert len(f) == 2, "Focal length must be a 2D vector (fx, fy)"
        assert len(c) == 2, "Principal point must be a 2D vector (cx, cy)"
        assert len(params) == 12, "Distortion parameters must be a 12D vector"

        f = f.float() if isinstance(f, torch.Tensor) else np.array(f, dtype=np.float32)
        c = c.float() if isinstance(c, torch.Tensor) else np.array(c, dtype=np.float32)
        params = params.float() if isinstance(params, torch.Tensor) else np.array(params, dtype=np.float32)

        self.f, self.c = f, c
        self.distortion_model = OVR624Distortion(params, width, height)
        self.params = params
        self.width = width
        self.height = height

    def camera_to_uv(self, v):
        """Project eye coordinates to 2d window coordinates"""
        p = self.project(v)
        
        q = self.distortion_model.evaluate(p)
        return q * self.f + self.c

    def camera_to_uvd(self, v):
        """Project eye coordinates to 3d window coordinates (uv + depth)"""
        p = self.project3(v)

        q = self.distortion_model.evaluate(p[..., :2])
        
        uv = q * self.f + self.c
        d = p[..., 2:]        
           
        return self.concatenate((uv, d), axis=1)
    
    def uv_to_theta_x_y(self, p, return_undistorted=False):
        lib = torch if isinstance(p, torch.Tensor) else np            
        p = p.clone() if isinstance(p, torch.Tensor) else p.copy()
        q = (p[..., :2] - self.c) / self.f
        q = self.distortion_model.inverse_evaluate(q)

        u, v = q[..., 0], q[..., 1]

        uv = self.stack((u, v), axis=-1)

        theta = torch.atan(uv) if isinstance(uv, torch.Tensor) else np.arctan(uv)

        if return_undistorted:
            return theta, (self.f * uv + self.c)

        return theta 

    def uvd_to_camera(self, p):
        """Unproject 3d window coordinates to eye coordinates"""
        p = p.clone() if isinstance(p, torch.Tensor) else p.copy()

        q = (p[..., :2] - self.c) / self.f
        q = self.distortion_model.inverse_evaluate(q)
        p[..., :2] = q
        return self.unproject3(p)

    def camera_to_d(self, p):
        assert p.shape[-1] == 3
        lib = torch if isinstance(p, torch.Tensor) else np            

        z = p[..., 2]
        r3 = self.norm(p, axis=-1)        
        return r3 * lib.sign(z)

    def uvz_to_d(self, uvz):
        assert uvz.shape[-1] == 3
        lib = torch if isinstance(uvz, torch.Tensor) else np

        u = uvz[..., 0]
        v = uvz[..., 1]
        z = uvz[..., 2]

        q = lib.stack([(u - self.c[0]) / self.f[0],
                    (v - self.c[1]) / self.f[1]], axis=-1)
        q = self.distortion_model.inverse_evaluate(q)

        # 2) Range from z and direction:  r = |z| * sqrt(1 + qx^2 + qy^2)
        s2 = 1.0 + (q * q).sum(axis=-1)
        r = lib.sqrt(s2) * lib.abs(z)

        # 3) Signed range (match camera_to_d sign convention)
        return r * lib.sign(z)

    def to(self, device):
        if isinstance(self.f, np.ndarray):
            self.f = torch.from_numpy(self.f).to(device)
            self.c = torch.from_numpy(self.c).to(device)
            self.params = torch.from_numpy(self.params).to(device)
            self.distortion_model = OVR624Distortion(self.params, self.width, self.height)

        elif isinstance(self.f, torch.Tensor):
            self.f = self.f.to(device)
            self.c = self.c.to(device)
            self.params = self.params.to(device)

            self.distortion_model = OVR624Distortion(self.params, self.width, self.height)

        return self
    
    def clone(self):
        if isinstance(self.f, torch.Tensor):
            return OVR624CameraModel(self.f.clone(), self.c.clone(), self.params.clone(), self.width, self.height)
        else:
            return OVR624CameraModel(self.f.copy(), self.c.copy(), self.params.copy(), self.width, self.height)
    
    def to_intrinsics_keypoint_encoding(self, keypoints, return_undistorted):
        return self.uv_to_theta_x_y(keypoints, return_undistorted)

    def get_K(self):
        fx, fy = self.f[0], self.f[1]
        cx, cy = self.c[0], self.c[1]

        return np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]], dtype=np.float32)
    
    def update_K(self, K, width=None, height=None):
        self.f[0] = K[0, 0]
        self.f[1] = K[1, 1]
        self.c[0] = K[0, 2]
        self.c[1] = K[1, 2]

        if width is not None and height is not None:
            self.width = width
            self.height = height

        return self

    def distort3d(self, v):
        p = self.project3(v)
        
        q_d = self.distortion_model.evaluate(p[..., :2])
            
        p_d = self.concatenate([q_d, p[..., -1:]], axis=-1)
        
        v_d = self.unproject3(p_d)

        return v_d

    def camera_to_uvz(self, v):
        z = v [..., 2:]
        p = self.project3(v)
        q = self.distortion_model.evaluate(p[..., :2])
        uv = q * self.f + self.c
        return self.concatenate((uv, z), axis=1)

    def uvz_to_camera(self, p, invalid_policy="nan", return_valid_mask=False):
        """
        p: (...,3) with [u_px, v_px, z_cam]
        returns:
          v: (...,3) back-projected XYZ
          valid_mask: (...,) boolean
        """
        lib = torch if isinstance(p, torch.Tensor) else np
        p = p.clone() if lib is torch else p.copy()

        # dtype/eps
        if lib is torch:
            eps = torch.finfo(p.dtype).eps
        else:
            eps = np.finfo(p.dtype if hasattr(p, "dtype") else np.float32).eps

        # 1) undistort & normalize pixels to arctan-plane coords q
        q = (p[..., :2] - self.c) / self.f                    # (...,2)
        q = self.distortion_model.inverse_evaluate(q)         # (...,2)

        # 2) unit ray from arctan projection
        vhat = self.unproject(q)                              # (...,3), ||vhat||=1
        vz   = vhat[..., 2:3]                                 # (...,1)
        z    = p[..., 2:3]                                    # (...,1)

        # Numerical validity checks
        if lib is np:
            finite_in = np.isfinite(p).all(axis=-1)
            finite_q  = np.isfinite(q).all(axis=-1)
            finite_vh = np.isfinite(vhat).all(axis=-1)
            near_grazing = (np.abs(vz)[..., 0] < 1e-9) & (np.abs(z)[..., 0] > 0)
            ambiguous    = (np.abs(z)[..., 0] == 0)
            valid_mask   = finite_in & finite_q & finite_vh & (~near_grazing) & (~ambiguous)
        else:
            finite_in = torch.isfinite(p).all(dim=-1)
            finite_q  = torch.isfinite(q).all(dim=-1)
            finite_vh = torch.isfinite(vhat).all(dim=-1)
            near_grazing = (vz.abs()[..., 0] < 1e-9) & (z.abs()[..., 0] > 0)
            ambiguous    = (z.abs()[..., 0] == 0)
            valid_mask   = finite_in & finite_q & finite_vh & (~near_grazing) & (~ambiguous)

        # 3) scale by d = z / vhat_z (safe division)
        if lib is torch:
            vz_safe = torch.where(vz >= 0, torch.clamp(vz, min=eps), torch.clamp(vz, max=-eps))
            d = z / vz_safe
            v = vhat * d
            if invalid_policy == "nan":
                nan_fill = torch.full_like(v, float('nan'))
                v = torch.where(valid_mask[..., None], v, nan_fill)
            elif invalid_policy == "clamp":
                d_clamped = torch.clamp(d, min=-1e6, max=1e6)
                v_clamped = vhat * d_clamped
                v = torch.where(valid_mask[..., None], v, v_clamped)
            else:
                raise ValueError("invalid_policy must be 'nan' or 'clamp'")
        else:
            vz_safe = np.where(vz >= 0, np.maximum(vz, eps), np.minimum(vz, -eps))
            d = z / vz_safe
            v = vhat * d
            if invalid_policy == "nan":
                v[~valid_mask] = np.nan
            elif invalid_policy == "clamp":
                d = np.clip(d, -1e6, 1e6)
                v_clamped = vhat * d
                v[~valid_mask] = v_clamped[~valid_mask]
            else:
                raise ValueError("invalid_policy must be 'nan' or 'clamp'")

        if not return_valid_mask:
            return v    

        return v, valid_mask
