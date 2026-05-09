import torch
import math
import numpy as np


class KannalaBrandtK3Distortion:
    """
    4-parameter 'KannalaBrandt' distortion model:
      dist_coeffs = [k1, k2, k3, k4]
    """
    _EPS_R     = 1e-8       # to avoid division by zero in r
    _EPS_DET   = 1e-12      # threshold for singular Jacobian
    _MAX_DELTA = 1e8        # threshold for large updates

    def __init__(self, dist_coeffs):
        assert len(dist_coeffs) == 4, "dist_coeffs must have 4 elements [k1, k2, k3, k4]"
        self.is_distorted = torch.count_nonzero(dist_coeffs) if isinstance(dist_coeffs, torch.Tensor) else np.count_nonzero(dist_coeffs)

        self.k1, self.k2, self.k3, self.k4 = dist_coeffs

    def evaluate(self, xy):
        """
        Forward distortion: xy -> xy_distorted
        xy: (..., 2)
        """
        lib = torch if isinstance(xy, torch.Tensor) else np

        r = lib.linalg.norm(xy, axis=-1)
        theta = lib.arctan(r)

        theta_d = theta * (
                1
                + self.k1 * (theta ** 2) # theta^2
                + self.k2 * (theta ** 4) # theta^4
                + self.k3 * (theta ** 6) # theta^6
                + self.k4 * (theta ** 8) # theta^8
        )

        if lib is np:
            scale = np.ones_like(r)
            np.divide(theta_d, r, out=scale, where=r > self._EPS_R)
        else:
            scale = torch.where(r > self._EPS_R, theta_d / r, torch.ones_like(r))

        xy_dist = xy * scale[..., None]

        return xy_dist

    def inverse_evaluate_fp_solve(self, uv, tol=1e-7, max_iter=100):
        """
        Inverse distortion: uv_dist -> uv_undist
        Iterative approach.

        uv: (..., 2)
        returns: same shape, undistorted xy
        """
        lib = torch if isinstance(uv, torch.Tensor) else np

        # Initialize guess = distorted coords
        xy = uv.clone() if lib is torch else uv.copy()

        for i in range(max_iter):
            # Forward-distort the current guess
            xy_dist = self.evaluate(xy)
            # Error
            error = uv - xy_dist
            error_norm = lib.linalg.norm(error, axis=-1) 

            invalid_mask = lib.isnan(xy_dist).any(axis=-1) | lib.isinf(xy_dist).any(axis=-1)
            large_mask = error_norm > 1e8  # threshold
            invalid_mask = invalid_mask | large_mask

            if lib.any(invalid_mask):
                xy[invalid_mask] = uv[invalid_mask] # reset invalid points
                error[invalid_mask] = 0  # reset error for invalid points

            # Update
            xy = xy + error

            # Check for convergence
            if lib.all(error_norm < tol):
                break

        return xy

    def inverse_evaluate(self, uv, tol=1e-7, max_iter=10, fp_fallback=True):
        """
        Inverse distortion using Newton's method:
        Solve F(x, y) = uv by finding (x, y) such that G(x,y) = F(x,y) - uv = 0.

        Here F(x, y) is the forward Kannala-Brandt distortion:
        F(x, y) = (x, y) * (theta_d / r),
        where:
            r = sqrt(x^2 + y^2)
            theta = arctan(r)
            theta_d = theta * (1 + k1*theta^2 + k2*theta^4 + k3*theta^6 + k4*theta^8)

        The Newton update:
            (x, y) <- (x, y) - J^{-1}(F(x,y)-uv)

        Args:
            uv: (..., 2) distorted coordinates.
            tol: Tolerance threshold.
            max_iter: Maximum iterations.

        Returns:
            Undistorted (x, y) coordinates with the same shape as uv.
        """
        lib = torch if isinstance(uv, torch.Tensor) else np
        xy = uv.clone() if lib is torch else uv.copy()

        for _ in range(max_iter):
            x, y = xy[..., 0], xy[..., 1]
            r = lib.linalg.norm(xy, axis=-1)
            theta = lib.arctan(r)
            theta2 = theta**2
            theta4 = theta2**2
            theta6 = theta2 * theta4
            theta8 = theta4**2

            theta_d = theta * (1 + self.k1*theta2 + self.k2*theta4 + self.k3*theta6 + self.k4*theta8)

            if lib is np:
                scale = np.ones_like(r)
                np.divide(theta_d, r, out=scale, where=r > self._EPS_R)
            else:
                scale = torch.where(r > self._EPS_R, theta_d / r, torch.ones_like(r))

            F_xy = xy * scale[..., None]

            error = F_xy - uv
            if lib.max(lib.linalg.norm(error, axis=-1)) < tol:
                break

            dtheta_dr = 1 / (1 + r**2)
            dtheta_d_dtheta = (1 + 3*self.k1*theta2 + 5*self.k2*theta4 + 7*self.k3*theta6 + 9*self.k4*theta8)
            dscale_dr = (dtheta_d_dtheta * dtheta_dr * r - theta_d) / (r**2 + self._EPS_R)

            J11 = scale + x**2 * dscale_dr / (r + self._EPS_R)
            J22 = scale + y**2 * dscale_dr / (r + self._EPS_R)
            J12 = x * y * dscale_dr / (r + self._EPS_R)
            J21 = J12

            det = J11 * J22 - J12 * J12
            singular = lib.abs(det) < self._EPS_DET
            good = ~singular

            delta = lib.zeros_like(xy)
            if lib.any(good):
                delta_x = (error[...,0] * J22 - error[...,1] * J12)[good] / det[good]
                delta_y = (error[...,1] * J11 - error[...,0] * J12)[good] / det[good]
                update = lib.stack([delta_x, delta_y], axis=-1)
                delta[good] = update

            err_norm = lib.linalg.norm(delta, axis=-1)
            bad = lib.isnan(delta).any(axis=-1) | lib.isinf(delta).any(axis=-1) | (err_norm > self._MAX_DELTA)
            singular |= bad

            if fp_fallback and lib.any(singular):
                xy_sing = self.inverse_evaluate_fp_solve(uv[singular], tol=tol, max_iter=25)
                xy[singular] = xy_sing
                delta = lib.where(singular[..., None], lib.zeros_like(delta), delta)

            xy = xy - delta

            if lib.all(err_norm < tol):
                break

        return xy


class KannalaBrandtK3CameraModel:
    TYPE_ID = 4
    
    """
    A pinhole + KannalaBrandtK3 distortion camera model, parallel to Fisheye624 style.
    The main difference is that we do not do an 'arctan fisheye' step, but standard
    pinhole + KannalaBrandt polynomial lens.

    f: (fx, fy)
    c: (cx, cy)
    dist_coeffs: 4-length array [k1, k2, k3, k4]
    width, height: image dimensions
    """

    def __init__(self, f, c, dist_coeffs, width, height):
        assert len(f) == 2, "Focal length must be a 2D vector (fx, fy)"
        assert len(c) == 2, "Principal point must be a 2D vector (cx, cy)"
        assert len(dist_coeffs) == 4, "Distortion parameters must be a 4D vector"

        # Convert to tensor or array
        if isinstance(f, torch.Tensor):
            self.f = f.float()
            self.c = c.float()
            self.params = dist_coeffs.float()
        else:
            self.f = np.array(f, dtype=np.float32)
            self.c = np.array(c, dtype=np.float32)
            self.params = np.array(dist_coeffs, dtype=np.float32)

        self.width = width
        self.height = height

        # Distortion model
        self.distortion_model = KannalaBrandtK3Distortion(self.params)

    def camera_to_uv(self, v):
        """
        3D camera coords -> 2D pixel coords (distorted).
        v: (..., 3)
        """
        lib = torch if isinstance(v, torch.Tensor) else np
        X, Y, Z = v[..., 0], v[..., 1], v[..., 2]
        # normalized pinhole
        x = X / Z
        y = Y / Z

        # Distort in normalized coords
        xy = lib.stack([x, y], axis=-1)
        xy_dist = self.distortion_model.evaluate(xy)

        # scale + shift
        fx, fy = self.f[0], self.f[1]
        cx, cy = self.c[0], self.c[1]

        u = fx * xy_dist[..., 0] + cx
        v = fy * xy_dist[..., 1] + cy

        return lib.stack([u, v], axis=-1)

    def camera_to_uvd(self, v):
        """
        3D camera coords -> (u,v,d).
        For 'd', we often store r3 * sign(z) or just z. 
        Let's choose to store Z for simplicity, or 
        do the Fisheye624 style (Eucl norm * sign(Z)).
        We'll match OVR624 style: d = ||p|| * sign(z).
        
        v: (..., 3)
        returns: (..., 3) -> (u, v, d)
        """
        lib = torch if isinstance(v, torch.Tensor) else np

        # 1) project to (u, v)
        uv = self.camera_to_uv(v)

        # 2) compute d
        # match the OVR624 approach: 
        #   d = norm_3d(p) * sign(Z)
        r3 = lib.linalg.norm(v, axis=-1)  # Eucl norm
        signz = lib.sign(v[..., 2])
        d = r3 * signz

        return lib.concatenate([uv, d[..., None]], axis=-1)

    def uvd_to_camera(self, p):
        """
        Invert (u,v,d) -> 3D camera coords.
        p: (..., 3) => (u, v, d)
        
        We'll un-distort (u,v), then scale by 'd' or reconstruct.
        If d = norm_3d(p)*sign(Z), we can find ||p|| = abs(d).
        Then we place it along the direction of the un-distorted ray.
        """
        lib = torch if isinstance(p, torch.Tensor) else np
        u, v, d = p[..., 0], p[..., 1], p[..., 2]

        # 1) Subtract principal point, divide by focal length => normalized coords
        fx, fy = self.f[0], self.f[1]
        cx, cy = self.c[0], self.c[1]

        x_dist = (u - cx) / fx
        y_dist = (v - cy) / fy
        xy_dist = lib.stack([x_dist, y_dist], axis=-1)

        # 2) Inverse distortion => (x_undist, y_undist)
        xy_undist = self.distortion_model.inverse_evaluate(xy_dist)

        # 3) Convert to a direction in 3D. In a standard pinhole, 
        #    direction is (x_undist, y_undist, 1) up to scale.
        #    We'll then scale so that the norm_3d is abs(d).
        x_u = xy_undist[..., 0]
        y_u = xy_undist[..., 1]

        # direction
        dir_3d = lib.stack([x_u, y_u, lib.ones_like(x_u)], axis=-1)
        norm_dir = lib.linalg.norm(dir_3d, axis=-1)

        # We'll scale direction so the total norm is abs(d).
        # sign(d) tells us if Z was + or - in original:
        # OVR624 style used sign(Z). We do the same:
        sign_z = lib.sign(d)  # to reflect if original Z < 0
        mag = lib.abs(d)
        # scale factor to get from norm_dir to mag
        scale = mag / lib.maximum(norm_dir, 1e-12)

        # final 3D
        pt_cam = dir_3d * scale[..., None]

        # Now if sign_z < 0, it means we want the final Z to be negative 
        # => multiply the 3D's Z by sign_z:
        # (Alternatively you can incorporate sign_z into scale, but let's do direct.)
        pt_cam[..., 2] *= sign_z

        return pt_cam

    def camera_to_d(self, p):
        """
        Return the 'd' used in camera_to_uvd. 
        We do d = ||p|| * sign(Z).
        p: (..., 3)
        """
        lib = torch if isinstance(p, torch.Tensor) else np
        r3 = lib.linalg.norm(p, axis=-1)
        s = lib.sign(p[..., 2])
        return r3 * s

    def uv_to_theta_x_y(self, uv, return_undistorted=False):
        # 1) Determine if we're using Torch or NumPy
        lib = torch if isinstance(uv, torch.Tensor) else np

        # 2) Extract focal lengths and principal point
        fx, fy = self.f[0], self.f[1]
        cx, cy = self.c[0], self.c[1]

        # 3) Subtract principal point, divide by focal length => normalized coords
        #    x = (u - cx) / fx
        #    y = (v - cy) / fy
        x_dist = (uv[..., 0] - cx) / fx
        y_dist = (uv[..., 1] - cy) / fy

        xy_dist = torch.stack([x_dist, y_dist], dim=-1) if isinstance(uv, torch.Tensor) else np.stack([x_dist, y_dist], axis=-1)
        
        xy_undist = self.distortion_model.inverse_evaluate(xy_dist)

        # 4) Take arctan for each dimension
        #    θx = arctan(x)
        #    θy = arctan(y)
        theta = torch.atan(xy_undist) if isinstance(uv, torch.Tensor) else np.arctan(xy_undist)

        if return_undistorted:
            return theta, (self.f * xy_undist + self.c)

        return theta

    def to(self, device):
        if isinstance(self.f, torch.Tensor):
            self.f = self.f.to(device)
            self.c = self.c.to(device)
            self.params = self.params.to(device)

            self.distortion_model = KannalaBrandtK3Distortion(self.params)

        return self
    
    def clone(self):
        if isinstance(self.f, torch.Tensor):
            return KannalaBrandtK3CameraModel(self.f.clone(), self.c.clone(), self.params.clone(), self.width, self.height)
        else:
            return KannalaBrandtK3CameraModel(self.f.copy(), self.c.copy(), self.params.copy(), self.width, self.height)
    
    def to_intrinsics_keypoint_encoding(self, keypoints, return_undistorted=False):
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
        lib = torch if isinstance(v, torch.Tensor) else np
        X, Y, Z = v[..., 0], v[..., 1], v[..., 2]

        # normalized pinhole
        x = X / Z
        y = Y / Z

        # Distort in normalized coords
        xy = lib.stack([x, y], axis=-1)
        xy_dist = self.distortion_model.evaluate(xy)

        # back to 3D
        Xd = xy_dist[..., 0] * Z
        Yd = xy_dist[..., 1] * Z

        return lib.stack([Xd, Yd, Z], axis=-1)

    def camera_to_uvz(self, p):
        lib = torch if isinstance(p, torch.Tensor) else np

        # 1) project to (u, v)
        uv = self.camera_to_uv(p)

        # 2) extract z
        z = p[..., 2]

        return lib.concatenate([uv, z[..., None]], axis=-1)

    def uvz_to_camera(self, p):
        lib = torch if isinstance(p, torch.Tensor) else np
        uv = p[..., :2]
        z = p[..., 2]
        
        fx, fy = self.f[0], self.f[1] 
        cx, cy = self.c[0], self.c[1]
        
        x_dist = (uv[..., 0] - cx) / fx
        y_dist = (uv[..., 1] - cy) / fy
        
        xy_dist = lib.stack([x_dist, y_dist], axis=-1)

        xy_undist = self.distortion_model.inverse_evaluate(xy_dist)

        x = xy_undist[..., 0] * z
        y = xy_undist[..., 1] * z
        
        return lib.stack([x, y, z], axis=-1)
