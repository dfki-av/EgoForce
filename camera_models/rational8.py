import torch
import math
import numpy as np


class Rational8Distortion:
    """
    8-parameter 'rational' distortion model:
      dist_coeffs = [k1, k2, p1, p2, k3, k4, k5, k6]

    Forward distortion (x,y -> x_d,y_d):

      r^2 = x^2 + y^2
      num = (1 + k1*r^2 + k2*r^4 + k3*r^6)
      den = (1 + k4*r^2 + k5*r^4 + k6*r^6)
      r_dist = num / den

      x_d = x*r_dist + 2*p1*x*y + p2*(r^2 + 2x^2)
      y_d = y*r_dist + 2*p2*x*y + p1*(r^2 + 2y^2)

    Inverse is solved iteratively by re-applying forward model and adjusting.
    """

    def __init__(self, dist_coeffs):
        assert len(dist_coeffs) == 8, "dist_coeffs must have 8 elements [k1, k2, p1, p2, k3, k4, k5, k6]"
        self.is_distorted = torch.count_nonzero(dist_coeffs) if isinstance(dist_coeffs, torch.Tensor) else np.count_nonzero(dist_coeffs)

        self.k1, self.k2, self.p1, self.p2, self.k3, self.k4, self.k5, self.k6 = dist_coeffs

    def evaluate(self, xy):
        """
        Forward distortion: xy -> xy_distorted
        xy: (..., 2)
        """
        lib = torch if isinstance(xy, torch.Tensor) else np
        x, y = xy[..., 0], xy[..., 1]

        r2 = lib.clip(x*x + y*y, 0, 1e10)
        r4 = lib.clip(r2 * r2, 0, 1e11)
        r6 = lib.clip(r4 * r2, 0, 1e12)
        
        # Rational radial
        num = 1 + self.k1*r2 + self.k2*r4 + self.k3*r6
        den = 1 + self.k4*r2 + self.k5*r4 + self.k6*r6
        r_dist = num / den

        # Tangential
        xy_dist_x = x*r_dist + 2*self.p1*x*y + self.p2*(r2 + 2*x*x)
        xy_dist_y = y*r_dist + 2*self.p2*x*y + self.p1*(r2 + 2*y*y)

        return lib.stack([xy_dist_x, xy_dist_y], axis=-1)

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

    def inverse_evaluate(self, uv, tol=1e-7, max_iter=10):
        """
        Inverse distortion using Newton's method:
        Solve F(x, y) = uv  by finding (x, y) such that G(x,y) = F(x,y) - uv = 0.
        
        Here F(x, y) is defined as the forward distortion:
        F(x, y) = [ x * r_dist + 2*p1*x*y + p2*(r2 + 2*x*x),
                    y * r_dist + 2*p2*x*y + p1*(r2 + 2*y*y) ]
        where:
        r2 = x^2 + y^2
        r4 = r2^2, r6 = r2^3
        r_dist = (1 + k1*r2 + k2*r4 + k3*r6) / (1 + k4*r2 + k5*r4 + k6*r6)
        
        The Newton update is:
            (x, y) <- (x, y) - J^{-1}(F(x,y)-uv)
        
        Args:
        uv: (..., 2) distorted coordinates.
        tol: Tolerance threshold for the update.
        max_iter: Maximum number of iterations.
        
        Returns:
        Undistorted (x, y) coordinates with the same shape as uv.
        """
        lib = torch if isinstance(uv, torch.Tensor) else np
        # Initialize guess to be the distorted coordinates.
        xy = uv.clone() if lib is torch else uv.copy()

        for idx in range(max_iter):
            # Compute the forward distortion F(x,y)
            # --------------------------------------
            # Unpack current guess
            x, y = xy[..., 0], xy[..., 1]
            # Compute radius powers.
            r2 = lib.clip(x * x + y * y, 0, 1e10)
            r4 = lib.clip(r2 * r2, 0, 1e11)
            r6 = lib.clip(r4 * r2, 0, 1e12)
            # Radial distortion (rational model)
            num = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6
            den = 1 + self.k4 * r2 + self.k5 * r4 + self.k6 * r6
            r_dist = num / den

            # Forward-distorted coordinates: F(x,y)
            F1 = x * r_dist + 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
            F2 = y * r_dist + 2 * self.p2 * x * y + self.p1 * (r2 + 2 * y * y)
            F_xy = lib.stack([F1, F2], axis=-1)

            # Compute the error vector: G(x,y) = F(x,y) - uv
            error = F_xy - uv
            error_norm = lib.linalg.norm(error, axis=-1)
            if lib.all(error_norm < tol):
                break

            # Compute the Jacobian J = [ [dF1/dx, dF1/dy],
            #                            [dF2/dx, dF2/dy] ]
            # --------------------------------------------------
            # We first need the derivatives of r_dist with respect to x and y.
            #
            # Let:
            #   N = 1 + k1*r2 + k2*r4 + k3*r6,    D = 1 + k4*r2 + k5*r4 + k6*r6,
            #   r_dist = N / D.
            # Then by the quotient rule:
            #   ∂r_dist/∂x = (D*(∂N/∂x) - N*(∂D/∂x)) / D²,
            #
            # where:
            #   ∂N/∂x = 2*k1*x + 4*k2*r2*x + 6*k3*r4*x,
            #   ∂D/∂x = 2*k4*x + 4*k5*r2*x + 6*k6*r4*x.
            # Similarly for the y-derivative.
            #
            # Define a common term for compactness:
            #   common = 2 / D² * [ D*(k1 + 4*k2*r2 + 6*k3*r4) - N*(k4 + 4*k5*r2 + 6*k6*r4) ].
            #
            # Then:
            #   dr_dist/dx = common * x,    dr_dist/dy = common * y.
            common = (2.0 * ( (den * (self.k1 + 4 * self.k2 * r2 + 6 * self.k3 * r4)
                            - num * (self.k4 + 4 * self.k5 * r2 + 6 * self.k6 * r4) ) ))
            common = common / (den * den)
            dr_dx = common * x
            dr_dy = common * y

            # Now, write F1 and F2 as follows:
            #   F1 = x * r_dist + 2*p1*x*y + p2*(r2 + 2*x^2)
            #   F2 = y * r_dist + 2*p2*x*y + p1*(r2 + 2*y^2)
            #
            # Compute partial derivatives:
            #
            # For F1:
            #   d/dx (x * r_dist) = r_dist + x * (dr_dx)
            #   d/dx (2*p1*x*y) = 2*p1*y
            #   d/dx (p2*(r2 + 2*x^2)) = p2*(2*x + 4*x) = 6*p2*x
            # Thus, dF1/dx = r_dist + x * dr_dx + 2*p1*y + 6*p2*x.
            #
            #   d/dy (x * r_dist) = x * dr_dy
            #   d/dy (2*p1*x*y) = 2*p1*x
            #   d/dy (p2*(r2 + 2*x^2)) = p2*(2*y)
            # Thus, dF1/dy = x * dr_dy + 2*p1*x + 2*p2*y.
            #
            # For F2:
            #   d/dx (y * r_dist) = y * dr_dx
            #   d/dx (2*p2*x*y) = 2*p2*y
            #   d/dx (p1*(r2 + 2*y^2)) = p1*(2*x)
            # Thus, dF2/dx = y * dr_dx + 2*p2*y + 2*p1*x.
            #
            #   d/dy (y * r_dist) = r_dist + y * dr_dy
            #   d/dy (2*p2*x*y) = 2*p2*x
            #   d/dy (p1*(r2 + 2*y^2)) = p1*(2*y + 4*y) = 6*p1*y
            # Thus, dF2/dy = r_dist + y * dr_dy + 2*p2*x + 6*p1*y.
            J11 = r_dist + x * dr_dx + 2 * self.p1 * y + 6 * self.p2 * x
            J12 = x * dr_dy + 2 * self.p1 * x + 2 * self.p2 * y
            J21 = y * dr_dx + 2 * self.p2 * y + 2 * self.p1 * x
            J22 = r_dist + y * dr_dy + 2 * self.p2 * x + 6 * self.p1 * y

            # Solve for the Newton update:
            #    delta = J^{-1} * (F(x,y) - uv)
            # For a 2x2 matrix:
            #    J^{-1} = (1/det) * [[J22, -J12], [-J21, J11]],  where det = J11*J22 - J12*J21.
            det = J11 * J22 - J12 * J21
            # Compute delta for each coordinate.
            delta_x = (error[..., 0] * J22 - error[..., 1] * J12) / det
            delta_y = (error[..., 1] * J11 - error[..., 0] * J21) / det
            delta = lib.stack([delta_x, delta_y], axis=-1)

            # Newton update: subtract the correction term.
            xy = xy - delta

            # Optionally, handle pathological cases.
            if lib is torch:
                err_norm = torch.linalg.norm(delta, dim=-1)
                invalid_mask = (torch.isnan(delta).any(dim=-1)) | (torch.isinf(delta).any(dim=-1)) | (err_norm > 1e8)
                if invalid_mask.any():
                    xy[invalid_mask] = uv[invalid_mask]  # reset invalid points
                    delta[invalid_mask] = 0
                if (err_norm < tol).all():
                    break
            else:
                err_norm = np.linalg.norm(delta, axis=-1)
                invalid_mask = (np.isnan(delta).any(axis=-1)) | (np.isinf(delta).any(axis=-1)) | (err_norm > 1e8)
                if np.any(invalid_mask):
                    xy[invalid_mask] = uv[invalid_mask]
                    delta[invalid_mask] = 0
                if np.all(err_norm < tol):
                    break
                
        return xy


class Rational8CameraModel:
    TYPE_ID = 2
    
    """
    A pinhole + rational-8 distortion camera model, parallel to Fisheye624 style.
    The main difference is that we do not do an 'arctan fisheye' step, but standard
    pinhole + rational polynomial lens.

    f: (fx, fy)
    c: (cx, cy)
    dist_coeffs: 8-length array [k1, k2, p1, p2, k3, k4, k5, k6]
    width, height: image dimensions
    """

    def __init__(self, f, c, dist_coeffs, width, height):
        assert len(f) == 2, "Focal length must be a 2D vector (fx, fy)"
        assert len(c) == 2, "Principal point must be a 2D vector (cx, cy)"
        assert len(dist_coeffs) == 8, "Distortion parameters must be a 8D vector"

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
        self.distortion_model = Rational8Distortion(self.params)

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

            self.distortion_model = Rational8Distortion(self.params)

        return self
    
    def clone(self):
        if isinstance(self.f, torch.Tensor):
            return Rational8CameraModel(self.f.clone(), self.c.clone(), self.params.clone(), self.width, self.height)
        else:
            return Rational8CameraModel(self.f.copy(), self.c.copy(), self.params.copy(), self.width, self.height)
    
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

        X = xy_dist[..., 0] * Z
        Y = xy_dist[..., 1] * Z

        return lib.stack([X, Y, Z], axis=-1)

    def camera_to_uvz(self, v):
        lib = torch if isinstance(v, torch.Tensor) else np
        uv = self.camera_to_uv(v)
        z = v[..., 2]
        return lib.concatenate([uv, z[..., None]], axis=-1)

    def uvz_to_camera(self, p):
        assert p.shape[-1] == 3
        lib = torch if isinstance(p, torch.Tensor) else np

        uv = p[..., :2]
        z = p[..., 2]

        fx, fy = self.f[0], self.f[1]
        cx, cy = self.c[0], self.c[1]
        
        x_d = (uv[..., 0] - cx) / fx
        y_d = (uv[..., 1] - cy) / fy
        
        xy_d = lib.stack([x_d, y_d], axis=-1)
        xy_u = self.distortion_model.inverse_evaluate(xy_d)
        
        x = xy_u[..., 0] * z
        y = xy_u[..., 1] * z
        
        return lib.stack([x, y, z], axis=-1)
