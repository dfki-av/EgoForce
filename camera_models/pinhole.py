import numpy as np
import torch


class IdentityDistortionModel:
    def __init__(self, *args, **kwargs):
        self.is_distorted = False

    def evaluate(self, p, *args, **kwargs):
        return p
    
    def inverse_evaluate(self, uv, *args, **kwargs):
        return uv


class OpenCVDistortion:
    def __init__(self, params):
        """
        Initializes the distortion model with 5 OpenCV distortion coefficients.
        
        Parameters:
        - params: Distortion coefficients [k1, k2, p1, p2, k3].
        """
        assert len(params) == 5, "Distortion parameters must be a 5D vector"
        
        self.k1, self.k2, self.p1, self.p2, self.k3 = params

    def apply_distortion(self, p):
        """
        Applies radial and tangential distortion to 2D normalized points.
        
        Parameters:
        - p: 2D normalized points (Nx2) before distortion.

        Returns:
        - Distorted 2D points (Nx2).
        """
        assert p.shape[-1] == 2, "Input must be 2D normalized points (Nx2)"
        
        lib = torch if isinstance(p, torch.Tensor) else np
        
        # Extract x and y coordinates
        x, y = p[..., 0], p[..., 1]
        
        # Compute radial distance squared
        r2 = x * x + y * y
        r4 = r2 * r2
        r6 = r2 * r4

        # Apply radial distortion
        radial = 1 + self.k1 * r2 + self.k2 * r4 + self.k3 * r6

        # Apply tangential distortion
        x_tangential = 2 * self.p1 * x * y + self.p2 * (r2 + 2 * x * x)
        y_tangential = self.p1 * (r2 + 2 * y * y) + 2 * self.p2 * x * y

        # Apply the distortion to x and y
        x_distorted = x * radial + x_tangential
        y_distorted = y * radial + y_tangential
        
        return self.stack([x_distorted, y_distorted], axis=-1)

    def undistort_points(self, uv, tol=1e-6, max_iter=100):
        """
        Undistorts 2D image points using iterative optimization.
        
        Parameters:
        - uv: 2D distorted points (Nx2).
        - tol: Convergence tolerance for iterative undistortion.
        - max_iter: Maximum number of iterations.

        Returns:
        - Undistorted 2D points (Nx2).
        """
        assert uv.shape[-1] == 2, "Input must be 2D points (Nx2)"
        
        lib = torch if isinstance(uv, torch.Tensor) else np
        
        # Initialize with distorted points as the initial guess
        p = uv.clone() if lib is torch else uv.copy()

        for _ in range(max_iter):
            # Apply the distortion model to the current guess
            distorted = self.apply_distortion(p)
            
            # Compute the error between the distorted guess and the original points
            error = uv - distorted

            # Update the estimate based on the error
            p = p + error
            
            # Check for convergence
            if lib.linalg.norm(error) < tol:
                break
                
        return p

    def stack(self, p, axis=0):
        """
        Helper function to stack arrays or tensors.
        """
        assert len(p) > 0
        return torch.stack(p, dim=axis) if isinstance(p[0], torch.Tensor) else np.stack(p, axis=axis)



class PinholeCameraModelWithOpenCVDistortion:
    def __init__(self, f, c, params, width, height):
        self.f, self.c = f, c
        self.distortion_model = OpenCVDistortion(params, width, height)
        self.width = width
        self.height = height

    def norm(self, x, ord=None, axis=None, keepdims=False):
        """
        Compute the Euclidean norm of the 3D points, used for depth calculation.
        """
        return torch.linalg.norm(x, ord=ord, dim=axis, keepdims=keepdims) if isinstance(x, torch.Tensor) else np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def camera_to_d(self, p):
        """
        Calculate the depth (d) based on the 3D Euclidean distance from the camera origin,
        and return the signed depth based on the Z coordinate.
        """
        assert p.shape[-1] == 3, "Input must be 3D points (Nx3)"
        lib = torch if isinstance(p, torch.Tensor) else np

        z = p[..., 2]  # Extract Z coordinate (depth)
        r3 = self.norm(p, axis=-1)  # 3D Euclidean norm
        return r3 * lib.sign(z)  # Signed depth

    def camera_to_uvd(self, v):
        """
        Project eye coordinates (3D) to 2D window coordinates with depth (UV + D).
        """
        # Project points to normalized 2D coordinates (without distortion)
        p = v[..., :2] / v[..., 2:3]  # Perspective division

        # Apply distortion to the 2D coordinates
        q = self.distortion_model.evaluate(p)

        # Apply intrinsic camera parameters (focal length and principal point)
        uv = q * self.f + self.c

        # Compute depth using camera_to_d method
        d = self.camera_to_d(v)  # Use the same depth computation as in the fisheye model

        # Return UV + Depth
        return np.concatenate((uv, d[..., None]), axis=-1) if isinstance(v, np.ndarray) else torch.cat((uv, d.unsqueeze(-1)), dim=-1)

    def uvd_to_camera(self, p):
        """
        Unproject 3D window coordinates (UV + D) back to eye coordinates.
        """
        lib = torch if isinstance(p, torch.Tensor) else np

        # Extract the UV and depth (D) components
        uv = p[..., :2]
        d = p[..., 2]

        # Normalize UV coordinates
        q = (uv - self.c) / self.f

        # Inverse distortion model to remove distortion
        q = self.distortion_model.inverse_evaluate(q)

        # Calculate 3D coordinates (x, y, z)
        x = q[..., 0] * d
        y = q[..., 1] * d
        z = lib.sign(d) * lib.sqrt(d ** 2 - x ** 2 - y ** 2)  # Compute z from the depth

        # Return unprojected 3D points
        return self.stack([x, y, z], axis=-1)



class PinholeCameraModel:
    TYPE = "pinhole"
    TYPE_ID = 0

    def __init__(self, f, c, width: int, height: int):
        """
        Initialize a pinhole camera model with no distortion.
        - f: Focal lengths (fx, fy) as a 2D vector.
        - c: Principal points (cx, cy) as a 2D vector.
        - width: Width of the camera image in pixels.
        - height: Height of the camera image in pixels.
        """
        self.f = f.float() if isinstance(f, torch.Tensor) else np.array(f, dtype=np.float32)
        self.c = c.float() if isinstance(c, torch.Tensor) else np.array(c, dtype=np.float32)
        self.width = width
        self.height = height
        self.params = torch.empty(0) if isinstance(f, torch.Tensor) else np.empty(0)
        
        self.distortion_model = IdentityDistortionModel()

    def stack(self, p, axis=0):
        return torch.stack(p, dim=axis) if isinstance(p[0], torch.Tensor) else np.stack(p, axis=axis)

    def norm(self, x, ord=None, axis=None, keepdims=False):
        """Compute the Euclidean norm of the 3D points, used for depth calculation."""
        return torch.linalg.norm(x, ord=ord, dim=axis, keepdims=keepdims) if isinstance(x, torch.Tensor) else np.linalg.norm(x, ord=ord, axis=axis, keepdims=keepdims)

    def camera_to_d(self, p):
        """
        Calculate the depth (d) based on the 3D Euclidean distance from the camera origin,
        and return the signed depth based on the Z coordinate.
        """
        assert p.shape[-1] == 3, "Input must be 3D points (Nx3)"
        lib = torch if isinstance(p, torch.Tensor) else np

        z = p[..., 2]  # Extract Z coordinate (depth)
        r3 = self.norm(p, axis=-1)  # 3D Euclidean norm
        return r3 * lib.sign(z)  # Signed depth

    def camera_to_uv(self, v):
        """
        Project 3D camera coordinates to 2D image coordinates.
        """
        uv = v[..., :2] / v[..., 2:3]  # Perspective division (x/z, y/z)
        return uv * self.f + self.c  # Apply focal length and principal point

    def camera_to_uvd(self, v):
        """
        Project 3D camera coordinates to 2D image coordinates with depth.
        """
        uv = self.camera_to_uv(v)  # Get 2D projection
        d = self.camera_to_d(v)    # Compute depth
        return np.concatenate((uv, d[..., None]), axis=-1) if isinstance(v, np.ndarray) else torch.cat((uv, d.unsqueeze(-1)), dim=-1)

    def uvd_to_camera(self, p):
        """
        Unproject 2D image coordinates (u, v) with signed Euclidean norm (depth 'd') back to 3D camera coordinates.
        
        Parameters:
        - p: A Nx3 or (..., 3) array representing 2D image coordinates (u, v) and depth (d).
        
        Returns:
        - A Nx3 or (..., 3) array of 3D points in camera coordinates (X, Y, Z).
        """
        assert p.shape[-1] == 3, "Input must have 3 components (u, v, d)"

        lib = torch if isinstance(p, torch.Tensor) else np

        # Extract UV coordinates and depth
        uv = p[..., :2]  # (u, v)
        d = p[..., 2]    # signed Euclidean norm depth

        # Normalize the UV coordinates by the focal lengths and principal points
        q = (uv - self.c) / self.f  # Normalized (X/Z, Y/Z) coordinates

        # The goal is to compute the Z coordinate in 3D space.
        # We have the following relation:
        # d^2 = X^2 + Y^2 + Z^2
        # where:
        # X = (u - cx) / fx * Z
        # Y = (v - cy) / fy * Z
        #
        # Substituting X and Y into the equation for d^2, we get:
        # d^2 = ((u - cx) / fx * Z)^2 + ((v - cy) / fy * Z)^2 + Z^2
        #
        # Factoring Z^2, we arrive at the quadratic form:
        # d^2 = Z^2 * [(u - cx)^2 / fx^2 + (v - cy)^2 / fy^2 + 1]
        #
        # Now, solve for Z^2:
        # Z^2 = d^2 / [(u - cx)^2 / fx^2 + (v - cy)^2 / fy^2 + 1]

        # Compute the denominator of the Z equation
        # denominator = (u - cx)^2 / fx^2 + (v - cy)^2 / fy^2 + 1
        denominator = q[..., 0]**2 + q[..., 1]**2 + 1

        # Solve for Z using the quadratic solution for Z^2
        # Z = sign(d) * sqrt(d^2 / denominator)
        # The sign of Z is determined by the sign of d, ensuring proper direction
        z = lib.sign(d) * lib.sqrt(d**2 / denominator)

        # Once Z is computed, we can recover X and Y using the normalized camera coordinates (q)
        # X = (u - cx) / fx * Z
        # Y = (v - cy) / fy * Z
        x = q[..., 0] * z 
        y = q[..., 1] * z  

        return self.stack([x, y, z], axis=-1)

    def uv_to_theta_x_y(self, uv):
        """
        Compute the intrinsics-aware positional encoding angles θₓ and θᵧ for given pixel coordinates.
        
        For each pixel coordinate (x, y), the angles are computed as:
            θₓ = arctan((x - pₓ) / fₓ)
            θᵧ = arctan((y - p_y) / f_y)
        where (pₓ, p_y) is the principal point and (fₓ, f_y) are the focal lengths.
        
        Parameters:
        - uv: An array or tensor of shape (..., 2) representing the pixel coordinates (x, y).
        
        Returns:
        - An array or tensor of shape (..., 2) containing the angles (θₓ, θᵧ).
        """
        # Determine the library (torch or numpy) based on the input type.
        lib = torch if isinstance(uv, torch.Tensor) else np
        
        # Compute the difference from the principal point.
        diff = uv - self.c
        # Normalize by the focal lengths.
        normalized_diff = diff / self.f
        # Compute the angular values using the arctan function.
        theta = torch.atan(normalized_diff) if isinstance(uv, torch.Tensor) else np.arctan(normalized_diff)

        return theta

    def to(self, device):
        if isinstance(self.f, np.ndarray):
            self.f = torch.tensor(self.f, dtype=torch.float32, device=device)
            self.c = torch.tensor(self.c, dtype=torch.float32, device=device)
            self.params = torch.tensor(self.params, dtype=torch.float32, device=device)
        elif isinstance(self.f, torch.Tensor):
            self.f = self.f.to(device)
            self.c = self.c.to(device)

        return self
    
    def clone(self):
        if isinstance(self.f, torch.Tensor):
            return PinholeCameraModel(self.f.clone(), self.c.clone(), self.width, self.height)
        else:
            return PinholeCameraModel(self.f.copy(), self.c.copy(), self.width, self.height)
    
    def to_intrinsics_keypoint_encoding(self, keypoints, return_undistorted=False):
        if return_undistorted:
            return self.uv_to_theta_x_y(keypoints), keypoints

        return self.uv_to_theta_x_y(keypoints)

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

    def camera_to_uvz(self, v):
        uv = self.camera_to_uv(v)  # Get 2D projection
        z = v[..., 2]    # Z coordinate (depth)
        return np.concatenate((uv, z[..., None]), axis=-1) if isinstance(v, np.ndarray) else torch.cat((uv, z.unsqueeze(-1)), dim=-1)

    def uvz_to_camera(self, p):
        assert p.shape[-1] == 3, "Input must have 3 components (u, v, z)"

        lib = torch if isinstance(p, torch.Tensor) else np

        # Extract UV coordinates and depth
        uv = p[..., :2]  # (u, v)
        z = p[..., 2]    # Z coordinate (depth)

        # Normalize the UV coordinates by the focal lengths and principal points
        q = (uv - self.c) / self.f  # Normalized (X/Z, Y/Z) coordinates
        x = q[..., 0] * z 
        y = q[..., 1] * z   

        return self.stack([x, y, z], axis=-1)
        