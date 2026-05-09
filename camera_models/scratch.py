
class Fisheye624Numpy:
    def __init__(self, params: Sequence[float], width: int, height: int):
        f, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4 = params

        focal_lengths = [f, f]
        principal_point = [cx, cy]
        radial_params = [k1, k2, k3, k4, k5, k6]
        tangential_params = [p1, p2]
        thin_prism_params = [s1, s2, s3, s4]

        self.num_k = len(radial_params)
        self.use_tangential = True
        self.use_thin_prism = True
        self.use_single_focal_length = True

        self.focal_lengths = np.array(focal_lengths)
        self.principal_point = np.array(principal_point)
        self.radial_params = np.array(radial_params)
        self.tangential_params = np.array(tangential_params)
        self.thin_prism_params = np.array(thin_prism_params)
        
        self.width = width
        self.height = height

    def project_point(self, point_optical: np.ndarray) -> np.ndarray:
        # Ensure that the input types and dimensions are correct
        assert point_optical.shape == (3,)
        
        # Compute [a; b] = [x/z; y/z]
        inv_z = 1.0 / point_optical[2]
        ab = point_optical[:2] * inv_z

        # Compute the squares of the elements of ab
        ab_squared = ab ** 2
        r_sq = ab_squared[0] + ab_squared[1]
        r = np.sqrt(r_sq)
        th = np.arctan(r)
        theta_sq = th ** 2

        # Compute the theta polynomial
        th_radial = 1.0
        theta2is = theta_sq
        for i in range(self.num_k):
            th_radial += theta2is * self.radial_params[i]
            theta2is *= theta_sq

        # Compute th/r, using the limit for small values
        th_div_r = 1.0 if r < np.finfo(float).eps else th / r

        # Compute the distorted coordinates -- except for focal length and principal point
        xr_yr = (th_radial * th_div_r) * ab
        xr_yr_squared_norm = np.sum(xr_yr ** 2)

        # Start computing the output
        uv_distorted = xr_yr

        if self.use_tangential:
            temp = 2.0 * np.dot(xr_yr, self.tangential_params)
            uv_distorted += temp * xr_yr + xr_yr_squared_norm * self.tangential_params

        if self.use_thin_prism:
            radial_powers_2_and_4 = np.array([xr_yr_squared_norm, xr_yr_squared_norm ** 2])
            uv_distorted[0] += np.dot(self.thin_prism_params[:2], radial_powers_2_and_4)
            uv_distorted[1] += np.dot(self.thin_prism_params[2:], radial_powers_2_and_4)

        # Compute the return value
        if self.use_single_focal_length:
            pts = self.focal_lengths[0] * uv_distorted + self.principal_point
        else:
            pts = uv_distorted * self.focal_lengths + self.principal_point

        valid = (pts[0] >= 0) & (pts[0] < self.width) & (pts[1] >= 0) & (pts[1] < self.height)
        return pts if valid else np.array([np.nan, np.nan])

    def project_batch(self, points_optical: np.ndarray) -> np.ndarray:
        # Ensure that the input types and dimensions are correct
        assert points_optical.ndim == 2 and points_optical.shape[1] == 3
        
        # Compute [a; b] = [x/z; y/z] for all points
        inv_z = 1.0 / points_optical[:, 2]
        ab = points_optical[:, :2] * inv_z[:, np.newaxis]

        # Compute the squares of the elements of ab
        ab_squared = ab ** 2
        r_sq = np.sum(ab_squared, axis=1)
        r = np.sqrt(r_sq)
        th = np.arctan(r)
        theta_sq = th ** 2
    
        # Compute the theta polynomial
        th_radial = np.ones_like(r)
        theta2is = theta_sq
        for i in range(self.num_k):
            th_radial += theta2is * self.radial_params[i]
            theta2is *= theta_sq

        # Compute th/r, using the limit for small values
        th_div_r = np.where(r < np.finfo(float).eps, 1.0, th / r)
    
        # Compute the distorted coordinates -- except for focal length and principal point
        xr_yr = (th_radial[:, np.newaxis] * th_div_r[:, np.newaxis]) * ab
        xr_yr_squared_norm = np.sum(xr_yr ** 2, axis=1)

        # Start computing the output
        uv_distorted = xr_yr

        if self.use_tangential:
            temp = 2.0 * np.dot(xr_yr, self.tangential_params)
            uv_distorted += (temp[:, np.newaxis] * xr_yr) + (xr_yr_squared_norm[:, np.newaxis] * self.tangential_params)

        if self.use_thin_prism:
            radial_powers_2_and_4 = np.column_stack([xr_yr_squared_norm, xr_yr_squared_norm ** 2])
            uv_distorted[:, 0] += np.dot(radial_powers_2_and_4, self.thin_prism_params[:2])
            uv_distorted[:, 1] += np.dot(radial_powers_2_and_4, self.thin_prism_params[2:])

        # Compute the return value
        if self.use_single_focal_length:
            return self.focal_lengths[0] * uv_distorted + self.principal_point
        else:
            return uv_distorted * self.focal_lengths + self.principal_point


class Fisheye624Torch:
    """
    OVRFisheye624 model, with 6 radial, 2 tangential coeffs, and 4 coeffs to model thin-prism.
    """
    def __init__(self, params: Sequence[float], width: int, height: int):
        f, cx, cy, k1, k2, k3, k4, k5, k6, p1, p2, s1, s2, s3, s4 = params

        focal_lengths = [f, f]
        principal_point = [cx, cy]
        radial_params = [k1, k2, k3, k4, k5, k6]
        tangential_params = [p1, p2]
        thin_prism_params = [s1, s2, s3, s4]

        self.num_k = len(radial_params)
        self.use_tangential = True
        self.use_thin_prism = True
        self.use_single_focal_length = True

        self.focal_lengths = torch.tensor(focal_lengths, dtype=torch.float64)
        self.principal_point = torch.tensor(principal_point, dtype=torch.float64)
        self.radial_params = torch.tensor(radial_params, dtype=torch.float64)
        self.tangential_params = torch.tensor(tangential_params, dtype=torch.float64)
        self.thin_prism_params = torch.tensor(thin_prism_params, dtype=torch.float64)

        self.width = width
        self.height = height

    def project_point(self, point_optical: torch.Tensor) -> torch.Tensor:
        # Ensure that the input types and dimensions are correct
        assert point_optical.shape == (3,)
        point_optical = point_optical
        dtype = point_optical.dtype
        device = point_optical.device

        # Compute [a; b] = [x/z; y/z]
        inv_z = 1.0 / point_optical[2]
        ab = point_optical[:2] * inv_z

        # Compute the squares of the elements of ab
        ab_squared = ab ** 2
        r_sq = ab_squared[0] + ab_squared[1]
        r = torch.sqrt(r_sq)
        th = torch.atan(r)
        theta_sq = th ** 2

        one_tensor = torch.tensor(1, dtype=dtype, device=device)

        # Compute the theta polynomial
        th_radial = one_tensor
        theta2is = theta_sq
        for i in range(self.num_k):
            th_radial += theta2is * self.radial_params[i]
            theta2is *= theta_sq

        # Compute th/r, using the limit for small values
        th_div_r = torch.where(r < np.finfo(float).eps, one_tensor, th / r)

        # Compute the distorted coordinates -- except for focal length and principal point
        xr_yr = (th_radial * th_div_r) * ab
        xr_yr_squared_norm = torch.sum(xr_yr ** 2)

        # Start computing the output
        uv_distorted = xr_yr

        if self.use_tangential:
            temp = 2.0 * torch.dot(xr_yr, self.tangential_params)
            uv_distorted += temp * xr_yr + xr_yr_squared_norm * self.tangential_params

        if self.use_thin_prism:
            radial_powers_2_and_4 = torch.tensor([xr_yr_squared_norm, xr_yr_squared_norm ** 2], dtype=dtype, device=device)
            uv_distorted[0] += torch.dot(self.thin_prism_params[:2], radial_powers_2_and_4)
            uv_distorted[1] += torch.dot(self.thin_prism_params[2:], radial_powers_2_and_4)

        # Compute the return value
        if self.use_single_focal_length:
            pts = self.focal_lengths[0] * uv_distorted + self.principal_point
        else:
            pts = uv_distorted * self.focal_lengths + self.principal_point

        valid = (pts[0] >= 0) & (pts[0] < self.width) & (pts[1] >= 0) & (pts[1] < self.height)
        return pts if valid else torch.tensor([float('nan'), float('nan')], dtype=dtype, device=device)

    def project_batch(self, points_optical: torch.Tensor) -> torch.Tensor:
        # Ensure that the input types and dimensions are correct
        assert points_optical.ndim == 2 and points_optical.shape[1] == 3
        points_optical = points_optical
        dtype = points_optical.dtype


        # Compute [a; b] = [x/z; y/z] for all points
        inv_z = 1.0 / points_optical[:, 2]
        ab = points_optical[:, :2] * inv_z[:, None]

        # Compute the squares of the elements of ab
        ab_squared = ab ** 2
        r_sq = torch.sum(ab_squared, dim=1)
        r = torch.sqrt(r_sq)
        th = torch.atan(r)
        theta_sq = th ** 2

        # Compute the theta polynomial
        th_radial = torch.ones_like(r)
        theta2is = theta_sq
        for i in range(self.num_k):
            th_radial += theta2is * self.radial_params[i]
            theta2is *= theta_sq

        # Compute th/r, using the limit for small values
        th_div_r = torch.where(r < torch.finfo(torch.float32).eps, torch.tensor(1.0, dtype=dtype), th / r)

        # Compute the distorted coordinates -- except for focal length and principal point
        xr_yr = (th_radial[:, None] * th_div_r[:, None]) * ab
        xr_yr_squared_norm = torch.sum(xr_yr ** 2, dim=1)

        # Start computing the output
        uv_distorted = xr_yr

        if self.use_tangential:
            temp = 2.0 * torch.matmul(xr_yr, self.tangential_params)
            uv_distorted += temp[:, None] * xr_yr + xr_yr_squared_norm[:, None] * self.tangential_params

        if self.use_thin_prism:
            radial_powers_2_and_4 = torch.stack([xr_yr_squared_norm, xr_yr_squared_norm ** 2], dim=1)
            uv_distorted[:, 0] += torch.matmul(radial_powers_2_and_4, self.thin_prism_params[:2])
            uv_distorted[:, 1] += torch.matmul(radial_powers_2_and_4, self.thin_prism_params[2:])

        # Compute the return value
        if self.use_single_focal_length:
            return self.focal_lengths[0] * uv_distorted + self.principal_point
        else:
            return uv_distorted * self.focal_lengths + self.principal_point

