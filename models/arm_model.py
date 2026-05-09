import numpy as np
import torch
from utils.rotations import axis_angle_to_matrix, matrix_to_axis_angle


def rotation_decompose_twist_swirl(R_axis):
    """
    Vectorized decomposition of a batch of axis-angle rotations into
    swirl (which aligns z-axis to the new direction) and twist (rotation
    around that new axis).

    Args:
        R_axis: (B, 3) axis-angle representation for B rotations.

    Returns:
        swirl_mat:  (B, 3, 3) rotation that sends z_rest -> R @ z_rest
        twist_mat:  (B, 3, 3) leftover rotation around that new axis
        twist_angle: (B,) twist magnitude in radians
    """

    device = R_axis.device
    B = R_axis.shape[0]
    # 1) Convert axis-angle to rotation matrix: shape (B, 3, 3)
    R = axis_angle_to_matrix(R_axis)

    # 2) We'll define our rest bone axis as z = (0, 0, 1).
    #    Expand so we can do batch matmul easily: shape (3,1)
    z_rest = torch.tensor([0.0, 0.0, 1.0], device=device).reshape(3,1)

    # 3) z_new = R @ z_rest, shape (B, 3, 1); flatten to (B, 3)
    z_new = R @ z_rest
    z_new = z_new.squeeze(-1)  # shape (B, 3)

    # Norm of z_new
    z_new_norm = z_new.norm(dim=1)  # shape (B,)

    # Prepare output buffers
    swirl_mat  = torch.zeros_like(R)  # (B, 3, 3)
    twist_mat  = torch.zeros_like(R)  # (B, 3, 3)
    twist_angle = torch.zeros(B, device=device)

    # ---- Handle degenerate case: z_new_norm < eps
    eps = 1e-8
    mask_degenerate = z_new_norm < eps
    # For degenerate rows, we can just set swirl=I, twist=R
    swirl_mat[mask_degenerate] = torch.eye(3, device=device)
    twist_mat[mask_degenerate] = R[mask_degenerate]

    # ---- Non-degenerate rows
    mask_ok = ~mask_degenerate
    if mask_ok.any():
        # 3a) Normalize z_new
        z_hat = z_new[mask_ok] / z_new_norm[mask_ok].unsqueeze(1)  # (N, 3)

        # 3b) cross(z_rest, z_hat)
        #    We treat z_rest as (1,3), so broadcast to (N,3)
        z_rest_expanded = z_rest.squeeze(-1).expand_as(z_hat)  # (N,3)
        axis_swirl = torch.cross(z_rest_expanded, z_hat, dim=1)  # (N,3)
        axis_swirl_norm = axis_swirl.norm(dim=1)  # (N,)

        # dot to detect collinearity
        dot_val = (z_rest_expanded * z_hat).sum(dim=1)  # (N,)

        # Prepare swirl as identity by default
        swirl_sub = torch.eye(3, device=device).unsqueeze(0).repeat(mask_ok.sum(),1,1) 
        # We'll fill in swirl_sub for the non-collinear rows.

        # Mask for collinearity (axis_swirl_norm < eps)
        mask_collinear = axis_swirl_norm < eps
        mask_noncollinear = ~mask_collinear

        # collinear & same direction => swirl=I
        # collinear & opposite direction => swirl=rotation 180 around e.g. x-axis
        if mask_collinear.any():
            # same direction => dot_val>0 => swirl=I
            mc_idx = torch.where(mask_collinear)[0]  # indices inside sub-batch
            dot_sub = dot_val[mask_collinear]
            same_dir  = dot_sub > 0
            oppo_dir  = dot_sub < 0
            # swirl=I for same_dir
            # swirl=diag(1, -1, -1) for oppo_dir
            swirl_sub[mc_idx[same_dir]] = torch.eye(3, device=device)
            swirl_sub[mc_idx[oppo_dir]] = torch.diag(torch.tensor([1,-1,-1], device=device))

        # non-collinear => axis_swirl_unit, angle_swirl
        if mask_noncollinear.any():
            # axis_swirl_unit
            mnc_idx = torch.where(mask_noncollinear)[0]
            asw = axis_swirl[mask_noncollinear]
            asw_norm = axis_swirl_norm[mask_noncollinear].unsqueeze(1)
            axis_swirl_unit = asw / asw_norm

            # angle_swirl = arccos(dot(z_rest,z_hat))
            angle_swirl = torch.acos(dot_val[mask_noncollinear].clamp(-1.0,1.0))
            # axis-angle => shape (N,3)
            swirl_axis_angle = axis_swirl_unit * angle_swirl.unsqueeze(1)
            swirl_batch = axis_angle_to_matrix(swirl_axis_angle) # (N,3,3)

            swirl_sub[mnc_idx] = swirl_batch

        # Now swirl_mat[mask_ok] = swirl_sub
        swirl_mat[mask_ok] = swirl_sub

        # 4) twist_mat = swirl_mat^T @ R
        #    R_sub = R[mask_ok], swirl_sub => shape (N,3,3)
        R_sub = R[mask_ok]
        twist_sub = torch.bmm(swirl_sub.transpose(1,2), R_sub)
        twist_mat[mask_ok] = twist_sub

        # 5) twist_angle => norm of axis_angle for twist_sub
        twist_axis_angle = matrix_to_axis_angle(twist_sub)  # (N,3)
        twist_angle_sub  = twist_axis_angle.norm(dim=1)     # (N,)
        twist_angle[mask_ok] = twist_angle_sub

    return swirl_mat, twist_mat, twist_angle



class ArM:
    def to(self, device):
        return self

    def __init__(self, n_theta=50, n_z=12):
        """
        Initialize the arm segment parameters.

        :param n_theta: Number of angular segments (discretization in theta direction).
        :param n_z: Number of height segments (discretization in height direction).
        """
        self.n_theta = n_theta
        self.n_z = n_z

        self.faces = self.create_faces()

    def generate_segment(self, r1, r2, h, roffsets=None, rheights=None):
        """
        Generate batched parametric coordinates of arm segments.

        :param r1: Tensor of shape (batch_size, 1) - Radii at the base of each arm segment.
        :param r2: Tensor of shape (batch_size, 1) - Radii at the top of each arm segment.
        :param h: Tensor of shape (batch_size, 1) - Heights of each arm segment.
        :param roffsets: Tensor of shape (batch_size, n_z) - Radial offsets for each segment.
        :param rheights: Tensor of shape (batch_size, n_z) - Relative heights for each segment.
        :return: Tensor of shape (batch_size, num_vertices, 3) - Points for each mesh in the batch.
        """
        batch_size = r1.shape[0]

        if roffsets is None:
            roffsets = torch.zeros((batch_size, self.n_z), dtype=torch.float32, device=r1.device)
        if rheights is None:
            rheights = torch.linspace(0, 1, self.n_z, dtype=torch.float32, device=r1.device).unsqueeze(0).repeat(batch_size, 1)

        # Scale the rheights to the actual segment heights
        rheights_scaled = rheights * h  # Shape: (batch_size, n_z)

        # Linear interpolation between r1 and r2
        linear_radii = torch.linspace(0, 1, self.n_z, dtype=torch.float32, device=r1.device).unsqueeze(0).repeat(batch_size, 1)  # Shape: (batch_size, n_z)
        linear_radii = r1 + (r2 - r1) * linear_radii  # Shape: (batch_size, n_z)

        # Apply offsets
        radii_with_offsets = linear_radii + roffsets  # Shape: (batch_size, n_z)

        # Create grid for theta and z
        theta = torch.linspace(0, 2 * torch.pi, self.n_theta, dtype=torch.float32, device=r1.device)
        theta_grid = theta.unsqueeze(0).repeat(batch_size, 1)  # Shape: (batch_size, n_theta)

        z = rheights_scaled  # Shape: (batch_size, n_z)

        # Expand radii to match grid shape
        radii_grid = radii_with_offsets.unsqueeze(1).repeat(1, self.n_theta, 1)  # Shape: (batch_size, n_theta, n_z)

        # Compute x, y, z coordinates
        x = radii_grid * torch.cos(theta_grid.unsqueeze(2))  # Shape: (batch_size, n_theta, n_z)
        y = radii_grid * torch.sin(theta_grid.unsqueeze(2))  # Shape: (batch_size, n_theta, n_z)

        # Combine coordinates
        z_grid = z.unsqueeze(1).repeat(1, self.n_theta, 1)  # Shape: (batch_size, n_theta, n_z)

        # Flatten the grids
        points = torch.stack([x, y, z_grid], dim=-1).reshape(batch_size, -1, 3)  # Shape: (batch_size, n_theta*n_z, 3)

        # Add top and bottom center points
        top_centers = torch.stack([torch.zeros(batch_size, 1, dtype=torch.float32, device=r1.device),
                                   torch.zeros(batch_size, 1, dtype=torch.float32, device=r1.device),
                                   h], dim=-1)  # Shape: (batch_size, 1, 3)
        bottom_centers = torch.zeros((batch_size, 1, 3), dtype=torch.float32, device=r1.device)  # Shape: (batch_size, 1, 3)

        # Compute midpoints (center of the segment along z)
        mid_z = h / 2  # Assuming linear height distribution
        midpoints = torch.stack([
            torch.zeros(batch_size, 1, dtype=torch.float32, device=r1.device),
            torch.zeros(batch_size, 1, dtype=torch.float32, device=r1.device),
            mid_z
        ], dim=-1)  # Shape: (batch_size, 1, 3)

        points = torch.cat([points, top_centers, bottom_centers, midpoints], dim=1)  # Shape: (batch_size, n_theta*n_z + 3, 3)

        # Store indices for joints
        self.top_center_idx = points.shape[1] - 3  # Index of the top center
        self.bottom_center_idx = points.shape[1] - 2  # Index of the bottom center
        self.mid_center_idx = points.shape[1] - 1  # Index of the midpoint

        radius_bottom = radii_with_offsets[:, :1]          # (B, 1)
        radius_mid    = radii_with_offsets[:, self.n_z//2:self.n_z//2+1]
        radius_top    = radii_with_offsets[:, -1:]

        def lr_pair(r, z_val):            
            left  = torch.stack([-r, torch.zeros_like(r), z_val], dim=-1)
            right = torch.stack([ r, torch.zeros_like(r), z_val], dim=-1)
            return left, right

        top_left,   top_right   = lr_pair(radius_top,    h)
        mid_left,   mid_right   = lr_pair(radius_mid,    mid_z)
        bottom_left,bottom_right= lr_pair(radius_bottom, torch.zeros_like(h))

        extrema = torch.cat([top_left, top_right,
                            mid_left, mid_right,
                            bottom_left, bottom_right], dim=1)

        self.top_left_idx     = self.mid_center_idx + 1
        self.top_right_idx    = self.top_left_idx   + 1
        self.mid_left_idx     = self.top_right_idx  + 1
        self.mid_right_idx    = self.mid_left_idx   + 1
        self.bottom_left_idx  = self.mid_right_idx  + 1
        self.bottom_right_idx = self.bottom_left_idx+ 1

        return points, extrema

    def __call__(self, r1, r2, h, roffsets=None, rheights=None, R_axis=None, T=None, return_params=False, full_joints=False):
        """
        Apply rotation and translation to batched arm segments.

        :param r1: Tensor of shape (batch_size, 1) - Radii at the base.
        :param r2: Tensor of shape (batch_size, 1) - Radii at the top.
        :param h: Tensor of shape (batch_size, 1) - Heights.
        :param R_axis: Tensor of shape (batch_size, 3) - Rotation axes.
        :param T: Tensor of shape (batch_size, 3) - Translations.
        :param roffsets: Tensor of shape (batch_size, n_z) - Radial offsets.
        :param rheights: Tensor of shape (batch_size, n_z) - Relative heights.
        :return: Tensor of shape (batch_size, num_vertices, 3) - Transformed points.
        """
        # Ensure input tensors have correct shapes
        assert r1.shape == r2.shape == h.shape, "r1, r2, and h must have the same shape."
        assert r1.dim() == 2 and r1.size(1) == 1, "r1, r2, and h must have shape (batch_size, 1)."

        batch_size = r1.size(0)

        assert roffsets is None or roffsets.shape == (batch_size, self.n_z), f"roffsets must have shape ({batch_size}, {self.n_z})."
        assert rheights is None or rheights.shape == (batch_size, self.n_z), f"rheights must have shape ({batch_size}, {self.n_z})."
        assert R_axis is None or R_axis.shape == (batch_size, 3), f"R_axis must have shape ({batch_size}, 3)."
        assert T is None or T.shape == (batch_size, 3), f"T must have shape ({batch_size}, 3)."

        # Generate batched points
        points, extrema = self.generate_segment(r1, r2, h, roffsets=roffsets, rheights=rheights)  # Shape: (batch_size, num_vertices, 3)
        z = points[:, :, 2]  # Shape: (batch_size, num_vertices)

        # Initialize translations and rotation axes if not provided
        if T is None:
            T = torch.zeros(batch_size, 3, dtype=torch.float32, device=r1.device)
        if R_axis is None:
            R_axis = torch.zeros(batch_size, 3, dtype=torch.float32, device=r1.device)

        # swirl_mat, twist_mat, twist_angle = rotation_decompose_twist_swirl(R_axis)
        # rotation_matrices = swirl_mat

        rotation_matrices = axis_angle_to_matrix(R_axis)  # Shape: (batch_size, 3, 3)

        # Compute midpoints for each arm segment
        midpoints_z = (z.max(dim=1).values + z.min(dim=1).values) / 2  # Shape: (batch_size,)
        midpoints = torch.stack([torch.zeros(batch_size, dtype=torch.float32, device=r1.device),
                                 torch.zeros(batch_size, dtype=torch.float32, device=r1.device),
                                 midpoints_z], dim=1)  # Shape: (batch_size, 3)


        points = torch.cat([points, extrema], dim=1)
        
        # Translate points to origin (midpoint)
        points_translated = points - midpoints.unsqueeze(1)  # Shape: (batch_size, num_vertices, 3)

        # Apply rotation
        rotated_points = torch.bmm(points_translated, rotation_matrices.transpose(1, 2))  # Shape: (batch_size, num_vertices, 3)
    
        # Translate back and apply translation T
        transformed_points = rotated_points + midpoints.unsqueeze(1) + T.unsqueeze(1)  # Shape: (batch_size, num_vertices, 3)
            
        transformed_joints = torch.stack([
            transformed_points[:, self.top_center_idx, :],     # Top joint
            transformed_points[:, self.mid_center_idx, :],     # Mid joint
            transformed_points[:, self.bottom_center_idx, :],   # Bottom joint
        ], dim=1)


        extra_joints = torch.stack([    
            transformed_points[:, self.top_left_idx, :],     
            transformed_points[:, self.top_right_idx, :],    

            transformed_points[:, self.mid_left_idx, :],     
            transformed_points[:, self.mid_right_idx, :],    

            transformed_points[:, self.bottom_left_idx, :],     
            transformed_points[:, self.bottom_right_idx, :],    
        ], dim=1)

        if full_joints:
            transformed_joints = torch.cat([transformed_joints, extra_joints], dim=1)

        transformed_points = transformed_points[:, :-(1+6), :]  # Exclude the mid joint, and the extrema joints

        if return_params:
            return transformed_points, transformed_joints, {
                'r1': r1,
                'r2': r2,
                'h': h,
                'roffsets': roffsets,
                'R_axis': R_axis,
                'T': T,
                'midpoints': midpoints,
            }

        return transformed_points, transformed_joints


    def create_faces(self):
        """
        Create triangular faces for the mesh from the (n_theta, n_z) grid.
        Faces are created by connecting adjacent points in the grid.
        Also, create faces for the top and bottom to make the mesh watertight.
        """
        faces = []

        # Side faces
        for i in range(self.n_theta - 1):
            for j in range(self.n_z - 1):
                p1 = i * self.n_z + j
                p2 = (i + 1) * self.n_z + j
                p3 = i * self.n_z + (j + 1)
                p4 = (i + 1) * self.n_z + (j + 1)

                faces.append([p1, p2, p3])
                faces.append([p2, p4, p3])

        # Top and bottom center indices
        top_center_idx = self.n_theta * self.n_z
        bottom_center_idx = top_center_idx + 1

        # Top faces
        for i in range(self.n_theta - 1):
            p1 = (i + 1) * self.n_z - 1
            p2 = (i + 2) * self.n_z - 1 if (i + 2) * self.n_z - 1 < self.n_theta * self.n_z else self.n_z - 1
            faces.append([p1, p2, top_center_idx])

        # Bottom faces
        for i in range(self.n_theta - 1):
            p1 = i * self.n_z
            p2 = (i + 1) * self.n_z if (i + 1) * self.n_z < self.n_theta * self.n_z else 0
            faces.append([bottom_center_idx, p2, p1])

        return np.array(faces)


    def compute_volume(self, r1, r2, h, roffsets=None, rheights=None):
        # Ensure input tensors have correct shapes
        assert r1.shape == r2.shape == h.shape, "r1, r2, and h must have the same shape."
        assert r1.dim() == 2 and r1.size(1) == 1, "r1, r2, and h must have shape (batch_size, 1)."

        batch_size = r1.size(0)
        
        assert roffsets is None or roffsets.shape == (batch_size, self.n_z), f"roffsets must have shape ({batch_size}, {self.n_z})."
        assert rheights is None or rheights.shape == (batch_size, self.n_z), f"rheights must have shape ({batch_size}, {self.n_z})."

        if rheights is None:
            rheights = torch.linspace(0, 1, self.n_z, dtype=torch.float32, device=r1.device).unsqueeze(0).repeat(batch_size, 1)
        
        if roffsets is None:
            roffsets = torch.zeros((batch_size, self.n_z), dtype=torch.float32, device=r1.device)

        # Compute radii at each height level
        radii_z = r1 + roffsets + (r2 - r1) * rheights  # Shape: (batch_size, n_z)

        # Height difference between each pair of sections
        dz = h / (self.n_z - 1)  # Shape: (batch_size, 1)

        # Compute the radii at each level
        r1_seg = radii_z[:, :-1]  # Shape: (batch_size, n_z -1)
        r2_seg = radii_z[:, 1:]   # Shape: (batch_size, n_z -1)

        # Compute volumes for each frustum (vectorized)
        volumes = (1/3) * torch.pi * dz * (r1_seg**2 + r1_seg * r2_seg + r2_seg**2)  # Shape: (batch_size, n_z -1)
        # Sum all the frustum volumes to get the total volume
        total_volume = torch.sum(volumes, dim=1) # Shape: (batch_size,)

        return total_volume

    def get_lower_vertex_indices(self):
        # Generate indices for the lower ring using step size of n_z
        lower_indices = np.arange(0, (self.n_theta - 1) * self.n_z, self.n_z)
        # Append the bottom center vertex index
        lower_indices = np.append(lower_indices, self.n_theta * self.n_z + 1)
        
        return lower_indices

    def get_upper_vertex_indices(self):
        # Generate indices for the upper ring starting from (n_z - 1) with step size of n_z
        upper_indices = np.arange(self.n_z - 1, (self.n_theta - 1) * self.n_z, self.n_z)
        # Append the top center vertex index
        upper_indices = np.append(upper_indices, self.n_theta * self.n_z)

        return upper_indices
    

class ArMPCA(ArM):
    def __init__(self, model_file, n_components=5):
        """
        Initialize the arm segment parameters.

        :param n_theta: Number of angular segments (discretization in theta direction).
        :param n_z: Number of height segments (discretization in height direction).
        :param n_components: Number of PCA components to use.
        """
        param_data = np.load(model_file, allow_pickle=True).item()
        pca_components = torch.tensor(param_data['pca_components'])[:n_components]
        pca_mean = torch.tensor(param_data['pca_mean'])
        scaler_mean = torch.tensor(param_data['scaler_mean'])
        scaler_scale = torch.tensor(param_data['scaler_scale'])
        
        self.W_fused = pca_components * scaler_scale
        self.W_fused.requires_grad = False

        self.b_fused = pca_mean * scaler_scale + scaler_mean
        self.b_fused.requires_grad = False

        self.n_components = n_components
        self.n_theta = param_data['arm_n_theta']
        self.n_z = param_data['arm_n_z']

        print('Constructing Arm Model')
        print('Number of theta segments:', self.n_theta)
        print('Number of z segments:', self.n_z)
        print('Number of PCA components:', self.n_components)

        super().__init__(self.n_theta, self.n_z)

    def to(self, device):
        self.W_fused = self.W_fused.to(device)
        self.b_fused = self.b_fused.to(device)

        return self

    def pca_to_params(self, pca_values):
        params = torch.matmul(pca_values, self.W_fused) + self.b_fused

        r1 = params[:, 0].unsqueeze(1)
        r2 = params[:, 1].unsqueeze(1)
        h = params[:, 2].unsqueeze(1)
        roffsets = params[:, 3:]

        return r1, r2, h, roffsets

    def params_to_pca(self, r1, r2, h, roffsets):
        params = torch.cat([r1, r2, h, roffsets], dim=1)
        W_fused_pinv = torch.pinverse(self.W_fused)
        pca_values = torch.matmul(params - self.b_fused, W_fused_pinv)
        return pca_values


    def __call__(self, pca_values, R_axis=None, T=None, return_params=False, full_joints=False):
        assert len(pca_values.shape) == 2, "pca_values must have shape (batch_size, n_components)."
        assert pca_values.shape[1] == self.n_components, f"pca_values must have {self.n_components} components."
        
        # print(pca_values)
        r1, r2, h, roffsets = self.pca_to_params(pca_values)
        
        # swap = (r1 < r2).squeeze(1) 
        # temp = r1[swap].clone()
        # r1[swap] = r2[swap]
        # r2[swap] = temp

        # # Reverse the order of radial offsets where necessary
        # roffsets_swap = torch.flip(roffsets[swap], dims=[1])
        # roffsets[swap] = roffsets_swap

        return super().__call__(r1, r2, h, roffsets, R_axis=R_axis, T=T, return_params=return_params, full_joints=full_joints)


def main():
    n_components = 5
    arm = ArMPCA(model_file='param_data.npy', n_components=n_components).to('cuda')
    pca_values = torch.randn(32, n_components).to('cuda')
    vertices, joints = arm(pca_values)
    faces = arm.faces

    print('Vertices:', vertices.shape)
    print('Faces:', faces.shape)
    print('Joints:', joints.shape)


if __name__ == '__main__':
    main()  
