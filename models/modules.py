import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import os
import numpy as np
from torch import nn
from torch.nn import functional as F


def rotation_6d_to_axis_angle_direct_np(d6):
    """
    Convert a 6D rotation representation (Zhou et al.) directly to an axis–angle representation.

    Args:
        d6: ndarray of shape (..., 6) representing the 6D rotation.

    Returns:
        ndarray of shape (..., 3) representing the axis–angle vector.
        The direction is the rotation axis, and the norm is the rotation angle in radians.
    """
    eps = 1e-7

    # 1) Convert 6D to orthonormal basis (b1, b2, b3) via Gram-Schmidt
    a1 = d6[..., :3]
    a2 = d6[..., 3:]

    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True)
    b2 = a2 - proj * b1
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2, axis=-1)

    # 2) Compute trace of rotation matrix R = [b1, b2, b3]
    trace = b1[..., 0] + b2[..., 1] + b3[..., 2]
    cos_theta = (trace - 1.0) * 0.5
    cos_theta_clamped = np.clip(cos_theta, -1.0 + eps, 1.0 - eps)
    theta = np.arccos(cos_theta_clamped)

    # 3) Compute skew-symmetric vector (from R - R^T)
    rx = b3[..., 1] - b2[..., 2]
    ry = b1[..., 2] - b3[..., 0]
    rz = b2[..., 0] - b1[..., 1]
    skew_vec = np.stack([rx, ry, rz], axis=-1)

    sin_theta = np.sin(theta)
    factor = np.where(np.abs(sin_theta) > eps,
                      theta / (2.0 * sin_theta),
                      0.5)

    axis_angle = factor[..., np.newaxis] * skew_vec
    return axis_angle


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                    bias=False, act_type='relu', **kwargs):
        is_sequence = isinstance(kernel_size, (list, tuple))
        padding_fns = (
            lambda ks: (ks - 1) // 2 * dilation,
            lambda ks: ((ks[0] - 1) // 2 * dilation, (ks[1] - 1) // 2 * dilation),
        )
        padding = padding_fns[int(is_sequence)](kernel_size)

        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
        self.out_channels = out_channels


class Conv1DBNAct(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                    bias=False, act_type='relu', **kwargs):
        is_sequence = isinstance(kernel_size, (list, tuple))
        padding_fns = (
            lambda ks: (ks - 1) // 2 * dilation,
            lambda ks: ((ks[0] - 1) // 2 * dilation, (ks[1] - 1) // 2 * dilation),
        )
        padding = padding_fns[int(is_sequence)](kernel_size)
        activation = (nn.Identity(), nn.ReLU(inplace=True))[int(act_type == 'relu')]

        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias),
            nn.BatchNorm1d(out_channels),
            activation,
        )




class HeatMap2DRegressor(nn.Module):
    def integral_pose_regression(self, heatmaps):
        batch_size, num_joints, height, width = heatmaps.shape
        
        # Apply softmax to each heatmap to normalize them into probability distributions
        heatmaps = heatmaps.view(batch_size, num_joints, -1)  # Flatten the heatmaps (height * width)
        heatmaps = F.softmax(heatmaps * self.hm_temp.exp(), dim=-1)  # Softmax over spatial dimensions
        heatmaps = heatmaps.view(batch_size, num_joints, height, width)  # Reshape back
        
        # Create a meshgrid of (x, y) coordinates
        grid_y, grid_x = torch.meshgrid(
            torch.arange(height, device=heatmaps.device),
            torch.arange(width,  device=heatmaps.device),
            indexing='ij')              # explicit is safer
                
        # Expand grid to batch and joint dimensions
        grid_x = grid_x.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, height, width)
        grid_y = grid_y.unsqueeze(0).unsqueeze(0)  # Shape (1, 1, height, width)
        
        # Compute the expected x and y coordinates by taking the weighted sum over the grid
        joint_x = torch.sum(heatmaps * grid_x, dim=(2, 3)) / (width - 1)  # Shape (batch_size, num_joints)
        joint_y = torch.sum(heatmaps * grid_y, dim=(2, 3)) / (height - 1) # Shape (batch_size, num_joints)
        
        # Stack the x and y coordinates to get the final (x, y) joint coordinates
        joint_coords = torch.stack([joint_x, joint_y ], dim=-1)  # Shape (batch_size, num_joints, 2)
        
        return joint_coords

    def __init__(self, image_size, in_channels, feature_size, n_joints, simdr_split_ratio=1.0):
        super().__init__()
        self.height, self.width = image_size
        self.simdr_width = int(self.width * simdr_split_ratio)
        self.simdr_height = int(self.height * simdr_split_ratio)

        # Convolutional layers (1D over spatial tokens)
        self.conv_layers = nn.Sequential(
            ConvBNAct(in_channels, in_channels // 2, kernel_size=3, act_type='relu'),
            ConvBNAct(in_channels // 2, in_channels // 4, kernel_size=3, act_type='relu'),
            ConvBNAct(in_channels // 4, in_channels // 8, kernel_size=3, act_type='relu'),
        )
        self.joint_regressor = nn.Sequential(
            nn.Conv2d(in_channels // 8, in_channels // 8, 3, padding=1, groups=in_channels // 8),
            nn.Conv2d(in_channels // 8, n_joints, 1)
        )
        # self
        self.T = nn.Parameter(torch.tensor(10.0))
        self.hm_temp = nn.Parameter(torch.tensor(100.0).log())

        self.weight_regressor = nn.Sequential(
            nn.Linear(in_channels // 8, in_channels // 4),
            nn.ReLU(),
            nn.Linear(in_channels // 4, in_channels // 2),
            nn.ReLU(),
            nn.Linear(in_channels // 2, 1),
            nn.Sigmoid()
        )

        last_lin = self.weight_regressor[-2]
        nn.init.zeros_(last_lin.weight)
        nn.init.constant_(last_lin.bias, 10.0)  # sigmoid(5) ≈ 0.993

    def forward(self, x, k, size):
        B, HW, C = x.shape

        a = torch.einsum('bhc,bc->bh', x, k).div(self.T)     # dot‑product similarity → (B, HW)
        a = a.softmax(dim=-1).unsqueeze(-1)      # normalise over patches
        x = (x * a).permute(0, 2, 1)             # each patch scaled by its own weight

        hm_sz = int(HW ** 0.5)
        x = x.reshape(B, C, hm_sz, hm_sz)

        x = self.conv_layers(x)

        x = F.interpolate(x, size=(56, 56), mode='bilinear', align_corners=False)
        
        hm = self.joint_regressor(x)

        xy = self.integral_pose_regression(hm)
        xy_f = self.sample_features_from_joints(x, xy)
        w = self.weight_regressor(xy_f.detach())

        xy[:, :, 0] *= (self.width - 1)
        xy[:, :, 1] *= (self.height - 1)

        B, T = size
        xy = xy.view(B, T, -1, 2)
        hm = hm.view(B, T, *hm.shape[1:])
        w = w.view(B, T, -1)

        # Always return a fixed output tuple for script/export stability.
        return xy, hm, w, xy_f

    def sample_features_from_joints(self, features, joint_positions):
        """
        features: (B, C, H, W) - Output from decoder
        joint_positions: (B, num_joints, 2) - 2D positions of joints (normalized between 0 and 1)
        Returns:
        node_features: (B, num_joints, C) - Extracted features for each joint
        """
        B, C, H, W = features.shape
        num_joints = joint_positions.shape[1]

        # Normalize joint_positions to range [-1, 1] for grid_sample
        grid = joint_positions * 2 - 1 

        grid = grid.view(B, num_joints, 1, 2) 

        # Sample features at joint positions
        node_features = F.grid_sample(features, grid, mode='bilinear', align_corners=True)

        node_features = node_features.squeeze(2).squeeze(-1).permute(0, 2, 1)  # (B, C, num_joints) -> (B, num_joints, C)

        return node_features  # Shape: (B, num_joints, C)


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.LayerNorm(dim)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.fc1(x)))
        out = self.bn2(self.fc2(out))
        return F.relu(out + residual)


class Conv1DNormReLU(nn.Module):
    def __init__(self, dim_in, dim_out=None, kernel_size=3, stride=1, num_groups=8):
        super().__init__()
        dim_out = (dim_out, dim_in)[int(dim_out is None)]

        # It’s important to be careful with padding so that the output size matches the input size.
        # Here, for a 1D convolution, using kernel_size//2 as padding often works for odd kernel sizes.
        self.conv = nn.Sequential(
            nn.Conv1d(dim_in, dim_out, kernel_size, stride, padding=kernel_size//2),
            nn.GroupNorm(num_groups, dim_out),
            # Note: removing ReLU here allows us to apply it after the skip addition.
        )
        self.relu = nn.ReLU(inplace=True)
        # Keep a concrete module in all cases so the forward graph has a single path.
        needs_projection = int((stride != 1) or (dim_in != dim_out))
        downsample_layers = (
            nn.Identity(),
            nn.Conv1d(dim_in, dim_out, kernel_size=1, stride=stride),
        )
        self.downsample = downsample_layers[needs_projection]

    def forward(self, x):
        identity = self.downsample(x)  # save projected input for the skip connection

        out = self.conv(x)

        # Add the skip connection.
        out += identity

        # Apply activation after the addition.
        out = self.relu(out)
        return out


class ReLUGate(nn.Hardtanh):
    def __init__(self, inplace: bool = False):
        super().__init__(0.0, 6.0, inplace)

    def extra_repr(self) -> str:
        return ("", "inplace=True")[int(self.inplace)]

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        relu_out = super().forward(input)
        relu_out = relu_out / 6.0  
        return relu_out


class ArmGenerativePrior(nn.Module):
    def __init__(self, input_dim, latent_dim, output_dim):
        super().__init__()

        hidden_dim = input_dim

        self.mu_fc = nn.Linear(hidden_dim, latent_dim)
        self.logvar_fc = nn.Linear(hidden_dim, latent_dim)

        # Latent projection
        self.latent_to_hidden = nn.Linear(latent_dim, hidden_dim)

        # 3 residual blocks
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        self.res_block3 = ResidualBlock(hidden_dim)

        # Final projection to output_dim
        self.final_layer = nn.Linear(hidden_dim, output_dim)

        self.hand_pooling = nn.Sequential(
            Conv1DNormReLU(input_dim, dim_out=input_dim, kernel_size=7),
            Conv1DNormReLU(input_dim, kernel_size=3, stride=2),
            Conv1DNormReLU(input_dim, kernel_size=3, stride=2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, hand_feats):
        hand_feats = self.hand_pooling(hand_feats.permute(0, 2, 1))          # [B, C]   
                
        mu = self.mu_fc(hand_feats)
        logvar = self.logvar_fc(hand_feats)
        z = self.reparameterize(mu, logvar)

        hidden = self.latent_to_hidden(z)         # [B, hidden_dim]
        hidden = self.res_block1(hidden)
        hidden = self.res_block2(hidden)
        hidden = self.res_block3(hidden)

        arm_embedding = self.final_layer(self.dropout(hidden))

        return arm_embedding, mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
 

class DecoderHead(nn.Module):
    def __init__(self, in_features, residual_features, out_features=None):
        super(DecoderHead, self).__init__()
        self.dropout = nn.Dropout(0.1)

        out_features = (out_features, residual_features)[int(out_features is None)]
        
        self.accumulator_fc = nn.Linear(in_features + residual_features, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, out_features)
        self.gate = nn.Linear(in_features, out_features)
        self.gate.weight.data.fill_(0.0)
        self.gate.bias.data.fill_(1.0) 
        self.gate_fn = ReLUGate(inplace=True)

    def forward(self, x, param):
        acc = torch.cat([x, param], dim=-1)  # Concatenate along the feature dimension

        f = F.relu(self.accumulator_fc(acc), inplace=True)
        f = self.dropout(f)
        f = F.relu(self.fc2(f), inplace=True)
        f = self.dropout(f)
        delta = self.fc3(f)
        gate = self.gate(x)
        delta = delta * self.gate_fn(gate)  # Apply the gate to the output

        return delta


class HandQueryDecoder(nn.Module):
    def __init__(self, feature_dim=2048, 
                 shape_dim=10, 
                 pose_local_dim=6 * 15, 
                 pose_global_dim=6,
                 transl_dim=3, 
                 num_iterations=3):
        super(HandQueryDecoder, self).__init__()
        self.rot_dim = pose_global_dim
        self.num_iterations = num_iterations

        mean_params = np.load(os.path.join(os.path.dirname(__file__), 'mano_mean_params.npz'))
        init_betas = torch.from_numpy(mean_params['shape'].astype('float32')).unsqueeze(0)
        
        init_hand_pose_complete = mean_params['pose'].astype(np.float32).reshape(-1, 6)
        pose_candidates = (
            init_hand_pose_complete,
            rotation_6d_to_axis_angle_direct_np(init_hand_pose_complete),
        )
        init_hand_pose_complete = pose_candidates[int(pose_global_dim == 3)]
    
        global_pose = init_hand_pose_complete[0].reshape(1, -1)
        local_pose = init_hand_pose_complete[1:].reshape(1, -1)

        init_hand_pose = torch.from_numpy(local_pose)

        self.register_buffer('init_hand_pose', init_hand_pose)
        self.register_buffer('init_betas', init_betas)

        # Each head predicts a delta to be added to the current parameter estimate.
        self.shape_initilizer = nn.Linear(feature_dim, shape_dim)
        self.pose_local_initilizer = nn.Linear(feature_dim, pose_local_dim)
        self.xglobal_initilizer = nn.Linear(feature_dim, pose_global_dim)

        self.xglobal_decoder = DecoderHead(feature_dim, pose_global_dim)
        self.shape_decoder = DecoderHead(feature_dim, shape_dim)
        self.pose_local_decoder = DecoderHead(feature_dim, pose_local_dim)

    def forward(self, features, size):
        batch_size = features.shape[0]
        device = features.device

        xglobal_feats = features[:, 1]
        shape_feats = features[:, 2]
        pose_local_feats = features[:, 3]

        # Initialize MANO parameters as zero vectors.
        shape = self.shape_initilizer(shape_feats)
        pose_local = self.pose_local_initilizer(pose_local_feats) + self.init_hand_pose.expand(batch_size, -1).to(device).clone()
        xglobal = self.xglobal_initilizer(xglobal_feats)
        pose_global = xglobal[:, :self.rot_dim]

        # Iteratively update the parameters.
        for _ in range(self.num_iterations):
            xglobal_delta = self.xglobal_decoder(xglobal_feats, xglobal)
            shape_delta = self.shape_decoder(shape_feats, shape)
            pose_local_delta = self.pose_local_decoder(pose_local_feats, pose_local)

            xglobal = xglobal + xglobal_delta
            shape = shape + shape_delta
            pose_local = pose_local + pose_local_delta
            
            pose_global = xglobal[:, :self.rot_dim]

   
        B, T = size

        shape = shape.view(B, T, -1)
        pose_global = pose_global.view(B, T, -1)
        pose_local = pose_local.view(B, T, 15, self.rot_dim)

        return shape, pose_global, pose_local


class ArmQueryDecoder(nn.Module):
    def __init__(self, feature_dim=2048, 
                 shape_dim=5, 
                 pose_global_dim=6,
                 transl_dim=3, 
                 num_iterations=3):
        super(ArmQueryDecoder, self).__init__()
        self.num_iterations = num_iterations
        self.rot_dim = pose_global_dim

        # Each head predicts a delta to be added to the current parameter estimate.
        self.shape_initilizer = nn.Linear(feature_dim, shape_dim)
        self.xglobal_initilizer = nn.Linear(feature_dim, pose_global_dim)

        self.shape_decoder = DecoderHead(feature_dim, shape_dim)
        self.xglobal_decoder = DecoderHead(feature_dim, pose_global_dim)
 
    def forward(self, features, size):
        batch_size = features.shape[0]
        device = features.device

        xglobal_feats = features[:, 1]
        shape_feats = features[:, 2]

        # Initialize MANO parameters as zero vectors.
        shape = self.shape_initilizer(shape_feats)
        xglobal = self.xglobal_initilizer(xglobal_feats)
        R = xglobal[:, :self.rot_dim]

        # Iteratively update the parameters.
        for _ in range(self.num_iterations):
            xglobal_delta = self.xglobal_decoder(xglobal_feats, xglobal)
            shape_delta = self.shape_decoder(shape_feats, shape)

            xglobal = xglobal + xglobal_delta
            shape = shape + shape_delta

            R = xglobal[:, :self.rot_dim]

        B, Tsz = size
        shape = shape.view(B, Tsz, -1)
        R = R.view(B, Tsz, -1)

        return shape, R


class HandKeypointQueryFE(nn.Module):
    def __init__(self, 
                 in_channels=2048,
                 out_channels=640): 
        super().__init__()

        self.conv_layers = nn.Sequential(
            Conv1DBNAct(in_channels, 256, kernel_size=5, stride=2, act_type='relu'),
            nn.Dropout(0.2),
            Conv1DBNAct(256, 512, kernel_size=5, stride=2, act_type='relu'),
            nn.Dropout(0.2),
            Conv1DBNAct(512, out_channels, kernel_size=5, stride=2, act_type='relu'),
        )
        self.mlp = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv_layers(x)    
        x = x.mean(-1)
        x = self.mlp(x)

        return x.unsqueeze(1) 


class ArmKeypointQueryFE(nn.Module):
    def __init__(self, 
                 in_channels=2048,
                 out_channels=640): 
        super().__init__()

        self.conv_layers = nn.Sequential(
            Conv1DBNAct(in_channels, 256, kernel_size=3, stride=1, act_type='relu'),
            nn.Dropout(0.2),
            Conv1DBNAct(256, 512, kernel_size=3, stride=1, act_type='relu'),
            nn.Dropout(0.2),
            Conv1DBNAct(512, out_channels, kernel_size=3, stride=1, act_type='relu'),
        )
        self.mlp = nn.Linear(out_channels, out_channels)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.conv_layers(x)    
        x = x.mean(-1)
        x = self.mlp(x)

        return x.unsqueeze(1)
