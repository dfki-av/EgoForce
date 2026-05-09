import os
import sys
import torch
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from models.mano_layer import MANOHandModel
from models.arm_model import ArMPCA
from utils.rotations import rotation_6d_to_axis_angle_direct
from types import SimpleNamespace


class LimbModel:
    def __init__(self, cfg, device="cpu", use_pose_pca=False, n_components=5):
        self.cfg = cfg
        
        self.hand_model = MANOHandModel(cfg.MANO_PATH, device=device, use_pose_pca=use_pose_pca)
        self.arm_model = ArMPCA(model_file=f'{ROOT_DIR}/models/arm_model_param_data.npy', n_components=n_components).to(device)

        self.faces = SimpleNamespace()
        self.faces.arm = self.arm_model.faces
        self.faces.left_hand = self.hand_model.mano_layer_left.faces
        self.faces.right_hand = self.hand_model.mano_layer_right.faces

    def __call__(self, betas, global_orient, hand_pose, transl, is_right_hand, arm_shape, arm_rot, arm_T=None):
        B = global_orient.shape[0]

        if global_orient.shape[-1] == 6:
            global_orient = rotation_6d_to_axis_angle_direct(global_orient.reshape(-1, 6))
            hand_pose = rotation_6d_to_axis_angle_direct(hand_pose.reshape(-1, 6))
            arm_rot = rotation_6d_to_axis_angle_direct(arm_rot.reshape(-1, 6))

        betas = betas.reshape(B, -1)
        global_orient = global_orient.reshape(B, -1)
        hand_pose = hand_pose.reshape(B, -1)
        transl = transl.reshape(B, -1)
        is_right_hand = is_right_hand.bool() # 0 - left, 1 - right

        arm_shape = arm_shape.reshape(B, -1)
        arm_rot = arm_rot.reshape(B, -1)

        global_xfrom = torch.cat([global_orient, transl], dim=1)

        hand_vertices, hand_joints = self.hand_model.forward_kinematics(shape_params=betas,
                                                                        joint_angles=hand_pose,
                                                                        global_xfrom=global_xfrom,
                                                                        is_right_hand=is_right_hand)

        if arm_T is None:
            arm_vertices, arm_joints = self.arm_model(pca_values=arm_shape, R_axis=arm_rot, T=torch.zeros_like(transl), full_joints=False)


            arm_vertices = arm_vertices - arm_joints[:, 2].unsqueeze(1) # Wrist is #2nd joint
            arm_joints   = arm_joints   - arm_joints[:, 2].unsqueeze(1)

            # 2) compute the “bone” vector from wrist → elbow:
            bone_vec = (arm_joints[:, 0] - arm_joints[:, 2])        # [B×3]
            #    └── where 0 is elbow/root, –1 is wrist

            length  = torch.norm(bone_vec, dim=-1).unsqueeze(1) + 1e-8
            unit    = bone_vec / length
            offset  = unit * (length * 0.03)   # still 3% of the full bone length

            # 4) shift back into world‐space:
            arm_vertices = arm_vertices + hand_joints[:, 0].unsqueeze(1)
            arm_joints   = arm_joints   + hand_joints[:, 0].unsqueeze(1)

            # 5) finally, move “below” the wrist by adding that small offset:
            arm_vertices = arm_vertices + offset.unsqueeze(1)
            arm_joints   = arm_joints   + offset.unsqueeze(1)
        else:
            arm_vertices, arm_joints = self.arm_model(pca_values=arm_shape, R_axis=arm_rot, T=arm_T, full_joints=True)

        output = SimpleNamespace()
        output.hand = SimpleNamespace()
        output.hand.vertices = hand_vertices
        output.hand.joints = hand_joints

        output.arm = SimpleNamespace()
        output.arm.vertices = arm_vertices
        output.arm.joints = arm_joints

        return output

    def zero_output(self, B=0, device='cpu'):
        output = SimpleNamespace()
        output.hand = SimpleNamespace()
        output.hand.vertices = torch.empty(B, 778, 3, device=device)
        output.hand.joints = torch.empty(B, 21, 3, device=device)

        output.arm = SimpleNamespace()
        output.arm.vertices = torch.empty(B, 503, 3, device=device)
        output.arm.joints = torch.empty(B, 3, 3, device=device)

        return output