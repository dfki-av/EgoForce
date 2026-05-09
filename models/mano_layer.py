# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
from typing import List, Optional

SMPLX_IMPORT_SUCCEEDED = False  # default suppose we can't import SMPLX

try:
    import smplx

    SMPLX_IMPORT_SUCCEEDED = True
except ImportError:
    print(
        "INFO: HOT3D hands requires smplx (See our GitHub repository for more information on its installation)."
    )

import torch


mano_joint_mapping = [0, 13, 14, 15, 16, 1, 2, 3, 17, 4, 5, 6, 18, 10, 11, 12, 19, 7, 8, 9, 20]


class MANOHandModel:
    N_VERT = 778
    N_LANDMARKS = 21
    MANO_FINGERTIP_VERT_INDICES = {
        "thumb": 744,
        "index": 320,
        "middle": 443,
        "ring": 554,
        "pinky": 671,
    }

    def __init__(
        self,
        mano_model_files_dir: str,
        joint_mapper: Optional[List] = mano_joint_mapping,
        device: str = "cpu",
        use_pose_pca: bool = True,
        num_pose_coeffs: int = 15,
        flat_hand_mean=False,
        fix_left_hand_shape=True
    ):
        mano_left_filename = os.path.join(mano_model_files_dir, "MANO_LEFT.pkl")
        mano_right_filename = os.path.join(mano_model_files_dir, "MANO_RIGHT.pkl")

        self.use_pose_pca = use_pose_pca
        self.num_pose_coeffs = num_pose_coeffs
        self.num_shape_params = 10
        self.device = device
        self.dtype = torch.float32
        self.joint_mapper = joint_mapper
        self.flat_hand_mean = flat_hand_mean

        self.mano_layer_left = smplx.create(
            mano_left_filename,
            "mano",
            use_pca=self.use_pose_pca,
            is_rhand=False,
            num_pca_comps=self.num_pose_coeffs,
            flat_hand_mean=self.flat_hand_mean,
        )
        self.mano_layer_left.to(self.device)

        self.mano_layer_right = smplx.create(
            mano_right_filename,
            "mano",
            use_pca=self.use_pose_pca,
            is_rhand=True,
            num_pca_comps=self.num_pose_coeffs,
            flat_hand_mean=self.flat_hand_mean,
        )
        self.mano_layer_right.to(self.device)

        if fix_left_hand_shape:
            # fix MANO shapedirs of the left hand bug (https://github.com/vchoutas/smplx/issues/48)
            if (
                torch.sum(
                    torch.abs(
                        self.mano_layer_left.shapedirs[:, 0, :]
                        - self.mano_layer_right.shapedirs[:, 0, :]
                    )
                )
                < 1
            ):
                self.mano_layer_left.shapedirs[:, 0, :] *= -1

        self._mano_extra_indices = torch.tensor(list(self.MANO_FINGERTIP_VERT_INDICES.values()), 
                                                dtype=torch.long, 
                                                device=self.device)

    def forward_kinematics(
        self,
        shape_params: torch.Tensor,
        joint_angles: torch.Tensor,
        global_xfrom: torch.Tensor,
        is_right_hand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        
        is_batched = len(joint_angles.shape) == 2
        if len(global_xfrom.shape) == 1:
            global_xfrom = global_xfrom.unsqueeze(0)
        assert global_xfrom.shape[1] == 6
        
        if len(joint_angles.shape) == 1:
            joint_angles = joint_angles.unsqueeze(0)
        
        if self.use_pose_pca:
            assert joint_angles.shape[1] == self.num_pose_coeffs
        
        assert is_right_hand.shape[0] == joint_angles.shape[0]

        num_frames = joint_angles.shape[0]

        if len(shape_params.shape) == 1:
            shape_params = shape_params.unsqueeze(0).expand(num_frames, -1)

        left_mask = ~is_right_hand
        right_mask = is_right_hand

        # Convert global transforms and joint angles once to target dtype.
        global_xfrom = global_xfrom.to(self.dtype)
        joint_angles = joint_angles.to(self.dtype)
        shape_params = shape_params.to(self.dtype)        
    
        # Left hand FK
        if left_mask.any():
            left_shape_params = shape_params[left_mask]
            left_global_xform = global_xfrom[left_mask]
            left_joint_angles = joint_angles[left_mask]            
        
            left_mano_output = self.mano_layer_left(
                betas=left_shape_params,
                global_orient=left_global_xform[:, :3],
                hand_pose=left_joint_angles,
                transl=left_global_xform[:, 3:],
                return_verts=True,  # MANO doesn't return landmarks as well if this is false
            )

        # Right hand FK
        if right_mask.any():
            right_shape_params = shape_params[right_mask]
            right_global_xform = global_xfrom[right_mask]
            right_joint_angles = joint_angles[right_mask]
            
            right_mano_output = self.mano_layer_right(
                betas=right_shape_params,
                global_orient=right_global_xform[:, :3],
                hand_pose=right_joint_angles,
                transl=right_global_xform[:, 3:],
                return_verts=True,  # MANO doesn't return landmarks as well if this is false
            )

        out_vertices = torch.empty(num_frames, self.N_VERT, 3, dtype=self.dtype, device=self.device)
        out_landmarks = torch.empty(num_frames, self.N_LANDMARKS, 3, dtype=self.dtype, device=self.device)

        if left_mask.any():
            out_vertices[left_mask] = left_mano_output.vertices

            if left_mano_output.joints.shape[1] != self.N_LANDMARKS:
                extra_joints = left_mano_output.vertices.index_select(1, self._mano_extra_indices)
                joints = torch.cat([left_mano_output.joints, extra_joints], dim=1)
            else:
                joints = left_mano_output.joints
            
            out_landmarks[left_mask] = joints


        if right_mask.any():
            out_vertices[right_mask] = right_mano_output.vertices

            if right_mano_output.joints.shape[1] != self.N_LANDMARKS:
                extra_joints = right_mano_output.vertices.index_select(1, self._mano_extra_indices)
                joints = torch.cat([right_mano_output.joints, extra_joints], dim=1)
            else:
                joints = right_mano_output.joints
            
            out_landmarks[right_mask] = joints


        if self.joint_mapper is not None:
            out_landmarks = out_landmarks[:, self.joint_mapper]

        if not is_batched:
            out_vertices = out_vertices.squeeze(0)
            out_landmarks = out_landmarks.squeeze(0)

        return out_vertices, out_landmarks


    def shape_only_forward_kinematics(
        self,
        shape_params: torch.Tensor,
        is_right_hand: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        is_batched = len(shape_params.shape) == 2
        if is_batched:
            assert shape_params.shape[1] == self.num_shape_params
            num_frames = shape_params.shape[0]
        else:
            assert shape_params.shape[0] == self.num_shape_params
            shape_params = shape_params.unsqueeze(0)
            num_frames = 1

        pose_params = torch.zeros((num_frames, 15), device=self.device)
        pose_xform = torch.zeros((num_frames, 6), device=self.device)

        return self.forward_kinematics(
            shape_params=shape_params,
            joint_angles=pose_params,
            global_xfrom=pose_xform,
            is_right_hand=is_right_hand,
        )

def loadManoHandModel(
    mano_model_files_dir: Optional[str],
) -> MANOHandModel:
    if not SMPLX_IMPORT_SUCCEEDED or mano_model_files_dir is None:
        return None

    return MANOHandModel(mano_model_files_dir)
