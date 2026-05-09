# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import cast

import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import drop_path, to_2tuple, trunc_normal_
from .modules import *


def sin_encode(angles: torch.Tensor, L: int = 4) -> torch.Tensor:
    """
    Sinusoidal encoding for angles.

    Args:
        angles (torch.Tensor): Tensor of shape (B, N), where B is batch size, N is number of angles.
        L (int): Number of frequency components (e.g., 4 => 1, 2, 4, 8).

    Returns:
        torch.Tensor: Encoded angles of shape (B, N, L*2), with sin and cos encodings interleaved.
    """
    # angles: (B, N)
    B, N = angles.shape

    # Create frequency vector: shape (L,)
    freqs = 2 ** torch.arange(L, dtype=angles.dtype, device=angles.device)

    # Expand dimensions to broadcast: (B, N, 1) * (1, 1, L) => (B, N, L)
    angle_expanded = angles.unsqueeze(-1) * freqs.view(1, 1, -1)

    # Compute sin and cos parts: shape (B, N, L)
    sin_part = torch.sin(angle_expanded)
    cos_part = torch.cos(angle_expanded)

    # Concatenate along last dim: shape (B, N, 2L)
    encoding = torch.cat([sin_part, cos_part], dim=-1)

    return encoding


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)
    
    def extra_repr(self):
        return 'p={}'.format(self.drop_prob)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(
            self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0.,
            proj_drop=0., attn_head_dim=None,):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.dim = dim

        head_dim = cast(int, (head_dim, attn_head_dim)[int(attn_head_dim is not None)])
        all_head_dim = head_dim * self.num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, all_head_dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(all_head_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, 
                 drop=0., attn_drop=0., drop_path=0., act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, attn_head_dim=None
                 ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, attn_head_dim=attn_head_dim
            )

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        drop_layers = (nn.Identity(), DropPath(drop_path))
        self.drop_path = drop_layers[int(drop_path > 0.0)]
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, ratio=1):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (ratio ** 2)
        self.patch_shape = (int(img_size[0] // patch_size[0] * ratio), int(img_size[1] // patch_size[1] * ratio))
        self.origin_patch_shape = (int(img_size[0] // patch_size[0]), int(img_size[1] // patch_size[1]))
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.ratio = ratio
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=(patch_size[0] // ratio), padding=4 + 2 * (ratio//2-1))

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape

        x = self.proj(x)
        Hp, Wp = x.shape[2], x.shape[3]

        x = x.flatten(2).transpose(1, 2)

        return x, (Hp, Wp)

    def get_patch_centers(self, Hp: int, Wp: int):
        """
        Returns a float tensor of shape (Hp*Wp, 2) giving (y, x) centers 
        in **original** image pixel coords.
        """
        # for your default (ratio=1) stride == patch_size
        patch_h, patch_w = self.patch_size
        stride_h = patch_h // self.ratio
        stride_w = patch_w // self.ratio

        # grid of output indices
        ys = torch.arange(Hp, dtype=torch.float32)
        xs = torch.arange(Wp, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

        # center formula (no need to account for conv‐padding if you want
        # centers _in the original image_):
        cy = grid_y * stride_h + patch_h * 0.5
        cx = grid_x * stride_w + patch_w * 0.5

        centers = torch.stack([cy, cx], dim=-1).view(-1, 2)
        return centers


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        use_inferred = int(feature_size is None)
        feature_info_fns = (
            lambda: (to_2tuple(feature_size), self.backbone.feature_info.channels()[-1]),
            lambda: self._infer_feature_info(in_chans),
        )
        feature_size, feature_dim = feature_info_fns[use_inferred]()
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def _infer_feature_info(self, in_chans):
        with torch.no_grad():
            training = self.backbone.training
            self.backbone.eval()
            o = self.backbone(torch.zeros(1, in_chans, self.img_size[0], self.img_size[1]))[-1]
            self.backbone.train(training)
        return o.shape[-2:], o.shape[1]

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x


class ViT(nn.Module):

    def __init__(self,
                 img_size=(224, 224), patch_size=16, in_chans=3, num_classes=80, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=None, use_checkpoint=False, 
                 frozen_stages=-1, ratio=1, last_norm=True,
                 patch_padding='pad', freeze_attn=False, freeze_ffn=False,
                 ):
        # Protect mutable default arguments
        super(ViT, self).__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.frozen_stages = frozen_stages
        self.use_checkpoint = use_checkpoint
        self.patch_padding = patch_padding
        self.freeze_attn = freeze_attn
        self.freeze_ffn = freeze_ffn
        self.depth = depth

        patch_embed_builders = (
            lambda: PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim, ratio=ratio
            ),
            lambda: HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim
            ),
        )
        self.patch_embed = patch_embed_builders[int(hybrid_backbone is not None)]()
        num_patches = self.patch_embed.num_patches

        # since the pretraining model has class token
        self.posi_embed = nn.Parameter(torch.zeros(1, 245 + 1, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                )
            for i in range(depth)])

        last_norm_layers = (nn.Identity(), norm_layer(embed_dim))
        self.last_norm = last_norm_layers[int(last_norm)]

        trunc_normal_(self.posi_embed, std=.02)

        self._freeze_stages()
        self.hand_kpe_linear = nn.Linear(embed_dim + 128, embed_dim)
        self.arm_kpe_linear = nn.Linear(embed_dim + 128, embed_dim)
        self.mask_embedding = nn.Parameter(torch.randn(1, embed_dim))

    def _noop(self):
        return None

    def _freeze_patch_embed(self):
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False

    def _freeze_attn_blocks(self):
        for i in range(0, self.depth):
            m = self.blocks[i]
            m.attn.eval()
            m.norm1.eval()
            for param in m.attn.parameters():
                param.requires_grad = False
            for param in m.norm1.parameters():
                param.requires_grad = False

    def _freeze_ffn_blocks(self):
        self.posi_embed.requires_grad = False
        self._freeze_patch_embed()
        for i in range(0, self.depth):
            m = self.blocks[i]
            m.mlp.eval()
            m.norm2.eval()
            for param in m.mlp.parameters():
                param.requires_grad = False
            for param in m.norm2.parameters():
                param.requires_grad = False

    def _freeze_stages(self):
        """Freeze parameters."""
        (self._noop, self._freeze_patch_embed)[int(self.frozen_stages >= 0)]()

        for i in range(1, self.frozen_stages + 1):
            m = self.blocks[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

        (self._noop, self._freeze_attn_blocks)[int(self.freeze_attn)]()
        (self._noop, self._freeze_ffn_blocks)[int(self.freeze_ffn)]()

    def init_weights(self):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        def _init_linear(m):
            trunc_normal_(m.weight, std=.02)
            bias_init = (
                lambda _: None,
                lambda b: nn.init.constant_(b, 0),
            )
            bias_init[int(m.bias is not None)](m.bias)

        def _init_layer_norm(m):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        init_dispatch = {
            nn.Linear: _init_linear,
            nn.LayerNorm: _init_layer_norm,
        }

        def _init_weights(m):
            init_dispatch.get(type(m), lambda _: None)(m)

        self.apply(_init_weights)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}


    def encode_hand(self, x, kpe):        
        B, T = x.shape[:2]

        norm_kpts = kpe[:, :, 0, :, :]
        dir = kpe[:, :, 2, :6, 0]
        kpe = kpe[:, :, 1, :5, :]

        x = x.view(B * T, *x.shape[2:])
        kpe = kpe.view(B * T, -1)
        norm_kpts = norm_kpts.view(B * T, -1)
        dir = dir.view(B * T, -1)

        kpe = sin_encode(kpe).view(B * T, -1)
        dir = sin_encode(dir).view(B * T, -1) 

        kpe = torch.cat([kpe, dir], dim=1)  # Concatenate along the last dimension

        features, (Hp, Wp) = self.patch_embed(x)  # Output shape: (batch_size, 2048, 7, 7)
        kpe_exp = kpe.unsqueeze(1).expand(-1, features.shape[1], -1)  # Expand kpe to match features shape
        
        features = self.hand_kpe_linear(torch.cat([features, kpe_exp], dim=-1)) + features

        centers = self.patch_embed.get_patch_centers(Hp, Wp)

        return features, kpe, centers

    def encode_arm(self, x, kpe):
        BxT = x.shape[0]
        x = F.interpolate(x, 
                          size=(224 // 2,  224 // 2), 
                          mode='bilinear', align_corners=False)  # Resize arm images to match hand images 

        norm_kpts = kpe[ :, 0, :, :]
        dir = kpe[:, 2, :6, 0]
        kpe = kpe[:, 1, :5, :]

        kpe = kpe.view(BxT, -1)
        norm_kpts = norm_kpts.view(BxT, -1)
        dir = dir.view(BxT, -1)

        kpe = sin_encode(kpe).view(BxT, -1)
        dir = sin_encode(dir).view(BxT, -1) 

        kpe = torch.cat([kpe, dir], dim=1) 

        features, (Hp, Wp) = self.patch_embed(x)  # Output shape: (batch_size, 2048, 7, 7)
        kpe_exp = kpe.unsqueeze(1).expand(-1, features.shape[1], -1)  # Expand kpe to match features shape
        
        features = self.arm_kpe_linear(torch.cat([features, kpe_exp], dim=-1)) + features

        H_small, W_small = x.shape[2], x.shape[3]
        y_scale = 224 / float(H_small)
        x_scale = 224 / float(W_small)

        centers = self.patch_embed.get_patch_centers(Hp, Wp) * torch.tensor([y_scale, x_scale])
        
        return features, kpe, centers

    def forward(self, x_hand, hand_kpe, x_arm, arm_kpe):
        x_hand, hand_kpe, hand_image_points = self.encode_hand(x_hand, hand_kpe)

        B, T = x_arm.shape[:2]; 
        BxT = B * T
        x_arm = x_arm.view(BxT, *x_arm.shape[2:])
        arm_kpe = arm_kpe.view(BxT, *arm_kpe.shape[2:])

        arm_visible = (x_arm.view(BxT, -1).sum(-1) != 0)

        x_arm_mask = self.mask_embedding.expand(BxT, -1, -1)
        x_arm_encoded, arm_kpe_encoded, arm_image_points = self.encode_arm(x_arm, arm_kpe)
        
        x_arm_out = torch.where(arm_visible.view(BxT, 1, 1), x_arm_encoded, x_arm_mask)
        arm_kpe_out = torch.where(arm_visible.view(BxT, -1), arm_kpe_encoded, arm_kpe_encoded.new_zeros(BxT, 128))

        x = torch.cat([x_hand, x_arm_out], dim=1) 

        x = x + self.posi_embed[:, 1:] + self.posi_embed[:, :1]

        for blk in self.blocks:
            x = blk(x)

        x = self.last_norm(x)

        # xp = x.permute(0, 2, 1)

        return x, hand_kpe, arm_kpe_out, arm_visible, (hand_image_points, arm_image_points)        

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()


class HALO(torch.nn.Module):
    def __init__(self, config):
        super(HALO, self).__init__()
        self.config = config
        self.pose_dim = (3, 6)[int(config.POSE_3D.ROT_6D)]

        hidden_dim = 1280
        hand_kpe_dim = 128
        arm_kpe_dim = 128
      
        self.model = ViT(
                img_size=(256, 192),
                patch_size=16,
                embed_dim=hidden_dim,
                depth=32,
                num_heads=16,
                ratio=1,
                use_checkpoint=False,
                mlp_ratio=4,
                qkv_bias=True,
                drop_path_rate=0.55,
            )

        self.num_hand_queries = 1 + 1 + 2  # hm + pose + shape 'n' transl 
        self.num_arm_queries = 1 + 1 + 2  # hm + pose + shape 'n' transl
        self.hand_q = torch.nn.Parameter(torch.randn(1, self.num_hand_queries, hidden_dim))  # shape: [1, 1, dim]
        self.arm_q = torch.nn.Parameter(torch.randn(1, self.num_arm_queries, hidden_dim))

        self.cross_context_decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8, batch_first=True), num_layers=2)
        
        img_size = (224, 224)
        self.hand_pose2d_hm_decoder = HeatMap2DRegressor(img_size, hidden_dim, 98, 21)        
        self.arm_pose2d_hm_decoder = HeatMap2DRegressor(img_size, hidden_dim, 98, 3)

        kpt_feat_dim = 640
        self.hand_decoder = HandQueryDecoder(hidden_dim + kpt_feat_dim, pose_local_dim=self.pose_dim * 15, pose_global_dim=self.pose_dim, num_iterations=config.POSE_3D.DECODER_ITRS)
        self.arm_decoder = ArmQueryDecoder(hidden_dim + kpt_feat_dim, pose_global_dim=self.pose_dim, num_iterations=config.POSE_3D.DECODER_ITRS)

        kpe_dim = 256
        self.compress_to_hidden_dim = nn.Sequential(
            nn.Linear(kpe_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.hand_kpe_encoder = nn.Linear(hand_kpe_dim, kpe_dim)
        self.arm_kpe_encoder = nn.Linear(arm_kpe_dim, kpe_dim)

        self.arm_generative_prior = ArmGenerativePrior(hidden_dim, 128, hidden_dim * self.num_arm_queries)

        self.hand_keypoint_features = HandKeypointQueryFE(160, kpt_feat_dim)
        self.arm_keypoint_features = ArmKeypointQueryFE(160, kpt_feat_dim)

    def forward(self, x_hand, hand_kpe, x_arm, arm_kpe):
        features, hand_kpe, arm_kpe, arm_visible, _ = self.model(x_hand, hand_kpe, x_arm, arm_kpe)
        arm_missing = ~arm_visible

        BxT = features.shape[0]    

        hand_q_exp = self.hand_q.repeat(BxT, 1, 1)
        arm_q_exp = self.arm_q.repeat(BxT, 1, 1)  

        tgt_key_padding_mask = torch.cat([
            features.new_zeros(BxT, self.num_hand_queries, dtype=torch.bool),  # Hand query always valid
            arm_missing.unsqueeze(-1).repeat(1, self.num_arm_queries) # mask arm token if not visible
        ], dim=1)


        # Stack the two queries into a single tensor along the sequence dimension.
        # Final shape: [B*T, 2, hidden_dim]. The first token corresponds to the hand and the second to the arm.
        queries = torch.cat([hand_q_exp, arm_q_exp], dim=1)

        # Pass through the Transformer decoder.
        # The decoder first computes self-attention on the query sequence (allowing hand and arm queries to interact),
        # and then attends to the external features ("memory").
        decoded = self.cross_context_decoder(tgt=queries, memory=features, tgt_key_padding_mask=tgt_key_padding_mask)
        # decoded has shape [B*T, 2, hidden_dim]

        # Separate the decoded queries back into hand and arm representations.
        hand_q_feats = decoded[:, :self.num_hand_queries, :]   # shape: [B*T, hidden_dim]
        arm_q_feats = decoded[:, self.num_hand_queries:, :]    # shape: [B*T, hidden_dim]
        
        hand_features = torch.cat([hand_q_feats, self.hand_kpe_encoder(hand_kpe).unsqueeze(1).repeat(1, self.num_hand_queries, 1)], dim=-1)
        arm_features = torch.cat([arm_q_feats, self.arm_kpe_encoder(arm_kpe).unsqueeze(1).repeat(1, self.num_arm_queries, 1)], dim=-1)
        
        hand_features = hand_q_feats + self.compress_to_hidden_dim(hand_features)
        arm_features = arm_q_feats + self.compress_to_hidden_dim(arm_features)
        
        arm_features_prior, mu_prior, logvar_prior = self.arm_generative_prior(hand_features)
        
        arm_features = torch.where(arm_visible.view(BxT, 1, 1), arm_features, arm_features_prior.view(BxT, self.num_arm_queries, -1))
        
        B, T = x_hand.shape[:2]

        hand_kpts_2d, hand_hm, hand_kpt_w, hand_kpt_feats = self.hand_pose2d_hm_decoder(features[:, :14*14, :], hand_features[:, 0], size=(B, T))
        arm_kpts_2d, arm_hm, arm_kpt_w, arm_kpt_feats = self.arm_pose2d_hm_decoder(features[:, 14*14:, :], arm_features[:, 0], size=(B, T))

        hand_kpt_feats = self.hand_keypoint_features(hand_kpt_feats).expand(-1, self.num_hand_queries, -1)
        arm_kpt_feats = self.arm_keypoint_features(arm_kpt_feats).expand(-1, self.num_arm_queries, -1)

        hand_features = torch.cat([hand_features, hand_kpt_feats], dim=-1)
        arm_features = torch.cat([arm_features, arm_kpt_feats], dim=-1)

        shape, pose_global, pose_local = self.hand_decoder(hand_features, size=(B, T))
        arm_shape, arm_R = self.arm_decoder(arm_features, size=(B, T))

        output = {
            'betas': shape,
            'global_orient': pose_global,
            'hand_pose': pose_local,

            'arm_shape': arm_shape,
            'arm_R': arm_R,

            'hand_hms': hand_hm,
            'arm_hms': arm_hm,

            'hand_kpts_2d': hand_kpts_2d,
            'arm_kpts_2d': arm_kpts_2d,

            'hand_kpt_w': hand_kpt_w,
            'arm_kpt_w': arm_kpt_w,
            
        }

        return output


class HALOAblations(HALO):
    def __init__(self, config, *args, use_cit: bool = True, use_arm_prior: bool = True, use_arm_input: bool = True, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.use_cit = use_cit
        self.use_arm_prior = use_arm_prior
        self.use_arm_input = use_arm_input

    # ------------------------------------------------------------------
    def forward(self, x_hand, hand_kpe, x_arm, arm_kpe):
        # ── Ablation: CIT disabled ─────────────────────────────────────
        # Zero both KPE tensors before the ViT backbone so no conditional-
        # input information flows anywhere in the network.
        if not self.use_cit:
            hand_kpe = torch.zeros_like(hand_kpe)
            arm_kpe  = torch.zeros_like(arm_kpe)
        # ──────────────────────────────────────────────────────────────

        # ── Ablation: ARM input disabled ─────────────────────────────────────
        # Zero arm image and arm KPE before the ViT backbone, simulating the case where no arm input is available at all.
        if not self.use_arm_input:
            x_arm = torch.zeros_like(x_arm)
            arm_kpe  = torch.zeros_like(arm_kpe)
        # ──────────────────────────────────────────────────────────────


        features, hand_kpe, arm_kpe, arm_visible, _ = self.model(x_hand, hand_kpe, x_arm, arm_kpe)
        arm_missing = ~arm_visible

        BxT = features.shape[0]

        hand_q_exp = self.hand_q.repeat(BxT, 1, 1)
        arm_q_exp  = self.arm_q.repeat(BxT, 1, 1)

        tgt_key_padding_mask = torch.cat([
            features.new_zeros(BxT, self.num_hand_queries, dtype=torch.bool),
            arm_missing.unsqueeze(-1).repeat(1, self.num_arm_queries),
        ], dim=1)

        queries = torch.cat([hand_q_exp, arm_q_exp], dim=1)
        decoded = self.cross_context_decoder(
            tgt=queries, memory=features, tgt_key_padding_mask=tgt_key_padding_mask
        )

        hand_q_feats = decoded[:, :self.num_hand_queries, :]
        arm_q_feats  = decoded[:, self.num_hand_queries:,  :]

        hand_features = torch.cat(
            [hand_q_feats, self.hand_kpe_encoder(hand_kpe).unsqueeze(1).repeat(1, self.num_hand_queries, 1)],
            dim=-1,
        )
        arm_features = torch.cat(
            [arm_q_feats, self.arm_kpe_encoder(arm_kpe).unsqueeze(1).repeat(1, self.num_arm_queries, 1)],
            dim=-1,
        )

        hand_features = hand_q_feats + self.compress_to_hidden_dim(hand_features)
        arm_features  = arm_q_feats  + self.compress_to_hidden_dim(arm_features)

        # ── Ablation: arm prior disabled ───────────────────────────────
        # When use_arm_prior=True (default) the prior fills arm features
        # for invisible-arm frames, matching HALO exactly.
        # When False, arm_features is kept as-is regardless of visibility.
        if self.use_arm_prior:
            arm_features_prior, mu_prior, logvar_prior = self.arm_generative_prior(hand_features)

            arm_features = torch.where(
                arm_visible.view(BxT, 1, 1),
                arm_features,
                arm_features_prior.view(BxT, self.num_arm_queries, -1),
            )
        # ──────────────────────────────────────────────────────────────

        B, T = x_hand.shape[:2]

        hand_kpts_2d, hand_hm, hand_kpt_w, hand_kpt_feats = self.hand_pose2d_hm_decoder(
            features[:, :14*14, :], hand_features[:, 0], size=(B, T)
        )
        arm_kpts_2d, arm_hm, arm_kpt_w, arm_kpt_feats = self.arm_pose2d_hm_decoder(
            features[:, 14*14:, :], arm_features[:, 0], size=(B, T)
        )

        hand_kpt_feats = self.hand_keypoint_features(hand_kpt_feats).expand(-1, self.num_hand_queries, -1)
        arm_kpt_feats  = self.arm_keypoint_features(arm_kpt_feats).expand(-1, self.num_arm_queries,  -1)

        hand_features = torch.cat([hand_features, hand_kpt_feats], dim=-1)
        arm_features  = torch.cat([arm_features,  arm_kpt_feats],  dim=-1)

        shape, pose_global, pose_local = self.hand_decoder(hand_features, size=(B, T))
        arm_shape, arm_R               = self.arm_decoder(arm_features,   size=(B, T))

        return {
            'betas':         shape,
            'global_orient': pose_global,
            'hand_pose':     pose_local,
            'arm_shape':     arm_shape,
            'arm_R':         arm_R,
            'hand_hms':      hand_hm,
            'arm_hms':       arm_hm,
            'hand_kpts_2d':  hand_kpts_2d,
            'arm_kpts_2d':   arm_kpts_2d,
            'hand_kpt_w':    hand_kpt_w,
            'arm_kpt_w':     arm_kpt_w,
        }
