from types import SimpleNamespace
from typing import Optional, Tuple

import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ----------------------------- DGP core -----------------------------

def _construct_AB(kpts_2d_norm: torch.Tensor, kpts_3d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build A, B such that min ||A t - B||_2, where t = (tx, ty, tz).
    For each point i with normalized 2D (u_i, v_i) and 3D (X_i, Y_i, Z_i):
        -tx + u_i * tz = X_i - Z_i * u_i
        -ty + v_i * tz = Y_i - Z_i * v_i
    Stacks as (2N x 3) system per batch.

    Args:
        kpts_2d_norm: (B, N, 2) normalized by K^{-1} (i.e., on the z=1 plane)
        kpts_3d:      (B, N, 3)

    Returns:
        A: (B, 2N, 3), B: (B, 2N, 1)
    """
    Bsz, N = kpts_2d_norm.shape[:2]
    A = kpts_2d_norm.new_zeros(Bsz, 2 * N, 3)
    Bm = kpts_2d_norm.new_zeros(Bsz, 2 * N, 1)

    u = kpts_2d_norm[..., 0]
    v = kpts_2d_norm[..., 1]
    X = kpts_3d[..., 0]
    Y = kpts_3d[..., 1]
    Z = kpts_3d[..., 2]

    # Even rows (x-equations)
    A[:, 0::2, 0] = -1.0                    # -tx
    A[:, 0::2, 2] = u                        # + u * tz
    Bm[:, 0::2, 0] = X - Z * u               # rhs

    # Odd rows (y-equations)
    A[:, 1::2, 1] = -1.0                    # -ty
    A[:, 1::2, 2] = v                        # + v * tz
    Bm[:, 1::2, 0] = Y - Z * v               # rhs

    return A, Bm


def _normalize_2d_by_K(kpts_2d_img: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """
    Normalize pixel 2D by K^{-1} to (u,v,1) on the z=1 plane.

    Args:
        kpts_2d_img: (B, N, 2) image-space (pixels) matching K.
        K:           (B, 3, 3) intrinsics per batch.

    Returns:
        (B, N, 2) normalized coordinates.
    """
    Bsz, N = kpts_2d_img.shape[:2]
    ones = torch.ones(Bsz, N, 1, device=kpts_2d_img.device, dtype=kpts_2d_img.dtype)
    hom = torch.cat([kpts_2d_img, ones], dim=-1)                         # (B,N,3)
    Kinv = torch.linalg.pinv(K)                                          # (B,3,3)
    norm = torch.bmm(Kinv, hom.transpose(1, 2)).transpose(1, 2)          # (B,N,3)
    # divide by the 3rd component is unnecessary for ideal pinhole (it's 1), but keep safe:
    norm_xy = norm[..., :2] / (norm[..., 2:3].clamp_min(1e-8))
    return norm_xy


def dgp_weighted_ls(
    kpts_3d: torch.Tensor,          # (B,N,3)
    kpts_2d_img: torch.Tensor,      # (B,N,2)
    weights: torch.Tensor,          # (B,N)   confidences in [0,1]
    K: torch.Tensor,                # (B,3,3)
) -> torch.Tensor:
    """
    Differentiable Global Positioning (translation-only) via **weighted** least squares:
      argmin_t || W^{1/2} (A t - B) ||_2

    Returns:
        transl: (B, 3)  (tx, ty, tz)
    """
    # 2D -> normalized
    k2d_norm = _normalize_2d_by_K(kpts_2d_img, K)                         # (B,N,2)
    A, Bm = _construct_AB(k2d_norm, kpts_3d)                              # (B,2N,3), (B,2N,1)

    # Build per-row sqrt weights efficiently (no huge diag matrices)
    # Each point contributes 2 rows => repeat weights along rows
    w_rows = torch.repeat_interleave(weights.clamp_min(0.0).sqrt(), 2, dim=-1)  # (B,2N)
    w_rows = w_rows.unsqueeze(-1)                                          # (B,2N,1)

    A_w = A * w_rows                                                       # (B,2N,3)
    B_w = Bm * w_rows                                                      # (B,2N,1)

    # Solve min ||A_w t - B_w|| via batched lstsq
    try:
        sol = torch.linalg.lstsq(A_w, B_w, driver=None).solution          # (B,3,1)
    except RuntimeError:
        # Normal equations fallback with Tikhonov reg (vectorized)
        AT = A_w.transpose(1, 2)                                          # (B,3,2N)
        ATA = torch.bmm(AT, A_w)                                          # (B,3,3)
        ATB = torch.bmm(AT, B_w)                                          # (B,3,1)
        # λ ~ 1e-6 * trace(ATA)/3 to stabilize ill-conditioned cases
        lam = (ATA.diagonal(dim1=1, dim2=2).sum(-1, keepdim=True) / 3.0) * 1e-6
        I = torch.eye(3, device=ATA.device, dtype=ATA.dtype).expand_as(ATA)
        sol = torch.linalg.solve(ATA + lam.unsqueeze(-1) * I, ATB)        # (B,3,1)

    return sol.squeeze(-1)                                                # (B,3)


# ---------------------- Robust gating for refinement ----------------------

def _reproj_residuals(
    kpts_3d: torch.Tensor, kpts_2d_img: torch.Tensor, K: torch.Tensor, t: torch.Tensor
) -> torch.Tensor:
    """
    Residuals of the linearized DGP equations after applying translation t.

    For each i:
        r_x = (X+tx) - (Z+tz) * u
        r_y = (Y+ty) - (Z+tz) * v
    where (u,v) = K^{-1} [x,y,1]^T.

    Args:
        kpts_3d:     (B,N,3) (already in current frame)
        kpts_2d_img: (B,N,2)
        K:           (B,3,3)
        t:           (B,3)   (tx,ty,tz)

    Returns:
        r: (B,N,2)
    """
    u_v = _normalize_2d_by_K(kpts_2d_img, K)                              # (B,N,2)
    X = kpts_3d[..., 0] + t[:, None, 0]
    Y = kpts_3d[..., 1] + t[:, None, 1]
    Z = kpts_3d[..., 2] + t[:, None, 2]
    r_x = X - Z * u_v[..., 0]
    r_y = Y - Z * u_v[..., 1]
    return torch.stack([r_x, r_y], dim=-1)                                # (B,N,2)


def _robust_loss(
    r: torch.Tensor, weights: torch.Tensor, kind: str = "tukey", c: float = 4.685, eps: float = 1e-12
) -> torch.Tensor:
    """
    Robust per-batch loss on residuals.

    Args:
        r:       (B,N,2)
        weights: (B,N) in [0,1]
        kind:    'tukey' or 'l2'
        c:       Tukey constant (in MAD units)

    Returns:
        (B,) loss
    """
    w = weights.clamp_min(0.0)
    rn = (r.square().sum(-1)).sqrt().clamp_min(eps)                        # (B,N) L2 per point

    if kind == "l2":
        loss = (w * rn.square()).sum(-1)
        return loss

    # Tukey biweight on standardized residuals
    # Robust scale: MAD per batch
    med = rn.median(dim=-1, keepdim=True).values                           # (B,1)
    mad = (rn - med).abs().median(dim=-1, keepdim=True).values.clamp_min(1e-6)
    z = rn / (c * 1.4826 * mad)                                            # standardized

    inside = (z < 1.0).float()
    t = (1 - z.square()).clamp_min(0.0)
    rho = (c ** 2 / 6.0) * (1 - t.pow(3))                                  # Tukey rho
    rho = rho * inside + (c ** 2 / 6.0) * (~inside.bool()).float()         # saturate outside
    loss = (w * rho).sum(-1)                                               # (B,)
    return loss


def _clamp_delta(delta: torch.Tensor, max_norm: Optional[float]) -> torch.Tensor:
    if (max_norm is None) or (max_norm <= 0):
        return delta
    n = delta.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = torch.minimum(torch.ones_like(n), (max_norm / n))
    return delta * scale


# ---------------------- Integrated camera-space solver ----------------------

def compute_camera_space_dgp(
    cfg,
    meta: dict,
    hand_j3d,
    arm_j3d  ,
    hand_v  ,
    arm_v    ,
    hand_2d ,
    arm_2d   ,
    w_hand ,
    w_arm  ,

    *,
    K: torch.Tensor,                          # (B,3,3) intrinsics matching crop_j2d coords
    prefer_hand: bool = True,
    bias: Optional[torch.Tensor] = None,      # (3,)
    refine_secondary: bool = True,
    min_visible: int = 6,
    delta_clamp: float = 0.02,                # meters (2 cm)
    robust_kind: str = "tukey",
    robust_c: float = 4.685,
    accept_only_if_loss_drops: bool = True,
) -> SimpleNamespace:
    """
    Camera-space placement using DGP (translation-only) with:
      • Primary pass on chosen limb (hand by default).
      • Optional guarded secondary refinement on the other limb (accept if robust loss drops).

    Inputs:
      - limb_output.{hand,arm}.joints:   (B, H/A, 3)
      - limb_output.{hand,arm}.vertices: (B, Vh/Va, 3)
      - limb_output.{hand,arm}.crop_j2d: (B, H/A, 2)  pixel coords in crop frame
      - limb_output.{hand,arm}.confidence: (B, H/A)   [0,1]
      - K: intrinsics for that same crop frame, per-batch (B,3,3)

    Returns:
      SimpleNamespace with .hand/.arm new vertices/joints and .transl (B,1,3)
    """
    # Unpack
    Bsz = hand_j3d.shape[0]
    device, dtype = hand_j3d.device, hand_j3d.dtype

    # Choose primary limb per batch (match your heuristic)
    sum_hand = (w_hand > 0).float().sum(dim=1)
    sum_arm  = (w_arm  > 0).float().sum(dim=1)
    choose_hand_first = prefer_hand | (sum_hand >= 0.25 * (sum_arm + 1e-8))

    idx_hand_first = choose_hand_first.nonzero(as_tuple=True)[0]
    idx_arm_first  = (~choose_hand_first).nonzero(as_tuple=True)[0]

    transl = torch.zeros(Bsz, 3, device=device, dtype=dtype)

    # ---------- primary pass (DGP WLS) ----------
    if idx_hand_first.numel() > 0:
        vis_ok = (sum_hand[idx_hand_first] >= min_visible)
        if vis_ok.any():
            i = idx_hand_first[vis_ok]
            t = dgp_weighted_ls(hand_j3d[i], hand_2d[i], w_hand[i], K[i])
            transl[i] = t

    if idx_arm_first.numel() > 0:
        vis_ok = (sum_arm[idx_arm_first] >= min_visible)
        if vis_ok.any():
            i = idx_arm_first[vis_ok]
            t = dgp_weighted_ls(arm_j3d[i], arm_2d[i], w_arm[i], K[i])
            transl[i] = t

    if bias is not None:
        transl = transl + bias.view(1, 3)

    # Apply
    transl_ = transl.view(Bsz, 1, 3)
    hand_j3d_w = hand_j3d + transl_
    arm_j3d_w  = arm_j3d  + transl_
    hand_v_w   = hand_v   + transl_
    arm_v_w    = arm_v    + transl_

    # ---------- secondary refinement (guarded Δt) ----------
    if refine_secondary and (idx_hand_first.numel() + idx_arm_first.numel() > 0):
        idx_secondary = idx_arm_first if idx_hand_first.numel() else idx_hand_first
        if idx_secondary.numel() > 0:
            # Secondary limb per those batches: if primary was hand, secondary is arm, and vice-versa
            primary_was_hand = (idx_secondary is idx_arm_first)  # True means hand was primary
            if primary_was_hand:
                # hand primary -> refine with arm
                j3d_sec = arm_j3d_w
                k2d_sec = arm_2d
                w_sec   = w_arm
            else:
                # arm primary -> refine with hand
                j3d_sec = hand_j3d_w
                k2d_sec = hand_2d
                w_sec   = w_hand

            i = idx_secondary

            # Compute Δt using DGP on *already translated* 3D -> this yields a residual update
            delta = dgp_weighted_ls(j3d_sec[i], k2d_sec[i], w_sec[i], K[i])  # (G,3)
            delta = _clamp_delta(delta, delta_clamp)

            # Robust gate: accept only if loss drops
            if accept_only_if_loss_drops:
                r0 = _reproj_residuals(j3d_sec[i], k2d_sec[i], K[i], t=torch.zeros_like(delta))
                r1 = _reproj_residuals(j3d_sec[i], k2d_sec[i], K[i], t=delta)
                L0 = _robust_loss(r0, w_sec[i], kind=robust_kind, c=robust_c)
                L1 = _robust_loss(r1, w_sec[i], kind=robust_kind, c=robust_c)
                improved = (L1 < L0)

                if improved.any():
                    good_idx = i[improved]
                    d_good = delta[improved].view(-1, 3)
                    transl[good_idx] = transl[good_idx] + d_good

                    # update all outputs with Δt
                    d_full = d_good.view(-1, 1, 3)
                    hand_j3d_w[good_idx] = hand_j3d_w[good_idx] + d_full
                    arm_j3d_w[good_idx]  = arm_j3d_w[good_idx]  + d_full
                    hand_v_w[good_idx]   = hand_v_w[good_idx]   + d_full
                    arm_v_w[good_idx]    = arm_v_w[good_idx]    + d_full
            else:
                # unconditional apply
                d_full = delta.view(-1, 1, 3)
                transl[i] = transl[i] + delta
                hand_j3d_w[i] = hand_j3d_w[i] + d_full
                arm_j3d_w[i]  = arm_j3d_w[i]  + d_full
                hand_v_w[i]   = hand_v_w[i]   + d_full
                arm_v_w[i]    = arm_v_w[i]    + d_full

    def _project_pinhole(X, K):
        x = torch.bmm(K, X.transpose(1, 2))         # (B,3,N)
        z = x[:, 2:3, :].clamp_min(1e-8)
        uv = (x[:, :2, :] / z).transpose(1, 2)      # (B,N,2)
        return uv


    hand_j2d = _project_pinhole(hand_j3d_w, K)
    arm_j2d  = _project_pinhole(arm_j3d_w,  K)

    return SimpleNamespace(
        hand=SimpleNamespace(vertices=hand_v_w, joints=hand_j3d_w, j2d=hand_j2d),
        arm= SimpleNamespace(vertices=arm_v_w,  joints=arm_j3d_w,  j2d=arm_j2d),
        transl=transl.view(Bsz, 1, 3),
    )


class DGPModelWrapper:
    def __init__(self):
        ...

    def __call__(self, left_batch, right_batch, left_outs, right_outs):
        data, meta = left_batch

        for outs, hand_type in [(left_outs, 'left'), (right_outs, 'right')]:
            if hand_type == 'left':
                pred_hand_j2d = left_outs[f'pred_hand_j2d']
                pred_hand_j3d = left_outs[f'pred_hand_j3d']
                pred_hand_vertices = left_outs[f'pred_hand_vertices']

                pred_arm_j2d = left_outs[f'pred_arm_j2d']
                pred_arm_j3d = left_outs[f'pred_arm_j3d']
                pred_arm_vertices = left_outs[f'pred_arm_vertices']

            else:
                pred_hand_j2d = right_outs[f'pred_hand_j2d']
                pred_hand_j3d = right_outs[f'pred_hand_j3d']
                pred_hand_vertices = right_outs[f'pred_hand_vertices']

                pred_arm_j2d = right_outs[f'pred_arm_j2d']
                pred_arm_j3d = right_outs[f'pred_arm_j3d']
                pred_arm_vertices = right_outs[f'pred_arm_vertices']

            pred_hand_j2d = torch.tensor(pred_hand_j2d).to(device)
            pred_hand_j3d = torch.tensor(pred_hand_j3d).to(device)
            pred_hand_vertices = torch.tensor(pred_hand_vertices).to(device)

            pred_arm_j2d = torch.tensor(pred_arm_j2d).to(device)
            pred_arm_j3d = torch.tensor(pred_arm_j3d).to(device)
            pred_arm_vertices = torch.tensor(pred_arm_vertices).to(device)

            focal_length = meta['focal_length']
            principal_point = meta['principal_point']

            K = torch.zeros((pred_hand_j2d.shape[0], 3,3), device=pred_hand_j2d.device, dtype=pred_hand_j2d.dtype)
            K[:,0,0] = focal_length[:,0]
            K[:,1,1] = focal_length[:,1]
            K[:,0,2] = principal_point[:,0]
            K[:,1,2] = principal_point[:,1]
            K[:,2,2] = 1.0  

            w_hand = torch.ones((pred_hand_j2d.shape[0], pred_hand_j2d.shape[1]), device=pred_hand_j2d.device, dtype=pred_hand_j2d.dtype)
            w_arm = torch.ones((pred_arm_j2d.shape[0], pred_arm_j2d.shape[1]), device=pred_arm_j2d.device, dtype=pred_arm_j2d.dtype)        

            dgp_out = compute_camera_space_dgp(
                cfg=None,
                meta=meta,
                hand_j3d=pred_hand_j3d,
                arm_j3d=pred_arm_j3d,       
                hand_v=pred_hand_vertices,
                arm_v=pred_arm_vertices,
                hand_2d=pred_hand_j2d,
                arm_2d=pred_arm_j2d,
                w_hand=w_hand,
                w_arm=w_arm,
                K=K,
                prefer_hand=True,)


            if hand_type == 'left':
                left_outs[f'pred_hand_j3d'] = dgp_out.hand.joints.cpu().numpy()
                left_outs[f'pred_hand_vertices'] = dgp_out.hand.vertices.cpu().numpy()
                left_outs[f'pred_hand_j2d'] = dgp_out.hand.j2d.cpu().numpy()
                
                left_outs[f'pred_arm_j3d'] = dgp_out.arm.joints.cpu().numpy()
                left_outs[f'pred_arm_vertices'] = dgp_out.arm.vertices.cpu().numpy()
                left_outs[f'pred_arm_j2d'] = dgp_out.arm.j2d.cpu().numpy()
            else:
                right_outs[f'pred_hand_j3d'] = dgp_out.hand.joints.cpu().numpy()
                right_outs[f'pred_hand_vertices'] = dgp_out.hand.vertices.cpu().numpy()   
                right_outs[f'pred_hand_j2d'] = dgp_out.hand.j2d.cpu().numpy()
                
                right_outs[f'pred_arm_j3d'] = dgp_out.arm.joints.cpu().numpy()
                right_outs[f'pred_arm_vertices'] = dgp_out.arm.vertices.cpu().numpy()
                right_outs[f'pred_arm_j2d'] = dgp_out.arm.j2d.cpu().numpy()

        return left_outs, right_outs