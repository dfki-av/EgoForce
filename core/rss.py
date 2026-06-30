import torch
from typing import Optional, Literal
from types import SimpleNamespace


def _require_pytorch3d_camera(module_name, class_name):
    try:
        module = __import__(module_name, fromlist=[class_name])
    except ImportError as exc:
        raise ImportError(
            f"{class_name} requires PyTorch3D. Install PyTorch3D or use "
            "pinhole camera intrinsics for the lightweight Visora path."
        ) from exc
    return getattr(module, class_name)


def unproject_unit_rays(cfg, meta, j2d):
    device = j2d.device

    hand_bbox = meta['hand_bbox'].to(device)
    hand_crop_size = meta['hand_crop_size'].to(device)

    arm_bbox = meta['arm_bbox'].to(device)
    arm_crop_size = meta['arm_crop_size'].to(device)

    focal_length = meta['focal_length'].to(device)
    principal_point = meta['principal_point'].to(device)
    projection_params = meta['projection_params'].to(device)
    image_size = meta['org_img_size'].to(device)
    camera_type = meta['camera_type'].to(device)

    if len(focal_length.shape) == 2:
        B = focal_length.shape[0]
    elif len(focal_length.shape) == 3:
        B, T = focal_length.shape[:2]
        B = B * T

    focal_length = focal_length.view(B, 2)
    principal_point = principal_point.view(B, 2)
    projection_params = projection_params.view(B, -1)
    image_size = image_size.view(B, 2)
    camera_type = camera_type.view(B)

    hand_bbox = hand_bbox.view(B, 4)
    arm_bbox = arm_bbox.view(B, 4)
    hand_crop_size = hand_crop_size.view(B, 2)
    arm_crop_size = arm_crop_size.view(B, 2)

    image_size = image_size.flip(1) # flip to get (H, W)
    mask_pinhole = (camera_type == 0)
    mask_rational = (camera_type == 2)
    mask_fisheye_624  = (camera_type == 3)
    mask_fisheye_KB3  = (camera_type == 4)
    mask_equisolid = (camera_type == 5)
    mask_equirectangular = (camera_type == 6)
    mask_stereographic = (camera_type == 7)

    idx_pinhole = mask_pinhole.nonzero(as_tuple=True)[0]
    idx_rational = mask_rational.nonzero(as_tuple=True)[0]
    idx_fisheye_624  = mask_fisheye_624.nonzero(as_tuple=True)[0]
    idx_fisheye_KB3  = mask_fisheye_KB3.nonzero(as_tuple=True)[0]
    idx_equisolid = mask_equisolid.nonzero(as_tuple=True)[0]
    idx_equirectangular = mask_equirectangular.nonzero(as_tuple=True)[0]
    idx_stereographic = mask_stereographic.nonzero(as_tuple=True)[0]


    hcrop_size = hand_crop_size.unsqueeze(1)     
    acrop_size = arm_crop_size.unsqueeze(1)
    inp_w, inp_h = cfg.POSE_3D.IMAGE_SIZE 
    inp_size   = torch.tensor([inp_w, inp_h], device=device).view(1, 1, 2)

    uv_hand = (j2d[..., :21, :] / inp_size) * hcrop_size + hand_bbox[:, None, :2]
    uv_arm  = (j2d[..., 21:, :]  / inp_size) * acrop_size + arm_bbox[:, None, :2]
    uv = torch.cat([uv_hand, uv_arm], dim=1)

    uv_norm = (uv - principal_point[:,None,:]) / (focal_length[:,None,:] + 1e-8) # normalised pixel coords

    direction = torch.zeros(j2d.shape[0], j2d.shape[1], 3, device=device)
    if idx_pinhole.numel() > 0:
        uv_pinhole = uv_norm[idx_pinhole]
        one = torch.ones(
            *uv_pinhole.shape[:-1],
            1,
            device=uv_pinhole.device,
            dtype=uv_pinhole.dtype,
        )
        dir_pinhole = torch.cat([uv_pinhole, one], dim=-1)
        dir_pinhole = dir_pinhole / torch.linalg.norm(
            dir_pinhole,
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-12)
        direction[idx_pinhole] = dir_pinhole

    if idx_rational.numel() > 0:
        Rational8CameraPytorch3D = _require_pytorch3d_camera(
            'camera_models.rational8_pytorch3d',
            'Rational8CameraPytorch3D',
        )
        focal_rational = focal_length[idx_rational]
        principal_rational = principal_point[idx_rational]
        proj_params_rational = projection_params[idx_rational][:, :8]
        camera = Rational8CameraPytorch3D(focal_rational,
                                                principal_rational,
                                                proj_params_rational,
                                                image_size=image_size[idx_rational],
                                                device=device)
        dir_rational = camera.unproject(uv_norm[idx_rational])
        direction[idx_rational] = dir_rational

    if idx_fisheye_624.numel() > 0:
        FishEyeCamera624Pytorch3D = _require_pytorch3d_camera(
            'camera_models.fisheye624_pytorch3d',
            'FishEyeCamera624Pytorch3D',
        )
        focal_fisheye = focal_length[idx_fisheye_624]
        principal_fisheye = principal_point[idx_fisheye_624]

        proj_params_fisheye = projection_params[idx_fisheye_624][:, 3:]
        camera = FishEyeCamera624Pytorch3D(focal_fisheye,
                                                principal_fisheye,
                                                proj_params_fisheye,
                                                image_size=image_size[idx_fisheye_624],
                                                device=device)
        dir_fisheye_624 = camera.unproject(uv_norm[idx_fisheye_624])
        direction[idx_fisheye_624] = dir_fisheye_624

    if idx_fisheye_KB3.numel() > 0:
        KannalaBrandtK3CameraPytorch3D = _require_pytorch3d_camera(
            'camera_models.kannalabrandtk3_pytorch3d',
            'KannalaBrandtK3CameraPytorch3D',
        )
        focal_fisheye_kb3 = focal_length[idx_fisheye_KB3]
        principal_fisheye_kb3 = principal_point[idx_fisheye_KB3]
        
        proj_params_fisheye_kb3 = projection_params[idx_fisheye_KB3][:, :4]
        cam_fisheye_kb3 = KannalaBrandtK3CameraPytorch3D(focal_fisheye_kb3,
                                                         principal_fisheye_kb3,
                                                         proj_params_fisheye_kb3,
                                                         image_size=image_size[idx_fisheye_KB3],
                                                         device=device)
        dir_fisheye_kb3 = cam_fisheye_kb3.unproject(uv_norm[idx_fisheye_KB3])
        direction[idx_fisheye_KB3] = dir_fisheye_kb3

    if idx_equisolid.numel() > 0:
        EquisolidCameraPytorch3D = _require_pytorch3d_camera(
            'camera_models.equisolid_pytorch3d',
            'EquisolidCameraPytorch3D',
        )
        cam_equisolid = EquisolidCameraPytorch3D(
            focal_length[idx_equisolid],
            principal_point[idx_equisolid],
            image_size=image_size[idx_equisolid],
            device=device,
        )
        dir_equisolid = cam_equisolid.inverse_evaluate(uv_norm[idx_equisolid])
        direction[idx_equisolid] = dir_equisolid

    if idx_equirectangular.numel() > 0:
        EquirectangularCameraPytorch3D = _require_pytorch3d_camera(
            'camera_models.equirectangular_pytorch3d',
            'EquirectangularCameraPytorch3D',
        )
        cam_equirectangular = EquirectangularCameraPytorch3D(
            focal_length[idx_equirectangular],
            principal_point[idx_equirectangular],
            image_size=image_size[idx_equirectangular],
            device=device,
        )
        dir_equirectangular = cam_equirectangular.inverse_evaluate(uv_norm[idx_equirectangular])
        direction[idx_equirectangular] = dir_equirectangular

    if idx_stereographic.numel() > 0:
        StereographicCameraPytorch3D = _require_pytorch3d_camera(
            'camera_models.stereographic_pytorch3d',
            'StereographicCameraPytorch3D',
        )
        cam_stereographic = StereographicCameraPytorch3D(
            focal_length[idx_stereographic],
            principal_point[idx_stereographic],
            image_size=image_size[idx_stereographic],
            device=device,
        )
        dir_stereographic = cam_stereographic.inverse_evaluate(uv_norm[idx_stereographic])
        direction[idx_stereographic] = dir_stereographic

    direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True).clamp_min(1e-12)

    return direction


def _proj_from_dirs(norm_dir: torch.Tensor):
    """Return P_i = I - d_i d_i^T for all rays.
    norm_dir: (B,N,3) assumed L2-normalised.
    Returns: P (B,N,3,3).
    """
    B, N, _ = norm_dir.shape
    I3 = torch.eye(3, device=norm_dir.device, dtype=norm_dir.dtype).view(1,1,3,3)
    d  = norm_dir.unsqueeze(-1)                       # (B,N,3,1)
    return I3 - d @ d.transpose(-2, -1)              # (B,N,3,3)

def _rdls_residuals(P: torch.Tensor, J: torch.Tensor, t: torch.Tensor):
    """Per-ray residual vectors r_i = P_i (t + J_i).
    Returns r: (B,N,3) and rnorm: (B,N).
    """
    TJ = t + J                       # (B,1,3) + (B,N,3)
    r  = (P @ TJ.unsqueeze(-1)).squeeze(-1)          # (B,N,3)
    rnorm = torch.linalg.norm(r, dim=-1)             # (B,N)
    return r, rnorm


def _rdls_leverage(P: torch.Tensor, w: torch.Tensor, eps: float=1e-8):
    """Leverage scores ℓ_i = tr(M^{-1} w_i P_i) for RDLS.
    Returns (B,N) non-negative scores.
    """
    B, N, _, _ = P.shape
    eye = torch.eye(3, device=P.device, dtype=P.dtype).expand(B,3,3)
    w_exp = w.unsqueeze(-1).unsqueeze(-1)
    M = (w_exp * P).sum(dim=1)                       # (B,3,3)
    Minv = torch.linalg.inv(M + 1e-8 * eye)          # (B,3,3)
    # ℓ_i = tr(Minv @ (w_i P_i)) = sum( (Minv^T ⊙ (w_i P_i)) )
    MinvT = Minv.transpose(-2, -1).unsqueeze(1)      # (B,1,3,3)
    S = w_exp * P                                    # (B,N,3,3)
    prod = MinvT * S                                 # (B,N,3,3)
    l = prod.sum(dim=(-1,-2)).clamp(min=0)           # (B,N)
    return l


def _robust_weights_scalar(rnorm: torch.Tensor, kind: str='tukey', c: float=4.685):
    """IRLS weights w_i = psi(r)/r for scalar residual magnitudes.
    rnorm: (B,N) ≥ 0  in pixel units.
    Returns (B,N) in [0,1].
    """
    if kind == 'huber':
        w = torch.where(rnorm <= c, torch.ones_like(rnorm), c / (rnorm + 1e-12))
    else:  # Tukey biweight
        z = rnorm / (c + 1e-12)
        mask = (z < 1)
        w = torch.zeros_like(z)
        w[mask] = (1 - z[mask]**2)**2
    return w


def _rdls_solve(P: torch.Tensor, J: torch.Tensor, w: torch.Tensor, eps: float=1e-8):
    """Solve M t = -m with M = sum w_i P_i,  m = sum w_i P_i J_i.
    P: (B,N,3,3), J: (B,N,3), w: (B,N) or (B,N,1).
    Returns: t (B,1,3), M (B,3,3)
    """
    # ---- shape normalisation & sanity checks ----
    if w.dim() == 3:
        w = w.squeeze(-1)
    assert P.ndim == 4 and P.shape[-2:] == (3,3), f"P shape must be (B,N,3,3), got {P.shape}"
    assert J.shape[:2] == P.shape[:2], f"J/B,N mismatch {J.shape} vs {P.shape}"
    assert w.shape[:2] == P.shape[:2], f"w/B,N mismatch {w.shape} vs {P.shape}"

    B, N, _, _ = P.shape
    w_exp = w.unsqueeze(-1).unsqueeze(-1)            # (B,N,1,1)

    # aggregate
    M = (w_exp * P).sum(dim=1)                       # (B,3,3)
    PJ = (P @ J.unsqueeze(-1)).squeeze(-1)           # (B,N,3)
    m = (w.unsqueeze(-1) * PJ).sum(dim=1)            # (B,3)

    eye = torch.eye(3, device=P.device, dtype=P.dtype).view(1,3,3).expand_as(M)
    Mr = M + eps * eye                                # (B,3,3)
    t = torch.linalg.solve(Mr, -m).unsqueeze(1)      # (B,1,3)
    return t, M


def solve_translation_full_ray_robust(
        norm_dir: torch.Tensor,  # (B,N,3) unit rays
        j3d: torch.Tensor,       # (B,N,3) relative 3D joints
        weight: Optional[torch.Tensor] = None,  # (B,N) confidences or (B,N,1)
        *,
        eps: float = 1e-8,
        cond_thresh: float = 1e6,
        enable_leverage: bool = False,
        enable_irls: bool = False,
        irls_iters: int = 3,
        robust_kind: str = 'tukey',
        robust_c: float = 4.685,
        enable_prosac: bool = False,
        prosac_m: int = 3,
        prosac_M: int = 32,
        accept_only_if_loss_drops: bool = True,
        return_loss: bool = False,
        compare_to_zero: bool = False,
        delta_clamp: Optional[float] = None,
    ):
    """Robust RDLS translation with IRLS + leverage + optional PROSAC.

    If return_loss=True, returns (t, loss) with loss the robust sum over all
    rays. If compare_to_zero=True, the loss baseline is computed at t=0 (useful
    for secondary refinement where we solve for Δt).
    """
    # ---- shape normalisation ----
    assert norm_dir.ndim == 3 and norm_dir.shape[-1] == 3, f"norm_dir (B,N,3) expected, got {norm_dir.shape}"
    assert j3d.ndim == 3 and j3d.shape[-1] == 3, f"j3d (B,N,3) expected, got {j3d.shape}"
    B, N, _ = j3d.shape
    if weight is None:
        w = torch.ones(B, N, device=j3d.device, dtype=j3d.dtype)
    else:
        w = weight.squeeze(-1) if weight.dim() == 3 else weight
        w = w.to(j3d.dtype)
        assert w.shape == (B, N), f"weight must be (B,N), got {w.shape} vs (B={B},N={N})"

    P = _proj_from_dirs(norm_dir)                    # (B,N,3,3)
    # additional consistency checks
    assert P.shape[:2] == j3d.shape[:2], f"P/J mismatch {P.shape[:2]} vs {j3d.shape[:2]}"

    def robust_loss(t):
        # ensure t has shape (B,1,3)
        if t.ndim == 2:
            t_ = t.unsqueeze(1)
        else:
            t_ = t
        _, rnorm = _rdls_residuals(P, j3d, t_)       # (B,N)
        # Tukey/Huber robust penalty per-joint (scalar), weight by w
        if robust_kind == 'huber':
            a = rnorm.abs()
            small = (a <= robust_c)
            loss = torch.zeros_like(a)
            loss[small] = 0.5 * a[small]**2
            loss[~small] = robust_c * (a[~small] - 0.5 * robust_c)
        else:
            z = rnorm / (robust_c + 1e-12)
            mask = (z < 1)
            loss = torch.zeros_like(z)
            loss[mask] = (robust_c**2 / 6.0) * (1 - (1 - z[mask]**2)**3)
        return (loss * w).sum(dim=1)                # (B,)

    # ---------- leverage reweighting ----------
    if enable_leverage:
        l = _rdls_leverage(P, w)
        l = l / (l.mean(dim=1, keepdim=True) + 1e-8)
        w = (w * l).clamp(min=1e-4, max=10.0)

    # ---------- initial solve ----------
    t, M = _rdls_solve(P, j3d, w, eps)
    # condition number guard
    with torch.no_grad():
        sv = torch.linalg.svdvals(M)
        cond = sv[..., 0] / (sv[..., -1] + 1e-12)
    bad = cond >= cond_thresh
    if bad.any():
        # small damping on all rays
        w_bad = w[bad] * 0.5
        t_bad, _ = _rdls_solve(P[bad], j3d[bad], w_bad, eps)
        t = t.clone()
        t[bad] = t_bad

    base_loss = robust_loss(torch.zeros_like(t)) if compare_to_zero else robust_loss(t)

    # ---------- PROSAC pre-screen (optional) ----------
    if enable_prosac and N >= prosac_m:
        importance = w.clone()
        best_t = t.clone()
        best_loss = robust_loss(t)
        for b in range(B):
            imp = importance[b]
            order = torch.argsort(imp, descending=True)
            upto = min(prosac_M, N)
            for it in range(upto):
                k = min(N, max(prosac_m, it+1))
                # choose the top-k set, then take the best prosac_m from it
                topk = order[:k]
                sel = topk[:prosac_m]
                P_sub = P[b:b+1, sel]
                J_sub = j3d[b:b+1, sel]
                w_sub = w[b:b+1, sel]
                t_h, _ = _rdls_solve(P_sub, J_sub, w_sub, eps)
                l_h = robust_loss(t_h)[0]
                if l_h < best_loss[b]:
                    best_loss[b] = l_h
                    best_t[b] = t_h[0]
        if accept_only_if_loss_drops:
            improved = best_loss < robust_loss(t)
            t = torch.where(improved.unsqueeze(-1).unsqueeze(-1), best_t, t)
        else:
            t = best_t

    # ---------- IRLS polish (optional) ----------
    if enable_irls and irls_iters > 0:
        t_curr = t
        for _ in range(irls_iters):
            _, rnorm = _rdls_residuals(P, j3d, t_curr)
            wr = _robust_weights_scalar(rnorm, kind=robust_kind, c=robust_c)
            w_eff = (w * wr).clamp(min=1e-6)
            t_next, _ = _rdls_solve(P, j3d, w_eff, eps)
            if torch.max(torch.abs(t_next - t_curr)) < 1e-6:
                t_curr = t_next
                break
            t_curr = t_next
        t = t_curr

    if delta_clamp is not None:
        # n: (B,1)
        n = torch.linalg.norm(t.squeeze(1), dim=-1, keepdim=True)
        scale = torch.where(
            n > delta_clamp,
            (delta_clamp / (n + 1e-12)),
            torch.ones_like(n)
        ).unsqueeze(-1)                                   # (B,1,1)
        t = t * scale                                     # out-of-place

    final_loss = robust_loss(t)
    if return_loss:
        return t, final_loss, (base_loss if compare_to_zero else None)
    return t



def compute_camera_space_mesh(
    cfg,
    meta: dict,
    limb_output: SimpleNamespace,
    pred_type='hand',
    *,
    prefer_hand: bool = True,
    bias: Optional[torch.Tensor] = None,  # (3,) optional
    rdls_opts: Optional[dict] = None,
    refine_secondary: bool = True,
    min_visible: int = 6,
    min_visible_arm: int = 2,
    delta_clamp: float = 0.02,            # 2 cm safety clamp for Δt
    min_positive_z: float = 0.05,
):
    """Place hand & arm in camera space using robust translation solves.

    • Primary pass: robust RDLS on chosen limb (hand by default).
    • Secondary pass (optional): PROSAC+IRLS-guarded small residual solve on
      the other limb; accepted only if robust loss drops.
    """
    # joints / verts / 2D
    hand_j3d = limb_output.hand.joints
    arm_j3d  = limb_output.arm.joints
    hand_v   = limb_output.hand.vertices
    arm_v    = limb_output.arm.vertices
    hand_2d  = limb_output.hand.crop_j2d
    arm_2d   = limb_output.arm.crop_j2d

    j2d = torch.cat([hand_2d, arm_2d], dim=1)
    norm_dir = unproject_unit_rays(cfg, meta, j2d)         # (B,N,3)
    H = hand_j3d.size(1)
    hand_dir, arm_dir = norm_dir[:, :H], norm_dir[:, H:]

    # base weights from confidences
    weight = torch.cat([limb_output.hand.confidence, limb_output.arm.confidence], dim=1)
    w_hand, w_arm = weight[:, :H], weight[:, H:]

    sum_hand = (w_hand > 0).float().sum(1)
    sum_arm  = (w_arm  > 0).float().sum(1)

    if isinstance(prefer_hand, bool):
        prefer_hand_mask = torch.full_like(sum_hand, prefer_hand, dtype=torch.bool)
    else:
        prefer_hand_mask = prefer_hand.to(device=hand_j3d.device, dtype=torch.bool)

    choose_hand_first = prefer_hand_mask | (sum_hand >= 0.25 * (sum_arm + 1e-8))

    B = hand_j3d.size(0)
    device, dtype = hand_j3d.device, hand_j3d.dtype
    transl = torch.zeros(B, 1, 3, device=device, dtype=dtype)

    # RDLS options with sensible defaults
    _rdls = dict(
        enable_leverage=True,
        enable_irls=False, irls_iters=2, robust_kind='tukey', robust_c=4.685,
        enable_prosac=False, prosac_m=3, prosac_M=24, accept_only_if_loss_drops=True,
        delta_clamp=None,
    )
    if rdls_opts:
        _rdls.update(rdls_opts)

    # ---------- primary pass ----------
    idx_hand_first = choose_hand_first.nonzero(as_tuple=True)[0]
    idx_arm_first  = (~choose_hand_first).nonzero(as_tuple=True)[0]

    if idx_hand_first.numel() > 0:
        # guard on visibility
        vis_ok = (sum_hand[idx_hand_first] >= min_visible)
        if vis_ok.any():
            i = idx_hand_first[vis_ok]
            transl[i] = solve_translation_full_ray_robust(
                hand_dir[i], hand_j3d[i], weight=w_hand[i], **_rdls
            )
    if idx_arm_first.numel() > 0:
        vis_ok = (sum_arm[idx_arm_first] >= min_visible_arm)
        if vis_ok.any():
            i = idx_arm_first[vis_ok]
            transl[i] = solve_translation_full_ray_robust(
                arm_dir[i], arm_j3d[i], weight=w_arm[i], **_rdls
            )

    # If the hand-only primary solve lands behind or effectively on the camera
    # plane, projection becomes numerically explosive. Use visible forearm rays
    # as a guarded fallback; arm has only three keypoints, so it needs its own
    # lower visibility threshold.
    bad_depth = transl[:, 0, 2] <= min_positive_z
    arm_fallback = bad_depth & (sum_arm >= min_visible_arm)
    if arm_fallback.any():
        i = arm_fallback.nonzero(as_tuple=True)[0]
        transl[i] = solve_translation_full_ray_robust(
            arm_dir[i], arm_j3d[i], weight=w_arm[i], **_rdls
        )

    if bias is not None:
        transl = transl + bias.view(1,1,3)

    # apply
    hand_j3d_w = hand_j3d + transl
    arm_j3d_w  = arm_j3d  + transl
    hand_v_w   = hand_v   + transl
    arm_v_w    = arm_v    + transl

    # ---------- secondary refinement (guarded) ----------
    if refine_secondary:
        secondary_specs = (
            (idx_hand_first, arm_dir, arm_j3d_w, w_arm, sum_arm, min_visible_arm),
            (idx_arm_first, hand_dir, hand_j3d_w, w_hand, sum_hand, min_visible),
        )
        for idx_secondary, dir_sec, j3d_sec, w_sec, sum_sec, secondary_min_visible in secondary_specs:
            if idx_secondary.numel() == 0:
                continue

            visible_secondary = sum_sec[idx_secondary] >= secondary_min_visible
            if not visible_secondary.any():
                continue

            idx_secondary = idx_secondary[visible_secondary]
            t_res, loss_after, loss_base = solve_translation_full_ray_robust(
                dir_sec[idx_secondary], j3d_sec[idx_secondary], weight=w_sec[idx_secondary],
                compare_to_zero=True, return_loss=True,
                enable_prosac=True, prosac_m=3, prosac_M=24, accept_only_if_loss_drops=True,
                enable_irls=True, irls_iters=1, robust_kind='tukey', robust_c=4.685,
                enable_leverage=True, delta_clamp=delta_clamp,
            )

            improved = loss_after < loss_base
            if improved.any():
                good = idx_secondary[improved]
                delta = t_res[improved]

                delta_full = torch.zeros_like(transl)
                delta_full[good] = delta

                transl = transl + delta_full
                hand_j3d_w = hand_j3d_w + delta_full
                arm_j3d_w = arm_j3d_w + delta_full
                hand_v_w = hand_v_w + delta_full
                arm_v_w = arm_v_w + delta_full

    return SimpleNamespace(
        hand=SimpleNamespace(vertices=hand_v_w, joints=hand_j3d_w),
        arm=SimpleNamespace(vertices=arm_v_w,  joints=arm_j3d_w),
        transl=transl,
    )
