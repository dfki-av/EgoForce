import numpy as np
import trimesh
import pyrender
import cv2


# -- Per-hand PBR materials matching Unity URP Lit settings --
# Unity: Metallic=0, Smoothness=0.5 → pyrender roughness = 1 - smoothness = 0.5
_LEFT_HAND_MATERIAL = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[1.0, 0.047, 0.053, 1.0],   # bright red (Unity LeftHandMat)
    metallicFactor=0.0,
    roughnessFactor=0.5,
    alphaMode='OPAQUE',
)

_RIGHT_HAND_MATERIAL = pyrender.MetallicRoughnessMaterial(
    baseColorFactor=[0.047, 0.402, 1.0, 1.0],   # bright blue (Unity RightHandMat)
    metallicFactor=0.0,
    roughnessFactor=0.5,
    alphaMode='OPAQUE',
)


# Background: dark studio
_BG_COLOR = (0.12, 0.12, 0.14, 1.0)

# 3rd-person camera controls
TP_ZOOM = 2.5   # >1 zooms out, <1 zooms in (1.10 = 10% further)
TP_UP   = -1.0   # vertical lift as fraction of offset (0.05 = 5% up)
TP_SIDE = 0.0    # lateral shift as fraction of offset (+right, -left)
GROUND_REFRESH_DISTANCE_M = 1.0


class Renderer:
    """
    SIGGRAPH-style offscreen mesh renderer.

    Produces a standalone beauty-pass image (not overlaid) with:
      - Three-point lighting (key + fill + rim)
      - Per-hand PBR materials (left=red, right=blue) matching Unity URP Lit
      - Ground plane
      - Dark studio background
    """

    def __init__(self, meta):
        focal_length = meta['focal_length']
        principal_point = meta['principal_point']
        org_img_size = meta['org_img_size'].cpu().numpy().astype(np.int32)

        W, H = org_img_size
        fx, fy = focal_length.cpu().numpy()
        cx, cy = principal_point.cpu().numpy()

        self.W, self.H = int(W), int(H)
        self.renderer = pyrender.OffscreenRenderer(self.W, self.H)

        # ---- scene (rebuilt each frame; kept as template params) ----
        self.fx, self.fy = float(fx), float(fy)
        self.cx, self.cy = float(cx), float(cy)
        self._ground_mesh = None  # cached after first frame
        self._ground_anchor = None
        self._tp_pose = None       # 3rd-person camera pose, cached after first frame

    def _update_ground_plane(self, all_verts):
        centroid = np.mean(all_verts, axis=0)
        should_refresh = (
            self._ground_mesh is None
            or self._ground_anchor is None
            or np.linalg.norm(centroid - self._ground_anchor) > GROUND_REFRESH_DISTANCE_M
        )
        if not should_refresh:
            return

        ground_y = float(all_verts[:, 1].min()) - 0.1
        ground, ground_tex = _make_ground_plane(center_y=ground_y, size=5)
        tex = pyrender.Texture(source=ground_tex, source_channels='RGB')
        ground_mat = pyrender.MetallicRoughnessMaterial(
            baseColorTexture=tex,
            emissiveTexture=tex,
            emissiveFactor=[0.7, 0.7, 0.7],
            metallicFactor=0.0,
            roughnessFactor=1.0,
            alphaMode='OPAQUE',
        )
        self._ground_mesh = pyrender.Mesh.from_trimesh(
            ground, material=ground_mat, smooth=False,
        )
        self._ground_anchor = centroid

    def _build_scene(self, limb_meshes, camera_pose=None):
        """
        Assemble a full scene: camera, lights, per-hand meshes, ground plane.
        limb_meshes: list of (trimesh, material) tuples.
        camera_pose: 4x4 ndarray for the camera (default: identity = ego view).
        """
        if camera_pose is None:
            camera_pose = np.eye(4)

        scene = pyrender.Scene(
            bg_color=_BG_COLOR,
            ambient_light=(0.08, 0.08, 0.10),
        )

        # --- camera ---
        cam = pyrender.IntrinsicsCamera(self.fx, self.fy, self.cx, self.cy)
        scene.add(cam, pose=camera_pose)

        # --- add each hand/arm mesh with its own material ---
        for mesh, material in limb_meshes:
            verts = mesh.vertices.copy()
            verts[:, [1, 2]] *= -1
            render_mesh = trimesh.Trimesh(
                vertices=verts, faces=mesh.faces, process=False,
            )
            render_mesh.visual = trimesh.visual.TextureVisuals()
            pr_mesh = pyrender.Mesh.from_trimesh(render_mesh, material=material, smooth=True)
            scene.add(pr_mesh)

        # --- ground plane (rebuilt only after large scene translations) ---
        if self._ground_mesh is not None:
            scene.add(self._ground_mesh)

        # --- lighting rig (cinematic 5-light + camera headlight) ---
        intensity = 0.2
        # 1) Camera headlight — point light at the camera position
        #    Tracks the active viewpoint. Reduced intensity to avoid washout
        #    from close-range ego view (inverse-square falloff).
        camera_light = pyrender.PointLight(color=[1.0, 0.98, 0.95], intensity=intensity * 1.0)
        scene.add(camera_light, pose=camera_pose)  # moves with the active camera

        # 2) Key light — warm directional, upper-right (main modelling light)
        key = pyrender.DirectionalLight(color=[1.0, 0.95, 0.88], intensity=intensity * 1.0)
        key_pose = _look_at_matrix(eye=[0.5, -0.8, -0.3], target=[0, 0, 0.4])
        scene.add(key, pose=key_pose)

        # # 3) Fill light — cool directional, upper-left (lifts shadows)
        fill = pyrender.DirectionalLight(color=[0.70, 0.80, 1.0], intensity=intensity * 1.5)
        fill_pose = _look_at_matrix(eye=[-0.6, -0.5, -0.2], target=[0, 0, 0.4])
        scene.add(fill, pose=fill_pose)

        # 4) Rim / back light — strong white edge highlight from behind
        rim = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity * 2.0)
        rim_pose = _look_at_matrix(eye=[0.0, -0.3, 1.0], target=[0, 0, 0.3])
        scene.add(rim, pose=rim_pose)

        # 7) Top-down kicker — subtle white from above for specular highlights
        top_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=intensity * 6.0)
        top_pose = _look_at_matrix(eye=[0.05, -1.0, 0.4], target=[0, 0, 0.4])
        scene.add(top_light, pose=top_pose)

        # 8) Back/3rd-person light — strong frontal light from the opposite side
        back_light = pyrender.DirectionalLight(color=[1.0, 0.95, 0.92], intensity=intensity * 6.0)
        back_pose = _look_at_matrix(eye=[0.0, -0.3, -0.5], target=[0, 0, 0.4])
        scene.add(back_light, pose=back_pose)

        return scene

    def _render_scene(self, scene):
        """Render a scene and return (H,W,3) uint8 RGB."""
        color, _ = self.renderer.render(
            scene,
            flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL | pyrender.RenderFlags.RGBA,
        )
        return color[:, :, :3].copy()

    def render(self, outs, limb_model, image, include_arm_mesh=False):
        """
        Build hand or hand+arm meshes, render two views:
          1) ego-camera (original intrinsics)
          2) 3rd-person view from the opposite side, slightly shifted

        Returns (ego_image, third_person_image), each (H, W, 3) uint8.
        """
        _MATERIALS = [_LEFT_HAND_MATERIAL, _RIGHT_HAND_MATERIAL]
        limb_meshes = []
        all_verts = []
        for hdx in range(2):
            hand_verts = outs['pred_vertices'][hdx]
            arm_verts = outs['pred_arm_vertices'][hdx]
            mesh = _build_limb_mesh(
                limb_model,
                hand_verts,
                arm_verts,
                hdx,
                include_arm_mesh=include_arm_mesh,
            )
            if mesh is not None:
                limb_meshes.append((mesh, _MATERIALS[hdx]))
                verts = mesh.vertices.copy()
                verts[:, [1, 2]] *= -1  # same flip as _build_scene
                all_verts.append(verts)

        blank = np.full((self.H, self.W, 3), 30, dtype=np.uint8)
        if not limb_meshes:
            return blank, blank

        all_verts = np.concatenate(all_verts, axis=0)
        self._update_ground_plane(all_verts)

        # --- ego view ---
        ego_scene = self._build_scene(limb_meshes)
        ego_image = self._render_scene(ego_scene)

        # --- 3rd-person view: camera facing from the opposite side (pose fixed on first frame) ---
        if self._tp_pose is None:
            centroid = np.mean(all_verts, axis=0)
            offset = np.array([0.1, -0.1, -0.35])
            offset *= TP_ZOOM
            norm = np.linalg.norm(offset)
            offset[1] -= TP_UP * norm
            offset[0] += TP_SIDE * norm
            tp_eye = centroid + offset
            self._tp_pose = _look_at_matrix(eye=tp_eye, target=centroid, up=(0, 1, 0))
        tp_scene = self._build_scene(limb_meshes, camera_pose=self._tp_pose)
        tp_image = self._render_scene(tp_scene)

        return ego_image, tp_image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_limb_mesh(limb_model, hand_vertices, arm_vertices, hdx, include_arm_mesh=False):
    """Create a single trimesh for the hand alone or the combined hand+arm mesh."""
    hand_faces = limb_model.faces.left_hand if hdx == 0 else limb_model.faces.right_hand

    if hasattr(hand_faces, 'cpu'):
        hand_faces = hand_faces.cpu().numpy()

    hand_faces = np.asarray(hand_faces, dtype=np.int64)

    if not include_arm_mesh:
        return trimesh.Trimesh(vertices=hand_vertices, faces=hand_faces, process=False)

    arm_faces = limb_model.faces.arm
    if hasattr(arm_faces, 'cpu'):
        arm_faces = arm_faces.cpu().numpy()
    arm_faces = np.asarray(arm_faces, dtype=np.int64)

    n_hand = hand_vertices.shape[0]
    vertices = np.concatenate([hand_vertices, arm_vertices], axis=0)
    faces = np.concatenate([hand_faces, arm_faces + n_hand], axis=0)

    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def _make_ground_plane(center_y, size=1.5):
    """
    Create a tiled ground plane at *center_y* with a procedural grid texture
    matching the Unity GridGround style (warm gray background + minor/major lines
    + soft vignette).
    """
    hs = size / 2.0
    vertices = np.array([
        [-hs, center_y, -hs],
        [ hs, center_y, -hs],
        [ hs, center_y,  hs],
        [-hs, center_y,  hs],
    ], dtype=np.float32)
    faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int64)

    # UV coordinates so the texture maps across the quad
    uvs = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [1.0, 1.0],
        [0.0, 1.0],
    ], dtype=np.float32)

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    # --- procedural grid texture (matches Unity GridGround settings) ---
    tex_res = 1024  # texture resolution (square)
    bg_color = np.array([230, 230, 225], dtype=np.uint8)         # warm light gray
    minor_color = np.array([189, 189, 184], dtype=np.uint8)      # thin gray lines
    major_color = np.array([140, 140, 140], dtype=np.uint8)      # darker gray lines

    # Grid cell counts across the texture
    minor_cells = 20       # 20 minor cells across the plane
    major_every = 4        # major line every 4 minor cells

    tex = np.full((tex_res, tex_res, 3), bg_color, dtype=np.uint8)
    minor_px = max(1, tex_res // (minor_cells * 64))   
    major_px = max(1, minor_px * 2)                   

    cell_size = tex_res / minor_cells
    for i in range(minor_cells + 1):
        pos = int(round(i * cell_size))
        is_major = (i % major_every == 0)
        color = major_color if is_major else minor_color
        thickness = major_px if is_major else minor_px
        lo = max(0, pos - thickness // 2)
        hi = min(tex_res, pos + (thickness + 1) // 2)
        tex[lo:hi, :, :] = color   # horizontal line
        tex[:, lo:hi, :] = color   # vertical line

    # Soft radial vignette (darken edges)
    yy, xx = np.mgrid[:tex_res, :tex_res]
    cx, cy = tex_res / 2.0, tex_res / 2.0
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / (tex_res / 2.0)
    vignette = 1.0 - 0.15 * np.clip(dist, 0, 1) ** 2
    tex = (tex.astype(np.float32) * vignette[..., None]).clip(0, 255).astype(np.uint8)

    # Apply texture via trimesh visual
    from PIL import Image
    image = Image.fromarray(tex)
    material = trimesh.visual.texture.SimpleMaterial(image=image)
    color_visuals = trimesh.visual.TextureVisuals(uv=uvs, image=image, material=material)
    mesh.visual = color_visuals

    return mesh, image


def _look_at_matrix(eye, target, up=(0, -1, 0)):
    """
    Build a 4x4 pose matrix so that a directional light at *eye* points toward
    *target*.  Uses OpenGL convention (camera looks down -Z).
    """
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)

    fwd = target - eye
    fwd /= np.linalg.norm(fwd) + 1e-12

    right = np.cross(fwd, up)
    right /= np.linalg.norm(right) + 1e-12

    new_up = np.cross(right, fwd)
    new_up /= np.linalg.norm(new_up) + 1e-12

    mat = np.eye(4, dtype=np.float64)
    mat[:3, 0] = right
    mat[:3, 1] = new_up
    mat[:3, 2] = -fwd
    mat[:3, 3] = eye
    return mat
