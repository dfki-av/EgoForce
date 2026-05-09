import os
import sys  

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

import argparse
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import quote

import cv2
import torch
import gradio as gr
import numpy as np

from anycalib import AnyCalib
from camera_models import OVR624CameraModel, PinholeCameraModel, Rational8CameraModel
from demo_utils import compose_output_frame

THIS_DIR = Path(__file__).resolve().parent

if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

_INFERENCE = None
_INIT_LOCK = threading.Lock()
_PROCESS_LOCK = threading.Lock()
_GRADIO_TMP_CLEANER_STARTED = False
_GRADIO_TMP_CLEANER_LOCK = threading.Lock()
_FFMPEG_H264_ENCODER = None
_FFMPEG_H264_ENCODER_LOCK = threading.Lock()

DEFAULT_LENS_MODE = "fisheye624"
DEFAULT_VIDEO_INFO = "Upload a video to choose a start time."
GRADIO_TMP_DIR = Path("/tmp/gradio")
GRADIO_TMP_CLEAN_INTERVAL_SECONDS = 30 * 60
GRADIO_TMP_MIN_AGE_SECONDS = 30 * 60
EGOFORCE_TMP_DIR = Path(tempfile.gettempdir())
EGOFORCE_TMP_PREFIX = "egoforce-gradio-"
ASSETS_DIR = Path(ROOT_DIR) / "assets"
ASSETS_IMAGE_DIR = ASSETS_DIR / "images"
ASSETS_CSS_DIR = ASSETS_DIR / "css"
GRADIO_HERO_CSS_PATH = ASSETS_CSS_DIR / "gradio_hero.css"
SAMPLE_VIDEOS_DIR = Path(ROOT_DIR) / "_DATA" / "sample_videos"
SAMPLE_VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".webm"}
MAX_STACKED_OUTPUT_WIDTH = (1080 * 3) - 2


ANYCALIB_LENS_SPECS = {
    "fisheye624": {
        "label": "Fisheye",
        "model_id": "anycalib_gen",
        "cam_id": "simple_kb:4",
        "repo_camera": "fisheye624",
    },
    "pinhole_distortion": {
        "label": "Pinhole + Distortion",
        "model_id": "anycalib_dist",
        "cam_id": "radial:4",
        "repo_camera": "rational8",
    },
    "pinhole": {
        "label": "Pinhole",
        "model_id": "anycalib_pinhole",
        "cam_id": "pinhole",
        "repo_camera": "pinhole",
    },
}
LENS_CHOICES = [(spec["label"], key) for key, spec in ANYCALIB_LENS_SPECS.items()]
PROJECT_LINKS = {
    "arxiv": "https://arxiv.org/abs/2511.06457",
    "code": "https://github.com/Chris10M/EgoForce/tree/main",
    "data": "https://huggingface.co/datasets/chris10/EgoForce",
    "venue": "https://www.siggraph.org/",
}


@lru_cache(maxsize=8)
def asset_to_app_url(asset_path):
    path = Path(asset_path).resolve()
    if not path.exists():
        return ""

    return f"/gradio_api/file={quote(path.as_posix(), safe='/')}"


@lru_cache(maxsize=1)
def build_gradio_hero_html():
    glasses_uri = asset_to_app_url(ASSETS_IMAGE_DIR / "ego_glasses.png")
    hand_uri = asset_to_app_url(ASSETS_IMAGE_DIR / "force_hand.png")

    return f"""
    <section class="egoforce-hero">
      <div class="egoforce-hero-card">
        <div class="egoforce-hero-title-row">
          <img class="egoforce-hero-icon egoforce-hero-icon-left" src="{glasses_uri}" alt="EgoForce glasses">
          <h1 class="egoforce-hero-title">
            <span class="egoforce-brand-black">Ego</span><span class="egoforce-brand-force">Force</span>
          </h1>
          <img class="egoforce-hero-icon egoforce-hero-icon-right" src="{hand_uri}" alt="EgoForce hand">
        </div>

        <p class="egoforce-hero-subtitle">
          Forearm-Guided Camera-Space 3D Hand Pose from a Monocular Egocentric Camera
        </p>

        <div class="egoforce-hero-authors">
          <a href="https://chris10m.github.io/" target="_blank" rel="noopener noreferrer">Christen Millerdurai</a><sup>1</sup>,
          <a href="https://shaoxiang777.github.io/" target="_blank" rel="noopener noreferrer">Shaoxiang Wang</a><sup>1,2</sup>,
          <a href="https://scholar.google.com/citations?user=3ZKuh9EAAAAJ" target="_blank" rel="noopener noreferrer">Yaxu Xie</a><sup>1</sup>,
          <a href="https://people.mpi-inf.mpg.de/~golyanik/" target="_blank" rel="noopener noreferrer">Vladislav Golyanik</a><sup>3</sup>,
          <a href="https://www.dfki.de/en/web/about-us/employee/person/dist01" target="_blank" rel="noopener noreferrer">Didier Stricker</a><sup>1,2</sup>,
          <a href="https://www.dfki.de/en/web/about-us/employee/person/alpa02" target="_blank" rel="noopener noreferrer">Alain Pagani</a><sup>1</sup>
        </div>

        <div class="egoforce-hero-affiliations">
          <span><sup>1</sup>German Research Center for Artificial Intelligence (DFKI)</span>
          <span><sup>2</sup>RPTU</span>
          <span><sup>3</sup>Max Planck Institute for Informatics</span>
        </div>

        <div class="egoforce-hero-venue">
          <a href="{PROJECT_LINKS['venue']}" target="_blank" rel="noopener noreferrer">
            ACM SIGGRAPH Conference Proceedings, 2026
          </a>
        </div>

        <div class="egoforce-hero-links">
          <a class="egoforce-hero-link" href="{PROJECT_LINKS['arxiv']}" target="_blank" rel="noopener noreferrer">arXiv</a>
          <a class="egoforce-hero-link" href="{PROJECT_LINKS['code']}" target="_blank" rel="noopener noreferrer">Code</a>
          <a class="egoforce-hero-link" href="{PROJECT_LINKS['data']}" target="_blank" rel="noopener noreferrer">Data</a>
        </div>
      </div>
    </section>
    """


def format_sample_video_label(video_path):
    return Path(video_path).stem.replace("_", " ").replace("-", " ").title()


@lru_cache(maxsize=1)
def get_sample_video_paths():
    if not SAMPLE_VIDEOS_DIR.exists():
        return tuple()

    paths = sorted(
        path.resolve()
        for path in SAMPLE_VIDEOS_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in SAMPLE_VIDEO_EXTENSIONS
    )
    return tuple(paths)


def build_sample_video_gallery_items():
    return [(str(path), format_sample_video_label(path)) for path in get_sample_video_paths()]


def clamp_stacked_output_frame_size(frame_rgb):
    height, width = frame_rgb.shape[:2]

    target_width = width
    target_height = height

    if width > MAX_STACKED_OUTPUT_WIDTH:
        scale = MAX_STACKED_OUTPUT_WIDTH / float(width)
        target_width = int(round(width * scale))
        target_height = int(round(height * scale))

    # Keep writer dimensions codec-friendly.
    target_width = max(2, target_width - (target_width % 2))
    target_height = max(2, target_height - (target_height % 2))

    if target_width == width and target_height == height:
        return frame_rgb

    return cv2.resize(frame_rgb, (target_width, target_height), interpolation=cv2.INTER_AREA)


def build_pending_calibration_info(lens_mode, use_random_calibration_frame=False):
    spec = ANYCALIB_LENS_SPECS.get(lens_mode, ANYCALIB_LENS_SPECS[DEFAULT_LENS_MODE])
    calibration_source = (
        "A random frame from the uploaded video will be selected at run time to estimate intrinsics."
        if use_random_calibration_frame
        else "The first frame of the uploaded video is used to estimate intrinsics before inference starts."
    )
    return (
        "Calibration pending.\n"
        f"- Lens mode: `{spec['label']}`\n"
        f"- AnyCalib model: `{spec['model_id']}`\n"
        f"- AnyCalib cam_id: `{spec['cam_id']}`\n"
        f"- Target repo camera model: `{spec['repo_camera']}`\n"
        f"- {calibration_source}"
    )


DEFAULT_CALIBRATION_INFO = build_pending_calibration_info(DEFAULT_LENS_MODE)

def get_inference():
    global _INFERENCE
    if _INFERENCE is None:
        with _INIT_LOCK:
            if _INFERENCE is None:
                from inference import Inference

                _INFERENCE = Inference()
    return _INFERENCE


def reset_inference_state(inference):
    from demo_utils import init_tracking_defaults

    if hasattr(inference, "reset_runtime_state"):
        inference.reset_runtime_state()
    else:
        init_tracking_defaults(inference)
        inference.renderer = None
        inference.frame_index = 0


def cleanup_session_artifacts(session_artifacts):
    if not session_artifacts:
        return

    for artifact in session_artifacts:
        if not artifact:
            continue

        path = Path(artifact)
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
        elif path.exists():
            try:
                path.unlink()
            except OSError:
                pass


def cleanup_stale_gradio_tmp_entries():
    if not GRADIO_TMP_DIR.exists():
        return

    cutoff_time = time.time() - GRADIO_TMP_MIN_AGE_SECONDS
    for child in GRADIO_TMP_DIR.iterdir():
        try:
            if child.stat().st_mtime > cutoff_time:
                continue

            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except FileNotFoundError:
            continue
        except OSError:
            continue


def cleanup_stale_egoforce_tmp_entries():
    if not EGOFORCE_TMP_DIR.exists():
        return

    cutoff_time = time.time() - GRADIO_TMP_MIN_AGE_SECONDS
    for child in EGOFORCE_TMP_DIR.iterdir():
        if not child.name.startswith(EGOFORCE_TMP_PREFIX):
            continue

        try:
            if child.stat().st_mtime > cutoff_time:
                continue

            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except FileNotFoundError:
            continue
        except OSError:
            continue


def _gradio_tmp_cleaner_loop():
    while True:
        cleanup_stale_gradio_tmp_entries()
        cleanup_stale_egoforce_tmp_entries()
        time.sleep(GRADIO_TMP_CLEAN_INTERVAL_SECONDS)


def ensure_gradio_tmp_cleaner():
    global _GRADIO_TMP_CLEANER_STARTED

    if _GRADIO_TMP_CLEANER_STARTED:
        return

    with _GRADIO_TMP_CLEANER_LOCK:
        if _GRADIO_TMP_CLEANER_STARTED:
            return

        cleaner = threading.Thread(
            target=_gradio_tmp_cleaner_loop,
            name="gradio-tmp-cleaner",
            daemon=True,
        )
        cleaner.start()
        _GRADIO_TMP_CLEANER_STARTED = True


def create_session_workspace(video_path, session_artifacts):
    cleanup_session_artifacts(session_artifacts)

    temp_dir = Path(tempfile.mkdtemp(prefix=EGOFORCE_TMP_PREFIX))
    stem = Path(video_path).stem or "video"
    return {
        "temp_dir": temp_dir,
        "input_path": temp_dir / f"{stem}_input.mp4",
        "raw_output_path": temp_dir / f"{stem}_stacked_raw.mp4",
        "output_path": temp_dir / f"{stem}_stacked.mp4",
        "artifacts": [str(temp_dir)],
    }


def get_capture_fps(capture):
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0
    return float(fps)


def get_capture_frame_count(capture):
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    if not np.isfinite(frame_count) or frame_count <= 0:
        return None
    return int(frame_count)


@lru_cache(maxsize=1)
def resolve_ffmpeg_bin():
    candidates = [
        str(Path(sys.executable).resolve().parent / "ffmpeg"),
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
        "/usr/local/bin/ffmpeg",
        "/snap/bin/ffmpeg",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    return None


def ensure_ffmpeg_available():
    ffmpeg_bin = resolve_ffmpeg_bin()
    if ffmpeg_bin is not None:
        return ffmpeg_bin
    raise gr.Error(
        "ffmpeg is required to remux videos into a web-friendly MP4, but it was not found on this machine."
    )


def run_ffmpeg(command, error_message):
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except FileNotFoundError as exc:
        raise gr.Error(
            "ffmpeg is required to remux videos into a web-friendly MP4, but it was not found on this machine."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        if stderr:
            raise gr.Error(f"{error_message}: {stderr}") from exc
        raise gr.Error(error_message) from exc

    return completed


def resolve_ffmpeg_h264_encoder(ffmpeg_bin):
    global _FFMPEG_H264_ENCODER

    if _FFMPEG_H264_ENCODER is not None:
        return _FFMPEG_H264_ENCODER

    with _FFMPEG_H264_ENCODER_LOCK:
        if _FFMPEG_H264_ENCODER is not None:
            return _FFMPEG_H264_ENCODER

        try:
            completed = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-encoders"],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            raise gr.Error("Could not inspect ffmpeg encoders for H.264 support.") from exc

        encoders_output = completed.stdout or ""
        if " libx264" in encoders_output:
            _FFMPEG_H264_ENCODER = "libx264"
        elif " libopenh264" in encoders_output:
            _FFMPEG_H264_ENCODER = "libopenh264"
        else:
            raise gr.Error(
                "This ffmpeg build does not provide a supported H.264 encoder. "
                "Install ffmpeg with libx264 or libopenh264 support."
            )

    return _FFMPEG_H264_ENCODER


def build_h264_transcode_args(ffmpeg_bin):
    encoder = resolve_ffmpeg_h264_encoder(ffmpeg_bin)
    args = ["-c:v", encoder, "-pix_fmt", "yuv420p", "-movflags", "+faststart"]

    if encoder == "libx264":
        args[2:2] = ["-preset", "ultrafast"]
    elif encoder == "libopenh264":
        args.extend(["-b:v", "4M"])

    return args


def remux_input_video_to_mp4(input_path, output_path):
    ffmpeg_bin = ensure_ffmpeg_available()
    base_cmd = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-map",
        "0:v:0",
        "-an",
        "-sn",
        "-dn",
    ]
    remux_cmd = base_cmd + [
        "-c:v",
        "copy",
        "-movflags",
        "+faststart",
        str(output_path),
    ]

    try:
        run_ffmpeg(remux_cmd, "Could not remux the uploaded input video")
    except gr.Error:
        transcode_cmd = base_cmd + build_h264_transcode_args(ffmpeg_bin) + [str(output_path)]
        run_ffmpeg(transcode_cmd, "Could not convert the uploaded input video to MP4")

    return output_path


def finalize_output_video_for_web(raw_output_path, output_path):
    ffmpeg_bin = ensure_ffmpeg_available()
    command = [
        ffmpeg_bin,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(raw_output_path),
        "-an",
        *build_h264_transcode_args(ffmpeg_bin),
        str(output_path),
    ]
    run_ffmpeg(command, "Could not finalize the output video for web playback")
    return output_path


def inspect_video(video_path):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    try:
        fps = get_capture_fps(capture)
        frame_count = get_capture_frame_count(capture)
    finally:
        capture.release()

    duration_seconds = None
    if frame_count is not None and fps > 0:
        duration_seconds = frame_count / fps

    return fps, frame_count, duration_seconds


def read_video_frame(video_path, frame_index):
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise gr.Error(f"Could not open video for frame selection: {video_path}")

    try:
        if frame_index > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ret, bgr_frame = capture.read()
    finally:
        capture.release()

    if not ret:
        raise gr.Error(f"Could not decode frame {frame_index} for calibration.")

    return bgr_frame


def select_calibration_frame(frame_count, use_random_calibration_frame):
    if use_random_calibration_frame and frame_count is not None and frame_count > 1:
        return int(np.random.randint(0, frame_count)), "random"
    if use_random_calibration_frame:
        return 0, "first_fallback"
    return 0, "first"


def format_vector(values, precision=4):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return ", ".join(f"{float(value):.{precision}f}" for value in values)


def parse_anycalib_intrinsics(intrinsics, lens_mode):
    intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(-1)

    if lens_mode == "pinhole":
        if intrinsics.size >= 4:
            focal = intrinsics[:2]
            principal = intrinsics[2:4]
            distortion = None
        elif intrinsics.size >= 3:
            focal = np.array([intrinsics[0], intrinsics[0]], dtype=np.float32)
            principal = intrinsics[1:3]
            distortion = None
        else:
            raise ValueError(f"Expected 3 or 4 intrinsics for pinhole, got {intrinsics.size}.")
        return focal, principal, distortion

    if lens_mode == "pinhole_distortion":
        if intrinsics.size >= 8:
            focal = intrinsics[:2]
            principal = intrinsics[2:4]
            radial = intrinsics[4:8]
        elif intrinsics.size >= 7:
            focal = np.array([intrinsics[0], intrinsics[0]], dtype=np.float32)
            principal = intrinsics[1:3]
            radial = intrinsics[3:7]
        else:
            raise ValueError(f"Expected 7 or 8 intrinsics for pinhole+distortion, got {intrinsics.size}.")

        distortion = np.zeros(8, dtype=np.float32)
        distortion[0] = radial[0]
        distortion[1] = radial[1]
        distortion[4] = radial[2]
        distortion[5] = radial[3]
        return focal, principal, distortion

    if lens_mode == "fisheye624":
        if intrinsics.size >= 8:
            focal = intrinsics[:2]
            principal = intrinsics[2:4]
            kb = intrinsics[4:8]
        elif intrinsics.size >= 7:
            focal = np.array([intrinsics[0], intrinsics[0]], dtype=np.float32)
            principal = intrinsics[1:3]
            kb = intrinsics[3:7]
        else:
            raise ValueError(f"Expected 7 or 8 intrinsics for fisheye624, got {intrinsics.size}.")

        distortion = np.zeros(12, dtype=np.float32)
        distortion[:4] = kb
        return focal, principal, distortion

    raise ValueError(f"Unsupported lens mode: {lens_mode}")


def infer_camera_model_from_frame(rgb_image, lens_mode, calibration_frame_index=0, calibration_frame_mode="first"):
    spec = ANYCALIB_LENS_SPECS.get(lens_mode)
    if spec is None:
        raise ValueError(f"Unknown lens mode: {lens_mode}")

    if rgb_image.ndim != 3 or rgb_image.shape[-1] != 3:
        raise ValueError(f"Expected an RGB frame shaped (H, W, 3), got {tuple(rgb_image.shape)}.")

    height, width = rgb_image.shape[:2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    image = torch.tensor(rgb_image, dtype=torch.float32, device=device).permute(2, 0, 1) / 255.0
    anycalib_model = None

    try:
        anycalib_model = AnyCalib(model_id=spec["model_id"]).to(device)
        with torch.no_grad():
            prediction = anycalib_model.predict(image, cam_id=spec["cam_id"])
        intrinsics = prediction["intrinsics"]
        if torch.is_tensor(intrinsics):
            intrinsics = intrinsics.detach().cpu().numpy()
        intrinsics = np.asarray(intrinsics, dtype=np.float32).reshape(-1)
    finally:
        del image
        if anycalib_model is not None:
            del anycalib_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    focal, principal, distortion = parse_anycalib_intrinsics(intrinsics, lens_mode)

    if lens_mode == "pinhole":
        camera_model = PinholeCameraModel(focal, principal, width, height)
        mapping_note = "Used the AnyCalib pinhole intrinsics directly."

    elif lens_mode == "pinhole_distortion":
        camera_model = Rational8CameraModel(focal, principal, distortion, width, height)
        mapping_note = (
            "Mapped AnyCalib radial coefficients to Rational8 as "
            "`[k1, k2, 0, 0, k3, k4, 0, 0]`."
        )

    elif lens_mode == "fisheye624":
        camera_model = OVR624CameraModel(focal, principal, distortion, width, height)
        mapping_note = (
            "Mapped AnyCalib KB coefficients into the fisheye624 radial slots "
            "`[k1, k2, k3, k4]`; the remaining fisheye624 coefficients are set to zero."
        )

    else:
        raise ValueError(f"Unsupported lens mode: {lens_mode}")

    lines = [
        "Calibration complete.",
        (
            f"- Calibration frame index: `{calibration_frame_index}`"
            if calibration_frame_mode == "first"
            else (
                f"- Calibration frame index: `{calibration_frame_index}` (randomly selected)"
                if calibration_frame_mode == "random"
                else (
                    f"- Calibration frame index: `{calibration_frame_index}` "
                    "(random requested, fell back to frame 0 because the video length could not be determined)"
                )
            )
        ),
        f"- Lens mode: `{spec['label']}`",
        f"- AnyCalib model: `{spec['model_id']}`",
        f"- AnyCalib cam_id: `{spec['cam_id']}`",
        f"- Image size: `{width} x {height}`",
        f"- Focal length `[fx, fy]`: `{format_vector(focal)}`",
        f"- Principal point `[cx, cy]`: `{format_vector(principal)}`",
    ]
    if distortion is not None:
        lines.append(f"- Repo distortion params: `{format_vector(distortion)}`")
    lines.append(f"- Mapping: {mapping_note}")

    return camera_model, "\n".join(lines)


def update_video_controls(video_path, lens_mode, use_random_calibration_frame, session_artifacts):
    cleanup_session_artifacts(session_artifacts)
    calibration_info = build_pending_calibration_info(lens_mode, use_random_calibration_frame)
    calibration_note = (
        "Calibration uses a random frame chosen per run."
        if use_random_calibration_frame
        else "Calibration always uses frame 0."
    )

    if not video_path:
        return (
            gr.update(value=0.0, minimum=0.0, maximum=0.0, interactive=False),
            DEFAULT_VIDEO_INFO,
            None,
            calibration_info,
            [],
        )

    input_path = Path(video_path)
    if not input_path.exists():
        return (
            gr.update(value=0.0, minimum=0.0, maximum=0.0, interactive=False),
            f"Input video not found: {input_path}",
            None,
            calibration_info,
            [],
        )

    try:
        fps, frame_count, duration_seconds = inspect_video(input_path)
    except ValueError as exc:
        return (
            gr.update(value=0.0, minimum=0.0, maximum=0.0, interactive=False),
            str(exc),
            None,
            calibration_info,
            [],
        )

    if duration_seconds is None:
        return (
            gr.update(value=0.0, minimum=0.0, maximum=0.0, interactive=False),
            f"FPS: {fps:.2f}. Could not determine video duration; start offset is disabled. {calibration_note}",
            None,
            calibration_info,
            [],
        )

    max_start_seconds = max(0.0, duration_seconds - (1.0 / fps))
    max_start_seconds_display = int(max_start_seconds)
    return (
        gr.update(
            value=0.0,
            minimum=0.0,
            maximum=max_start_seconds_display,
            interactive=True,
        ),
        (
            f"Duration: {duration_seconds:.2f}s, FPS: {fps:.2f}, Frames: {frame_count}, "
            f"Max start: {max_start_seconds_display}s. {calibration_note}"
        ),
        None,
        calibration_info,
        [],
    )


def select_sample_video(lens_mode, use_random_calibration_frame, session_artifacts, evt: gr.SelectData):
    sample_paths = get_sample_video_paths()
    if not sample_paths:
        raise gr.Error(f"No sample videos found in {SAMPLE_VIDEOS_DIR}")

    sample_index = evt.index[0] if isinstance(evt.index, tuple) else evt.index
    if not isinstance(sample_index, int) or sample_index < 0 or sample_index >= len(sample_paths):
        raise gr.Error("Could not determine which sample video was selected.")

    selected_path = str(sample_paths[sample_index])
    start_slider, video_info, output_video, calibration_info, new_session_artifacts = update_video_controls(
        selected_path,
        lens_mode,
        use_random_calibration_frame,
        session_artifacts,
    )
    return (
        selected_path,
        start_slider,
        video_info,
        output_video,
        calibration_info,
        new_session_artifacts,
    )


def process_video(
    video_path,
    session_artifacts,
    start_seconds,
    only_ten_seconds,
    include_arm_mesh,
    use_random_calibration_frame,
    lens_mode,
    progress=gr.Progress(track_tqdm=False),
):
    if not video_path:
        raise gr.Error("Upload a video file.")

    progress(0.0, desc="Preparing video...")

    input_path = Path(video_path)
    if not input_path.exists():
        raise gr.Error(f"Input video not found: {input_path}")

    capture = None
    writer = None
    processed_frames = 0
    session_workspace = create_session_workspace(input_path, session_artifacts)
    processing_input_path = session_workspace["input_path"]
    raw_output_path = session_workspace["raw_output_path"]
    final_output_path = session_workspace["output_path"]
    new_session_artifacts = session_workspace["artifacts"]
    calibration_info = build_pending_calibration_info(lens_mode, use_random_calibration_frame)

    try:
        remux_input_video_to_mp4(input_path, processing_input_path)
        progress(0.08, desc="Opening remuxed video...")
        capture = cv2.VideoCapture(str(processing_input_path))
        if not capture.isOpened():
            raise gr.Error(f"Could not open video after ffmpeg remux: {processing_input_path}")

        fps = get_capture_fps(capture)
        frame_count = get_capture_frame_count(capture)
        progress(0.12, desc="Reading calibration frame...")

        ret, first_bgr_frame = capture.read()
        if not ret:
            raise gr.Error("Could not decode the first video frame for calibration.")

        calibration_frame_index, calibration_frame_mode = select_calibration_frame(
            frame_count,
            use_random_calibration_frame,
        )
        calibration_bgr_frame = (
            first_bgr_frame
            if calibration_frame_index == 0
            else read_video_frame(processing_input_path, calibration_frame_index)
        )
        calibration_rgb_frame = cv2.cvtColor(calibration_bgr_frame, cv2.COLOR_BGR2RGB)
        try:
            progress(0.18, desc="Estimating camera intrinsics...")
            camera_model, calibration_info = infer_camera_model_from_frame(
                calibration_rgb_frame,
                lens_mode,
                calibration_frame_index=calibration_frame_index,
                calibration_frame_mode=calibration_frame_mode,
            )
        except Exception as exc:
            raise gr.Error(f"AnyCalib calibration failed: {exc}") from exc

        start_seconds = max(0.0, float(start_seconds or 0.0))
        start_frame = int(start_seconds * fps)
        if frame_count is not None and start_frame >= frame_count:
            raise gr.Error("Start position is past the end of the video.")

        prefetched_bgr_frame = first_bgr_frame if start_frame == 0 else None
        if start_frame > 0:
            capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        max_frames = int(round(10.0 * fps)) if only_ten_seconds else None
        total_frames_to_process = max_frames
        if total_frames_to_process is None and frame_count is not None:
            total_frames_to_process = max(0, frame_count - start_frame)

        with _PROCESS_LOCK:
            inference = get_inference()
            reset_inference_state(inference)
            inference.set_camera_model(camera_model)
            if hasattr(inference, "set_kalman_filter_frequency"):
                inference.set_kalman_filter_frequency(fps)

            progress(0.22, desc="Running inference...")
            while True:
                if max_frames is not None and processed_frames >= max_frames:
                    break

                if prefetched_bgr_frame is not None:
                    bgr_image = prefetched_bgr_frame
                    prefetched_bgr_frame = None
                else:
                    ret, bgr_image = capture.read()
                    if not ret:
                        break

                rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
                render_image, tp_image = inference.run(
                    rgb_image.copy(),
                    inference.device,
                    include_arm_mesh=include_arm_mesh,
                )
                stacked_rgb = compose_output_frame(rgb_image, render_image, tp_image)
                stacked_rgb = clamp_stacked_output_frame_size(stacked_rgb)

                if writer is None:
                    height, width = stacked_rgb.shape[:2]
                    writer = cv2.VideoWriter(
                        str(raw_output_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps,
                        (width, height),
                    )
                    if not writer.isOpened():
                        raise gr.Error(f"Could not create output video: {raw_output_path}")

                writer.write(cv2.cvtColor(stacked_rgb, cv2.COLOR_RGB2BGR))
                processed_frames += 1
                if total_frames_to_process:
                    inference_fraction = min(processed_frames / total_frames_to_process, 1.0)
                    progress(
                        0.22 + (0.70 * inference_fraction),
                        desc=(
                            f"Running inference... "
                            f"{processed_frames}/{total_frames_to_process} frames"
                        ),
                    )
                elif processed_frames == 1 or processed_frames % max(1, int(round(fps))) == 0:
                    progress(
                        (processed_frames, None),
                        desc=f"Running inference... {processed_frames} frames",
                        unit="frames",
                    )
    except Exception:
        cleanup_session_artifacts(new_session_artifacts)
        raise
    finally:
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()

    if processed_frames == 0:
        cleanup_session_artifacts(new_session_artifacts)
        raise gr.Error("No frames were decoded from the uploaded video.")

    try:
        progress(0.95, desc="Finalizing output video...")
        finalize_output_video_for_web(raw_output_path, final_output_path)
    except Exception:
        cleanup_session_artifacts(new_session_artifacts)
        raise

    progress(1.0, desc="Done.")
    return str(final_output_path), new_session_artifacts, calibration_info


def clear_session(session_artifacts):
    cleanup_session_artifacts(session_artifacts)
    return (
        None,
        None,
        [],
        gr.update(value=0.0, minimum=0.0, maximum=0.0, interactive=False),
        True,
        False,
        False,
        DEFAULT_LENS_MODE,
        DEFAULT_VIDEO_INFO,
        DEFAULT_CALIBRATION_INFO,
    )


def build_app():
    with gr.Blocks(
        title="EgoForce Demo",
        delete_cache=(3600, 3600),
    ) as app:
        gr.HTML(build_gradio_hero_html())

        session_artifacts = gr.State(
            value=[],
            delete_callback=cleanup_session_artifacts,
        )

        with gr.Row():
            input_video = gr.Video(
                label="Input Video",
                sources=["upload"],
            )
            output_video = gr.Video(
                label="Output Video",
                interactive=False,
            )

        with gr.Row():
            start_slider = gr.Slider(
                minimum=0.0,
                maximum=0.0,
                value=0.0,
                step=0.1,
                precision=1,
                label="Start Time (seconds)",
                interactive=False,
            )
            only_ten_seconds = gr.Checkbox(
                label="Process Only 10 Seconds",
                value=True,
                info="If enabled, the app processes at most 10 seconds starting from the selected offset.",
            )
            include_arm_mesh = gr.Checkbox(
                label="Show Hand-Arm Mesh",
                value=False,
                info="Enalbe to output the hand-arm meshes.",
            )
            random_calibration_frame = gr.Checkbox(
                label="Use Random Calibration Frame",
                value=False,
                info="By default calibration uses frame 0; enable to sample one random frame per run.",
            )

        sample_video_items = build_sample_video_gallery_items()
        sample_video_gallery = None
        if sample_video_items:
            sample_video_gallery = gr.Gallery(
                value=sample_video_items,
                label="Example Videos",
                show_label=True,
                columns=len(sample_video_items),
                rows=1,
                object_fit="cover",
                height="auto",
                allow_preview=False,
                preview=False,
                selected_index=None,
                elem_id="sample-video-carousel",
            )
        else:
            gr.Markdown(
                f"No sample videos were found under `{SAMPLE_VIDEOS_DIR}`.",
            )

        lens_mode = gr.Radio(
            choices=LENS_CHOICES,
            value=DEFAULT_LENS_MODE,
            label="Lens Model",
            info="AnyCalib estimates intrinsics on the first frame and maps them to the selected repo camera model.",
        )

        video_info = gr.Markdown(DEFAULT_VIDEO_INFO)
        calibration_info = gr.Markdown(DEFAULT_CALIBRATION_INFO)

        with gr.Row():
            run_button = gr.Button("Run Inference", variant="primary")
            clear_button = gr.Button("Clear")

        input_video.change(
            fn=update_video_controls,
            inputs=[input_video, lens_mode, random_calibration_frame, session_artifacts],
            outputs=[start_slider, video_info, output_video, calibration_info, session_artifacts],
        )
        lens_mode.change(
            fn=update_video_controls,
            inputs=[input_video, lens_mode, random_calibration_frame, session_artifacts],
            outputs=[start_slider, video_info, output_video, calibration_info, session_artifacts],
        )
        random_calibration_frame.change(
            fn=update_video_controls,
            inputs=[input_video, lens_mode, random_calibration_frame, session_artifacts],
            outputs=[start_slider, video_info, output_video, calibration_info, session_artifacts],
        )
        if sample_video_gallery is not None:
            sample_video_gallery.select(
                fn=select_sample_video,
                inputs=[lens_mode, random_calibration_frame, session_artifacts],
                outputs=[
                    input_video,
                    start_slider,
                    video_info,
                    output_video,
                    calibration_info,
                    session_artifacts,
                ],
                show_progress="hidden",
            )
        run_button.click(
            fn=process_video,
            inputs=[
                input_video,
                session_artifacts,
                start_slider,
                only_ten_seconds,
                include_arm_mesh,
                random_calibration_frame,
                lens_mode,
            ],
            outputs=[output_video, session_artifacts, calibration_info],
            show_progress="full",
        )
        clear_button.click(
            fn=clear_session,
            inputs=session_artifacts,
            outputs=[
                input_video,
                output_video,
                session_artifacts,
                start_slider,
                only_ten_seconds,
                include_arm_mesh,
                random_calibration_frame,
                lens_mode,
                video_info,
                calibration_info,
            ],
        )

    return app


def parse_args():
    parser = argparse.ArgumentParser(description="Run the EgoForce Gradio demo.")
    parser.add_argument("--share", action="store_true", help="Enable a public Gradio share link.")
    parser.add_argument("--server-name", default=None, help="Optional Gradio server host.")
    parser.add_argument("--server-port", type=int, default=None, help="Optional Gradio server port.")
    return parser.parse_args()


def main():
    args = parse_args()
    ensure_gradio_tmp_cleaner()
    app = build_app().queue(default_concurrency_limit=1)
    app.launch(
        share=args.share,
        server_name=args.server_name,
        server_port=args.server_port,
        css_paths=[str(GRADIO_HERO_CSS_PATH.resolve())] if GRADIO_HERO_CSS_PATH.exists() else None,
        allowed_paths=[str(ASSETS_DIR.resolve())],
    )


if __name__ == "__main__":
    main()
