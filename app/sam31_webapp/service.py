from __future__ import annotations

import contextlib
import copy
import gc
import io
import locale
import math
import os
import re
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO, Callable
from urllib.parse import quote

import cv2
import numpy as np
import torch
from PIL import Image

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency fallback
    psutil = None

from .mask_postprocess import (
    MASK_POSTPROCESS_VIDEOMAMA,
    MASK_POSTPROCESS_VITMATTE,
    MaskPostprocessOptions,
    VideoMaMaRefiner,
    VitMatteRefiner,
    validate_mask_postprocess_options,
)


WEBAPP_ROOT = Path(__file__).resolve().parent
APP_ROOT = WEBAPP_ROOT.parent
PROJECT_ROOT = APP_ROOT.parent
MODELS_ROOT = PROJECT_ROOT / "models"
SAM3_REPO = MODELS_ROOT / "sam3"
SAM31_CHECKPOINT = MODELS_ROOT / "sam3.1" / "sam3.1_multiplex.pt"
VITMATTE_MODEL_DIR = MODELS_ROOT / "vitmatte-base-composition-1k"
VIDEOMAMA_REPO_DIR = MODELS_ROOT / "VideoMaMa"
VIDEOMAMA_MODEL_DIR = VIDEOMAMA_REPO_DIR / "model"
VIDEOMAMA_BASE_MODEL_DIR = VIDEOMAMA_MODEL_DIR / "stable-video-diffusion-img2vid-xt"
DATA_ROOT = PROJECT_ROOT / "cache"
SESSIONS_ROOT = DATA_ROOT
DEFAULT_PREVIEW_BITRATE = "10M"
DEFAULT_SAM_MAX_INFERENCE_PIXELS = 1920 * 1080
MIN_SAM_MAX_INFERENCE_PIXELS = 64 * 64
MAX_SAM_MAX_INFERENCE_PIXELS = 16_384 * 16_384
COLOR_TABLE = [
    (255, 99, 71),
    (65, 105, 225),
    (60, 179, 113),
    (255, 215, 0),
    (186, 85, 211),
    (0, 206, 209),
]
BITRATE_RE = re.compile(r"^\d+(?:\.\d+)?[kKmM]$")
SIZE_RE = re.compile(r"^\s*(\d+)\s*[xX*]\s*(\d+)\s*$")
SPINNER_FRAMES = "|/-\\"
PROGRESS_SUFFIX_RE = re.compile(r"^(.*?)(\d+/\d+)\s*$")
PROPAGATION_CACHE_DIRNAME = "propagation_cache"
PROPAGATION_CACHE_FILE_TEMPLATE = "frame_{frame_index:06d}.npz"
MIN_PROPAGATION_BATCH_FRAMES = 4
MAX_PROPAGATION_BATCH_FRAMES = 48
MIN_PROPAGATION_BATCH_BYTES = 64 * 1024 * 1024
MAX_PROPAGATION_BATCH_BYTES = 384 * 1024 * 1024
GPU_STATE_OFFLOAD_FREE_BYTES = 8 * 1024 * 1024 * 1024
GPU_STATE_OFFLOAD_PIXEL_FRAMES = 1920 * 1080 * 120
PIL_LANCZOS = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
)


class ConsoleReporter:
    def __init__(self, prefix: str) -> None:
        self.prefix = prefix
        self._lock = threading.Lock()
        self._stream = getattr(sys, "__stdout__", None) or sys.stdout
        self._active = False
        self._active_channel: str | None = None
        self._active_message = ""
        self._last_render_at = 0.0

    def info(self, message: str) -> None:
        with self._lock:
            self._flush_active_locked()
            self._write_line_locked(message)

    def error(self, message: str) -> None:
        with self._lock:
            self._flush_active_locked()
            self._write_line_locked(f"ERROR {message}")

    def update(
        self,
        message: str,
        *,
        channel: str = "default",
        min_interval_s: float = 0.0,
    ) -> None:
        now = time.time()
        with self._lock:
            if self._active and self._active_channel != channel:
                self._flush_active_locked()

            if (
                self._active
                and self._active_channel == channel
                and message == self._active_message
            ):
                return
            if (
                self._active
                and self._active_channel == channel
                and (now - self._last_render_at) < min_interval_s
            ):
                return

            if not self._active:
                self._start_active_locked(message)
            else:
                self._append_active_locked(message)
            self._active = True
            self._active_channel = channel
            self._active_message = message
            self._last_render_at = now

    def finish(self, message: str | None = None, *, channel: str = "default") -> None:
        with self._lock:
            if not self._active or self._active_channel != channel:
                if message:
                    self._write_line_locked(message)
                return
            final_message = message or self._active_message
            if final_message and final_message != self._active_message:
                self._append_fragment_locked(
                    self._build_append_fragment(self._active_message, final_message)
                )
                self._active_message = final_message
            self._stream.write("\n")
            self._stream.flush()
            self._reset_active_locked()

    def _flush_active_locked(self) -> None:
        if not self._active:
            return
        self._stream.write("\n")
        self._stream.flush()
        self._reset_active_locked()

    def _reset_active_locked(self) -> None:
        self._active = False
        self._active_channel = None
        self._active_message = ""
        self._last_render_at = 0.0

    def _start_active_locked(self, message: str) -> None:
        self._stream.write(f"[{self.prefix}] {message}")
        self._stream.flush()

    def _append_active_locked(self, message: str) -> None:
        self._append_fragment_locked(
            self._build_append_fragment(self._active_message, message)
        )

    def _append_fragment_locked(self, fragment: str) -> None:
        if not fragment:
            return
        self._stream.write(fragment)
        self._stream.flush()

    def _build_append_fragment(self, previous_message: str, message: str) -> str:
        if not message or message == previous_message:
            return ""
        previous_label, previous_progress = self._split_progress_message(previous_message)
        current_label, current_progress = self._split_progress_message(message)
        if (
            previous_label is not None
            and current_label is not None
            and previous_label == current_label
            and current_progress
            and current_progress != previous_progress
        ):
            return f"...{current_progress}"
        return f" | {message}"

    def _split_progress_message(self, message: str) -> tuple[str | None, str | None]:
        match = PROGRESS_SUFFIX_RE.match(message)
        if not match:
            return None, None
        return match.group(1).rstrip(), match.group(2)

    def _write_line_locked(self, message: str) -> None:
        self._stream.write(f"[{self.prefix}] {message}\n")
        self._stream.flush()


CONSOLE = ConsoleReporter("sam31-webapp")


def console_info(message: str) -> None:
    CONSOLE.info(message)


def console_error(message: str) -> None:
    CONSOLE.error(message)


def console_update(
    message: str,
    *,
    channel: str = "default",
    min_interval_s: float = 0.0,
) -> None:
    CONSOLE.update(message, channel=channel, min_interval_s=min_interval_s)


def console_finish(message: str | None = None, *, channel: str = "default") -> None:
    CONSOLE.finish(message, channel=channel)


def reset_runtime_cache(cache_root: Path) -> None:
    cache_root = Path(cache_root)
    if cache_root.exists():
        if cache_root.is_dir():
            shutil.rmtree(cache_root)
        else:
            cache_root.unlink()
    cache_root.mkdir(parents=True, exist_ok=True)


def ensure_repo_on_path(repo_dir: Path) -> None:
    import sys

    repo_root = str(repo_dir)
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)


def build_predictor(repo_dir: Path, checkpoint_path: Path):
    ensure_repo_on_path(repo_dir)
    from sam3.model_builder import build_sam3_predictor

    # The OSS builder prints checkpoint diagnostics that are not actionable for this app.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
        io.StringIO()
    ):
        return build_sam3_predictor(
            checkpoint_path=str(checkpoint_path),
            version="sam3.1",
            compile=False,
            warm_up=False,
            use_fa3=False,
            use_rope_real=True,
            async_loading_frames=False,
        )


def safe_filename(filename: str) -> str:
    candidate = Path(filename or "uploaded.mp4").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", candidate).strip("._")
    return cleaned or "uploaded.mp4"


def validate_bitrate(value: str | None) -> str:
    bitrate = (value or "").strip()
    if not bitrate:
        return DEFAULT_PREVIEW_BITRATE
    if bitrate.isdigit():
        bitrate = f"{bitrate}M"
    if not BITRATE_RE.match(bitrate):
        raise ValueError("码率格式应类似 4M、8000k 或 12m。")
    return bitrate


def validate_sam_max_inference_pixels(
    value: int | str | None,
    default: int = DEFAULT_SAM_MAX_INFERENCE_PIXELS,
) -> int:
    if value is None or value == "":
        return int(default)
    if isinstance(value, str):
        text = value.strip()
        match = SIZE_RE.match(text)
        if match:
            parsed = int(match.group(1)) * int(match.group(2))
        else:
            parsed = int(text)
    else:
        parsed = int(value)
    if parsed < MIN_SAM_MAX_INFERENCE_PIXELS:
        raise ValueError(
            f"SAM 3.1 推理像素上限不能小于 {MIN_SAM_MAX_INFERENCE_PIXELS}。"
        )
    if parsed > MAX_SAM_MAX_INFERENCE_PIXELS:
        raise ValueError(
            f"SAM 3.1 推理像素上限不能大于 {MAX_SAM_MAX_INFERENCE_PIXELS}。"
        )
    return parsed


def fit_size_to_max_pixels(
    width: int, height: int, max_pixels: int
) -> tuple[int, int, bool]:
    if width * height <= max_pixels:
        return width, height, False

    scale = math.sqrt(max_pixels / float(width * height))
    target_width = max(2, int(round(width * scale)))
    target_height = max(2, int(round(height * scale)))
    target_width -= target_width % 2
    target_height -= target_height % 2
    target_width = max(2, target_width)
    target_height = max(2, target_height)

    while target_width * target_height > max_pixels and (
        target_width > 2 or target_height > 2
    ):
        if target_width >= target_height and target_width > 2:
            target_width -= 2
        elif target_height > 2:
            target_height -= 2
        else:
            break

    return target_width, target_height, True


def get_available_system_memory_bytes() -> int:
    if psutil is not None:
        try:
            return int(psutil.virtual_memory().available)
        except Exception:
            pass
    return 4 * 1024 * 1024 * 1024


def get_available_gpu_memory_bytes() -> int | None:
    if not torch.cuda.is_available():
        return None
    try:
        free_bytes, _ = torch.cuda.mem_get_info()
    except Exception:
        return None
    return int(free_bytes)


def should_offload_state_to_cpu(
    frame_count: int, sam_width: int, sam_height: int
) -> bool:
    frame_pixels = frame_count * sam_width * sam_height
    gpu_free_bytes = get_available_gpu_memory_bytes()
    if gpu_free_bytes is not None and gpu_free_bytes < GPU_STATE_OFFLOAD_FREE_BYTES:
        return True
    return frame_pixels >= GPU_STATE_OFFLOAD_PIXEL_FRAMES


def resize_bgr_frame_lanczos(
    frame_bgr: np.ndarray, width: int, height: int
) -> np.ndarray:
    if frame_bgr.shape[1] == width and frame_bgr.shape[0] == height:
        return frame_bgr
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = Image.fromarray(frame_rgb).resize((width, height), resample=PIL_LANCZOS)
    return cv2.cvtColor(np.asarray(resized), cv2.COLOR_RGB2BGR)


def resize_gray_frame_lanczos(
    frame_gray: np.ndarray, width: int, height: int
) -> np.ndarray:
    if frame_gray.shape[1] == width and frame_gray.shape[0] == height:
        return frame_gray
    resized = Image.fromarray(frame_gray, mode="L").resize(
        (width, height),
        resample=PIL_LANCZOS,
    )
    return np.asarray(resized, dtype=np.uint8)


def outputs_cache_path(cache_dir: Path, frame_index: int) -> Path:
    return cache_dir / PROPAGATION_CACHE_FILE_TEMPLATE.format(frame_index=frame_index)


def serialize_outputs(outputs: dict[str, Any]) -> dict[str, np.ndarray]:
    return {
        "out_obj_ids": np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64),
        "out_probs": np.asarray(outputs.get("out_probs", []), dtype=np.float32),
        "out_boxes_xywh": np.asarray(
            outputs.get("out_boxes_xywh", []), dtype=np.float32
        ).reshape(-1, 4),
        "out_binary_masks": np.asarray(
            outputs.get("out_binary_masks", []), dtype=bool
        ),
    }


def estimate_outputs_bytes(outputs: dict[str, Any]) -> int:
    arrays = serialize_outputs(outputs)
    return sum(int(array.nbytes) for array in arrays.values())


def load_cached_outputs(
    cache_dir: Path, frame_index: int, height: int, width: int
) -> dict[str, Any]:
    cache_path = outputs_cache_path(cache_dir, frame_index)
    if not cache_path.is_file():
        return empty_outputs(height, width)
    with np.load(cache_path, allow_pickle=False) as data:
        return {
            "out_obj_ids": data["out_obj_ids"],
            "out_probs": data["out_probs"],
            "out_boxes_xywh": data["out_boxes_xywh"],
            "out_binary_masks": data["out_binary_masks"],
        }


def choose_propagation_batch_limits(
    frame_count: int, sam_width: int, sam_height: int
) -> tuple[int, int]:
    available_bytes = get_available_system_memory_bytes()
    batch_bytes = int(
        max(
            MIN_PROPAGATION_BATCH_BYTES,
            min(MAX_PROPAGATION_BATCH_BYTES, available_bytes * 0.08),
        )
    )
    estimated_per_frame_bytes = max(1, sam_width * sam_height * 3)
    batch_frames = batch_bytes // estimated_per_frame_bytes
    batch_frames = max(MIN_PROPAGATION_BATCH_FRAMES, batch_frames)
    batch_frames = min(MAX_PROPAGATION_BATCH_FRAMES, batch_frames, frame_count)
    return batch_frames, batch_bytes


def resize_video_for_sam(
    input_path: Path,
    output_path: Path,
    frame_count: int,
    fps: float,
    width: int,
    height: int,
) -> None:
    progress_channel = f"resize:{output_path.name}"
    console_info(
        f"SAM 推理视频将按原比例缩放到 {width}x{height}：{input_path.name}"
    )
    completed = False
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{input_path}")

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"创建 SAM 3.1 推理视频失败：{output_path}")

    try:
        for frame_idx in range(frame_count):
            ok, frame_bgr = cap.read()
            if not ok:
                raise RuntimeError(f"读取第 {frame_idx} 帧时失败。")
            resized = resize_bgr_frame_lanczos(frame_bgr, width, height)
            writer.write(resized)
            if frame_idx == frame_count - 1 or frame_idx % 10 == 0:
                console_update(
                    f"正在生成 SAM 推理视频... {frame_idx + 1}/{frame_count}",
                    channel=progress_channel,
                )
        completed = True
    finally:
        cap.release()
        writer.release()
        if completed:
            console_finish(
                f"SAM 推理视频已生成：{output_path.name} ({width}x{height})",
                channel=progress_channel,
            )


def open_video(video_path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count <= 0 or width <= 0 or height <= 0:
            raise RuntimeError(f"视频元数据无效：{video_path}")
        return frame_count, fps, width, height
    finally:
        cap.release()


def read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频：{video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame_bgr = cap.read()
        if not ok:
            raise RuntimeError(f"读取第 {frame_index} 帧失败：{video_path}")
        return frame_bgr
    finally:
        cap.release()


def get_color(obj_id: int) -> tuple[int, int, int]:
    return COLOR_TABLE[obj_id % len(COLOR_TABLE)]


def overlay_mask(
    frame_bgr: np.ndarray, mask: np.ndarray, color: tuple[int, int, int]
) -> np.ndarray:
    overlay = frame_bgr.copy()
    overlay[mask] = color
    return cv2.addWeighted(overlay, 0.35, frame_bgr, 0.65, 0.0)


def resize_mask_to_shape(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    mask_bool = np.asarray(mask).astype(bool)
    if mask_bool.ndim > 2:
        mask_bool = np.squeeze(mask_bool)
    if mask_bool.shape == (height, width):
        return mask_bool
    resized = cv2.resize(
        mask_bool.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    return resized.astype(bool)


def draw_outputs(frame_bgr: np.ndarray, outputs: dict[str, Any]) -> np.ndarray:
    rendered = frame_bgr.copy()
    obj_ids = outputs.get("out_obj_ids", [])
    boxes = outputs.get("out_boxes_xywh", [])
    probs = outputs.get("out_probs", [])
    masks = outputs.get("out_binary_masks", [])

    for obj_id, box_xywh, prob, mask in zip(obj_ids, boxes, probs, masks):
        color = get_color(int(obj_id))
        height, width = rendered.shape[:2]
        mask_bool = resize_mask_to_shape(mask, height, width)
        if mask_bool.any():
            rendered = overlay_mask(rendered, mask_bool, color)

        x, y, box_w, box_h = box_xywh
        x1 = int(round(x * width))
        y1 = int(round(y * height))
        x2 = int(round((x + box_w) * width))
        y2 = int(round((y + box_h) * height))
        cv2.rectangle(rendered, (x1, y1), (x2, y2), color, 2)
        label = f"id={int(obj_id)} score={float(prob):.3f}"
        cv2.putText(
            rendered,
            label,
            (x1, max(24, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
            cv2.LINE_AA,
        )

    return rendered


def build_mask_frame(outputs: dict[str, Any], height: int, width: int) -> np.ndarray:
    mask_frame = np.zeros((height, width), dtype=np.uint8)
    for mask in outputs.get("out_binary_masks", []):
        mask_bool = resize_mask_to_shape(mask, height, width)
        mask_frame[mask_bool] = 255
    return mask_frame


def empty_outputs(height: int, width: int) -> dict[str, Any]:
    return {
        "out_obj_ids": np.zeros(0, dtype=np.int64),
        "out_probs": np.zeros(0, dtype=np.float32),
        "out_boxes_xywh": np.zeros((0, 4), dtype=np.float32),
        "out_binary_masks": np.zeros((0, height, width), dtype=bool),
    }


def summarize_outputs(outputs: dict[str, Any]) -> dict[str, Any]:
    obj_ids = [
        int(x) for x in np.asarray(outputs.get("out_obj_ids", []), dtype=np.int64)
    ]
    scores = [
        float(x) for x in np.asarray(outputs.get("out_probs", []), dtype=np.float32)
    ]
    return {
        "num_objects": len(obj_ids),
        "object_ids": obj_ids,
        "scores": scores,
    }


def save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError(f"PNG 编码失败：{path}")
    path.write_bytes(encoded.tobytes())


def build_mask_rgba(mask_frame: np.ndarray) -> np.ndarray:
    rgba = np.zeros((mask_frame.shape[0], mask_frame.shape[1], 4), dtype=np.uint8)
    active = mask_frame > 0
    rgba[active] = (70, 210, 200, 165)
    return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)


def reencode_h264(
    input_path: Path, output_path: Path, bitrate: str, ffmpeg_path: str = "ffmpeg"
) -> None:
    command = [
        ffmpeg_path,
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-b:v",
        bitrate,
        "-maxrate",
        bitrate,
        "-bufsize",
        bitrate,
        str(output_path),
    ]
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
    )
    if result.returncode != 0:
        stdout_text = _decode_subprocess_output(result.stdout)
        stderr_text = _decode_subprocess_output(result.stderr)
        raise RuntimeError(
            "ffmpeg 重编码失败。\n"
            f"stdout:\n{stdout_text}\n"
            f"stderr:\n{stderr_text}"
        )


def _decode_subprocess_output(payload: bytes | None) -> str:
    if not payload:
        return ""
    candidate_encodings = ("utf-8", locale.getpreferredencoding(False), "gbk")
    for encoding in candidate_encodings:
        if not encoding:
            continue
        with contextlib.suppress(UnicodeDecodeError, LookupError):
            return payload.decode(encoding)
    return payload.decode("utf-8", errors="replace")


@dataclass
class WebSession:
    session_id: str
    predictor_session_id: str | None
    video_path: Path
    sam_video_path: Path
    session_dir: Path
    frame_count: int
    fps: float
    width: int
    height: int
    sam_width: int
    sam_height: int
    sam_max_inference_pixels: int
    sam_resized: bool
    predictor_offload_video_to_cpu: bool = True
    predictor_offload_state_to_cpu: bool = True
    predictor_generation: int = 0
    preview_run_index: int = 0
    status: str = "idle"
    message: str = "就绪"
    progress_current: int = 0
    progress_total: int = 0
    prompt_frame_index: int | None = None
    prompt_mode: str | None = None
    prompt_summary: dict[str, Any] = field(default_factory=dict)
    prompt_mask_rgba_path: Path | None = None
    prompt_mask_bw_path: Path | None = None
    overlay_preview_path: Path | None = None
    overlay_source_path: Path | None = None
    mask_preview_path: Path | None = None
    mask_source_path: Path | None = None
    mask_postprocess_options: MaskPostprocessOptions = field(
        default_factory=MaskPostprocessOptions
    )
    last_prompt_outputs: dict[str, Any] | None = None
    last_prompt_request: dict[str, Any] | None = None
    prompt_keyframes: dict[int, dict[str, Any]] = field(default_factory=dict)
    prompt_keyframe_requests: dict[int, dict[str, Any]] = field(default_factory=dict)
    keyframe_mode_enabled: bool = False
    worker: threading.Thread | None = field(default=None, repr=False)
    error: str | None = None
    updated_at: float = field(default_factory=time.time)
    lock: threading.Lock = field(
        default_factory=threading.Lock, repr=False, compare=False
    )


class InferenceWorker:
    def __init__(self, sam3_repo: Path, checkpoint_path: Path) -> None:
        self.sam3_repo = sam3_repo
        self.checkpoint_path = checkpoint_path
        self._queue: queue.Queue[
            tuple[Callable[[Any], Any] | None, dict[str, Any]]
        ] = (
            queue.Queue()
        )
        self._predictor = None
        self._thread = threading.Thread(
            target=self._loop,
            name="sam31-inference-worker",
            daemon=True,
        )
        self._thread.start()

    def run(self, func: Callable[[Any], Any]) -> Any:
        if threading.current_thread() is self._thread:
            return func(self._get_predictor())

        job_state: dict[str, Any] = {
            "event": threading.Event(),
            "result": None,
            "error": None,
        }
        self._queue.put((func, job_state))
        job_state["event"].wait()
        error = job_state["error"]
        if error is not None:
            raise error
        return job_state["result"]

    def release_predictor(self) -> bool:
        if threading.current_thread() is self._thread:
            return self._release_predictor()

        job_state: dict[str, Any] = {
            "event": threading.Event(),
            "result": None,
            "error": None,
        }
        self._queue.put((None, job_state))
        job_state["event"].wait()
        error = job_state["error"]
        if error is not None:
            raise error
        return bool(job_state["result"])

    def _loop(self) -> None:
        while True:
            func, job_state = self._queue.get()
            try:
                if func is None:
                    job_state["result"] = self._release_predictor()
                else:
                    with contextlib.redirect_stdout(
                        io.StringIO()
                    ), contextlib.redirect_stderr(io.StringIO()):
                        job_state["result"] = func(self._get_predictor())
            except Exception as exc:  # pragma: no cover - thread hop path
                console_error(f"推理线程异常：{type(exc).__name__}: {exc}")
                job_state["error"] = exc
            finally:
                job_state["event"].set()

    def _get_predictor(self):
        if self._predictor is not None:
            return self._predictor
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"未找到模型权重：{self.checkpoint_path}")
        if not self.sam3_repo.is_dir():
            raise FileNotFoundError(f"未找到 SAM3 源码目录：{self.sam3_repo}")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device != "cuda":
            raise RuntimeError("SAM 3.1 Web 应用当前需要 CUDA 环境。")
        console_info("正在加载 SAM 3.1 模型...")
        self._predictor = build_predictor(self.sam3_repo, self.checkpoint_path)
        console_info("SAM 3.1 模型已加载。")
        return self._predictor

    def _release_predictor(self) -> bool:
        if self._predictor is None:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            return False

        console_info("正在释放 SAM 3.1 模型显存，准备运行 VideoMaMa...")
        predictor = self._predictor
        self._predictor = None
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
        console_info("SAM 3.1 模型显存已释放。")
        return True


class Sam31WebService:
    def __init__(
        self,
        sam3_repo: Path = SAM3_REPO,
        checkpoint_path: Path = SAM31_CHECKPOINT,
        vitmatte_model_dir: Path = VITMATTE_MODEL_DIR,
        videomama_repo_dir: Path = VIDEOMAMA_REPO_DIR,
        videomama_base_model_dir: Path = VIDEOMAMA_BASE_MODEL_DIR,
        videomama_checkpoint_dir: Path = VIDEOMAMA_MODEL_DIR,
        data_root: Path = DATA_ROOT,
        default_sam_max_inference_pixels: int | str | None = None,
    ) -> None:
        self.sam3_repo = Path(sam3_repo)
        self.checkpoint_path = Path(checkpoint_path)
        self.vitmatte_model_dir = Path(vitmatte_model_dir)
        self.videomama_repo_dir = Path(videomama_repo_dir)
        self.videomama_base_model_dir = Path(videomama_base_model_dir)
        self.videomama_checkpoint_dir = Path(videomama_checkpoint_dir)
        self.data_root = Path(data_root)
        self.default_sam_max_inference_pixels = validate_sam_max_inference_pixels(
            default_sam_max_inference_pixels
            if default_sam_max_inference_pixels is not None
            else os.environ.get("SAM31_MAX_INFERENCE_PIXELS"),
        )
        self.sessions_root = self.data_root
        reset_runtime_cache(self.sessions_root)
        console_info(f"启动时已清理缓存目录：{self.sessions_root}")
        self._inference_worker = InferenceWorker(self.sam3_repo, self.checkpoint_path)
        self._vitmatte_refiner = VitMatteRefiner(self.vitmatte_model_dir)
        self._videomama_refiner = VideoMaMaRefiner(
            self.videomama_repo_dir,
            self.videomama_base_model_dir,
            self.videomama_checkpoint_dir,
        )
        self._sessions: dict[str, WebSession] = {}
        self._sessions_lock = threading.Lock()
        self._predictor_generation = 0

    def _run_inference(self, func: Callable[[Any], Any]) -> Any:
        return self._inference_worker.run(func)

    def _get_session(self, session_id: str) -> WebSession:
        with self._sessions_lock:
            session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"未找到会话：{session_id}")
        return session

    def _build_start_session_request(self, session: WebSession) -> dict[str, Any]:
        return {
            "type": "start_session",
            "resource_path": str(session.sam_video_path),
            "offload_video_to_cpu": session.predictor_offload_video_to_cpu,
            "offload_state_to_cpu": session.predictor_offload_state_to_cpu,
        }

    @staticmethod
    def _bind_request_to_predictor_session(
        request: dict[str, Any], predictor_session_id: str
    ) -> dict[str, Any]:
        bound_request = copy.deepcopy(request)
        bound_request["session_id"] = predictor_session_id
        return bound_request

    def _collect_replay_requests(self, session: WebSession) -> list[dict[str, Any]]:
        with session.lock:
            prompt_mode = session.prompt_mode
            keyframe_requests = [
                copy.deepcopy(request)
                for _, request in sorted(session.prompt_keyframe_requests.items())
            ]
            last_prompt_request = copy.deepcopy(session.last_prompt_request)

        if prompt_mode == "points" and keyframe_requests:
            return keyframe_requests
        if last_prompt_request is not None:
            return [last_prompt_request]
        return []

    def _replay_predictor_state(
        self, session: WebSession, predictor_session_id: str
    ) -> None:
        replay_requests = [
            self._bind_request_to_predictor_session(request, predictor_session_id)
            for request in self._collect_replay_requests(session)
        ]
        if not replay_requests:
            return

        console_info(
            f"正在恢复会话 {session.session_id} 的 SAM 内部提示状态："
            f"{len(replay_requests)} 条提示"
        )

        def _replay_requests(predictor):
            for request in replay_requests:
                predictor.handle_request(request)

        self._run_inference(_replay_requests)

    def _rebuild_predictor_session(
        self, session: WebSession, *, restore_prompts: bool
    ) -> str:
        session_channel = f"session:{session.session_id}:restore"
        console_update("正在恢复 SAM 3.1 会话...", channel=session_channel)
        response = self._run_inference(
            lambda predictor: predictor.handle_request(
                self._build_start_session_request(session)
            )
        )
        predictor_session_id = str(response["session_id"])
        try:
            if restore_prompts:
                self._replay_predictor_state(session, predictor_session_id)
        except Exception:
            try:
                self._run_inference(
                    lambda predictor: predictor.handle_request(
                        {
                            "type": "close_session",
                            "session_id": predictor_session_id,
                        }
                    )
                )
            except Exception:
                pass
            raise

        with session.lock:
            session.predictor_session_id = predictor_session_id
            session.predictor_generation = self._predictor_generation
            session.updated_at = time.time()
        console_finish("SAM 3.1 会话已恢复。", channel=session_channel)
        return predictor_session_id

    def _ensure_predictor_session(
        self, session: WebSession, *, restore_prompts: bool
    ) -> str:
        with session.lock:
            predictor_session_id = session.predictor_session_id
            predictor_generation = session.predictor_generation
        if (
            predictor_session_id is not None
            and predictor_generation == self._predictor_generation
        ):
            return predictor_session_id
        return self._rebuild_predictor_session(
            session,
            restore_prompts=restore_prompts,
        )

    def _invalidate_predictor_sessions(self) -> None:
        with self._sessions_lock:
            self._predictor_generation += 1
            sessions = list(self._sessions.values())
        for session in sessions:
            with session.lock:
                session.predictor_session_id = None
                session.predictor_generation = -1
                session.updated_at = time.time()

    def _allocate_preview_paths(
        self, session: WebSession
    ) -> tuple[Path, Path, Path, Path]:
        with session.lock:
            run_index = session.preview_run_index
        while True:
            run_index += 1
            suffix = f"_{run_index:04d}"
            overlay_source = (
                session.session_dir / f"overlay_preview_source{suffix}.mp4"
            )
            overlay_preview = session.session_dir / f"overlay_preview{suffix}.mp4"
            mask_source = session.session_dir / f"mask_preview_source{suffix}.mp4"
            mask_preview = session.session_dir / f"mask_preview{suffix}.mp4"
            if not any(
                path.exists()
                for path in (
                    overlay_source,
                    overlay_preview,
                    mask_source,
                    mask_preview,
                )
            ):
                break
        with session.lock:
            session.preview_run_index = run_index
        return overlay_source, overlay_preview, mask_source, mask_preview

    def _media_url(self, session: WebSession, file_path: Path) -> str:
        relative = file_path.resolve().relative_to(session.session_dir.resolve())
        stamp = int(file_path.stat().st_mtime) if file_path.exists() else int(time.time())
        encoded_relative = quote(relative.as_posix(), safe="/")
        return f"/media/{session.session_id}/{encoded_relative}?v={stamp}"

    def _snapshot(self, session: WebSession) -> dict[str, Any]:
        with session.lock:
            snapshot = {
                "sessionId": session.session_id,
                "status": session.status,
                "message": session.message,
                "updatedAt": session.updated_at,
                "progressCurrent": session.progress_current,
                "progressTotal": session.progress_total,
                "frameCount": session.frame_count,
                "fps": session.fps,
                "width": session.width,
                "height": session.height,
                "samInference": {
                    "maxPixels": session.sam_max_inference_pixels,
                    "width": session.sam_width,
                    "height": session.sam_height,
                    "resized": session.sam_resized,
                },
                "durationSeconds": session.frame_count / session.fps,
                "promptFrameIndex": session.prompt_frame_index,
                "promptMode": session.prompt_mode,
                "promptSummary": session.prompt_summary,
                "hasPrompt": session.last_prompt_outputs is not None,
                "keyframeEnabled": session.keyframe_mode_enabled,
                "keyframeCount": len(session.prompt_keyframes),
                "keyframeFrames": sorted(session.prompt_keyframes),
                "error": session.error,
                "videoUrl": self._media_url(session, session.video_path),
                "promptMaskUrl": (
                    self._media_url(session, session.prompt_mask_rgba_path)
                    if session.prompt_mask_rgba_path and session.prompt_mask_rgba_path.exists()
                    else None
                ),
                "promptBwMaskUrl": (
                    self._media_url(session, session.prompt_mask_bw_path)
                    if session.prompt_mask_bw_path and session.prompt_mask_bw_path.exists()
                    else None
                ),
                "overlayPreviewUrl": (
                    self._media_url(session, session.overlay_preview_path)
                    if session.overlay_preview_path and session.overlay_preview_path.exists()
                    else None
                ),
                "maskPreviewUrl": (
                    self._media_url(session, session.mask_preview_path)
                    if session.mask_preview_path and session.mask_preview_path.exists()
                    else None
                ),
                "maskPostprocess": {
                    "mode": session.mask_postprocess_options.mode,
                    "trimapErodePx": session.mask_postprocess_options.trimap_erode_px,
                    "trimapDilatePx": session.mask_postprocess_options.trimap_dilate_px,
                    "vitmatteDevice": session.mask_postprocess_options.vitmatte_device,
                    "videomamaMaxResolution": session.mask_postprocess_options.videomama_max_resolution,
                    "videomamaChunkFrames": session.mask_postprocess_options.videomama_chunk_frames,
                    "videomamaOverlapFrames": session.mask_postprocess_options.videomama_overlap_frames,
                },
            }
        return snapshot

    def _propagation_cache_dir(self, session: WebSession) -> Path:
        return session.session_dir / PROPAGATION_CACHE_DIRNAME

    def _clear_propagation_cache(self, session: WebSession) -> None:
        cache_dir = self._propagation_cache_dir(session)
        if cache_dir.is_dir():
            shutil.rmtree(cache_dir, ignore_errors=True)

    def _flush_propagation_cache_batch(
        self, cache_dir: Path, pending_batch: dict[int, dict[str, Any]]
    ) -> int:
        if not pending_batch:
            return 0
        cache_dir.mkdir(parents=True, exist_ok=True)
        for frame_index, outputs in sorted(pending_batch.items()):
            np.savez_compressed(
                outputs_cache_path(cache_dir, frame_index),
                **serialize_outputs(outputs),
            )
        written = len(pending_batch)
        pending_batch.clear()
        gc.collect()
        return written

    def _clear_prompt_artifacts(self, session: WebSession) -> None:
        for file_path in [
            session.prompt_mask_rgba_path,
            session.prompt_mask_bw_path,
        ]:
            if file_path is not None and file_path.exists():
                file_path.unlink(missing_ok=True)
        with session.lock:
            session.prompt_mask_rgba_path = None
            session.prompt_mask_bw_path = None
            session.overlay_preview_path = None
            session.overlay_source_path = None
            session.mask_preview_path = None
            session.mask_source_path = None
        self._clear_propagation_cache(session)

    def create_session_from_upload(
        self,
        file_obj: BinaryIO,
        filename: str,
        previous_session_id: str | None = None,
        sam_max_inference_pixels: int | str | None = None,
    ) -> dict[str, Any]:
        if previous_session_id:
            self.close_session(previous_session_id, raise_if_missing=False)

        session_id = uuid.uuid4().hex
        session_dir = self.sessions_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        video_name = safe_filename(filename)
        video_path = session_dir / video_name

        with video_path.open("wb") as handle:
            shutil.copyfileobj(file_obj, handle, length=1024 * 1024)

        frame_count, fps, width, height = open_video(video_path)
        console_info(
            f"已导入视频：{video_name} ({width}x{height}, {frame_count} 帧, {fps:.2f} fps)"
        )
        max_inference_pixels = validate_sam_max_inference_pixels(
            sam_max_inference_pixels,
            default=self.default_sam_max_inference_pixels,
        )
        sam_width, sam_height, sam_resized = fit_size_to_max_pixels(
            width,
            height,
            max_inference_pixels,
        )
        sam_video_path = video_path
        if sam_resized:
            sam_video_path = session_dir / (
                f"{video_path.stem}_sam31_{sam_width}x{sam_height}.mp4"
            )
            resize_video_for_sam(
                video_path,
                sam_video_path,
                frame_count,
                fps,
                sam_width,
                sam_height,
            )
        else:
            console_info(
                f"SAM 推理将直接使用原视频尺寸：{sam_width}x{sam_height}"
            )
        offload_video_to_cpu = True
        offload_state_to_cpu = should_offload_state_to_cpu(
            frame_count,
            sam_width,
            sam_height,
        )
        session_channel = f"session:{session_id}"
        console_update("正在启动 SAM 3.1 会话...", channel=session_channel)
        response = self._run_inference(
            lambda predictor: predictor.handle_request(
                {
                    "type": "start_session",
                    "resource_path": str(sam_video_path),
                    "offload_video_to_cpu": offload_video_to_cpu,
                    "offload_state_to_cpu": offload_state_to_cpu,
                }
            )
        )
        console_finish("SAM 3.1 会话已就绪。", channel=session_channel)

        session = WebSession(
            session_id=session_id,
            predictor_session_id=response["session_id"],
            video_path=video_path,
            sam_video_path=sam_video_path,
            session_dir=session_dir,
            frame_count=frame_count,
            fps=fps,
            width=width,
            height=height,
            sam_width=sam_width,
            sam_height=sam_height,
            sam_max_inference_pixels=max_inference_pixels,
            sam_resized=sam_resized,
            predictor_offload_video_to_cpu=offload_video_to_cpu,
            predictor_offload_state_to_cpu=offload_state_to_cpu,
            predictor_generation=self._predictor_generation,
            message="视频已上传。请选择一帧并添加提示。",
        )
        with self._sessions_lock:
            self._sessions[session_id] = session
        console_info(
            f"会话已创建：{session_id}，SAM 推理尺寸 {sam_width}x{sam_height}"
        )
        return self._snapshot(session)

    def close_session(
        self, session_id: str, raise_if_missing: bool = True
    ) -> dict[str, Any] | None:
        with self._sessions_lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            if raise_if_missing:
                raise KeyError(f"未找到会话：{session_id}")
            return None
        if session.worker is not None and session.worker.is_alive():
            raise RuntimeError("传播运行中，无法关闭会话。")
        try:
            with session.lock:
                predictor_session_id = session.predictor_session_id
                predictor_generation = session.predictor_generation
            if (
                predictor_session_id is not None
                and predictor_generation == self._predictor_generation
            ):
                self._run_inference(
                    lambda predictor: predictor.handle_request(
                        {"type": "close_session", "session_id": predictor_session_id}
                    )
                )
        except Exception:
            pass
        with session.lock:
            session.status = "closed"
            session.message = "会话已关闭。"
            session.updated_at = time.time()
        console_info(f"会话已关闭：{session_id}")
        return self._snapshot(session)

    def reset_session(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        if session.worker is not None and session.worker.is_alive():
            raise RuntimeError("传播运行中，无法重置。")
        predictor_session_id = self._ensure_predictor_session(
            session,
            restore_prompts=False,
        )
        self._run_inference(
            lambda predictor: predictor.handle_request(
                {
                    "type": "reset_session",
                    "session_id": predictor_session_id,
                }
            )
        )
        self._clear_prompt_artifacts(session)
        with session.lock:
            session.status = "idle"
            session.message = "提示状态已清空。"
            session.progress_current = 0
            session.progress_total = 0
            session.prompt_frame_index = None
            session.prompt_mode = None
            session.prompt_summary = {}
            session.last_prompt_outputs = None
            session.last_prompt_request = None
            session.prompt_keyframes.clear()
            session.prompt_keyframe_requests.clear()
            session.prompt_mask_rgba_path = None
            session.prompt_mask_bw_path = None
            session.overlay_preview_path = None
            session.overlay_source_path = None
            session.mask_preview_path = None
            session.mask_source_path = None
            session.mask_postprocess_options = MaskPostprocessOptions()
            session.error = None
            session.updated_at = time.time()
        console_info(f"会话已重置：{session_id}")
        return self._snapshot(session)

    def apply_prompt(
        self,
        session_id: str,
        frame_index: int,
        mode: str,
        points: list[dict[str, Any]] | None = None,
        box: dict[str, Any] | None = None,
        text_prompt: str | None = None,
        keyframe_enabled: bool = False,
    ) -> dict[str, Any]:
        session = self._get_session(session_id)
        console_info(f"正在应用提示：frame={frame_index}, mode={mode}")
        if session.worker is not None and session.worker.is_alive():
            raise RuntimeError("传播运行中，无法更新提示。")
        if frame_index < 0 or frame_index >= session.frame_count:
            raise ValueError("帧序号超出范围。")

        request: dict[str, Any] = {
            "type": "add_prompt",
            "session_id": session.predictor_session_id,
            "frame_index": int(frame_index),
        }

        use_keyframes = bool(keyframe_enabled and mode == "points")

        if mode == "points":
            valid_points = points or []
            if not valid_points:
                raise ValueError("点选模式至少需要一个点。")
            request.update(
                {
                    "points": [[float(p["x"]), float(p["y"])] for p in valid_points],
                    "point_labels": [int(p["label"]) for p in valid_points],
                    "obj_id": 1,
                    "rel_coordinates": True,
                    "clear_old_points": True,
                }
            )
        elif mode == "bbox":
            if not box:
                raise ValueError("框选模式需要一个框。")
            x = float(box["x"])
            y = float(box["y"])
            w = float(box["w"])
            h = float(box["h"])
            request.update(
                {
                    "points": [[x, y], [x + w, y + h]],
                    "point_labels": [2, 3],
                    "obj_id": 1,
                    "rel_coordinates": True,
                    "clear_old_points": True,
                }
            )
        elif mode == "text":
            prompt_text = (text_prompt or "").strip()
            if not prompt_text:
                raise ValueError("文字模式需要输入非空提示。")
            request["text"] = prompt_text
        else:
            raise ValueError(f"不支持的提示模式：{mode}")

        with session.lock:
            can_extend_keyframes = (
                use_keyframes
                and session.prompt_mode in (None, "points")
                and bool(session.prompt_keyframes)
            )
        reset_before_prompt = not can_extend_keyframes
        predictor_session_id = self._ensure_predictor_session(
            session,
            restore_prompts=can_extend_keyframes,
        )
        request["session_id"] = predictor_session_id
        prompt_request = copy.deepcopy(request)

        def _apply_prompt(predictor):
            if reset_before_prompt:
                predictor.handle_request(
                    {
                        "type": "reset_session",
                        "session_id": predictor_session_id,
                    }
                )
            return predictor.handle_request(request)

        response = self._run_inference(_apply_prompt)

        outputs = response["outputs"]
        prompt_summary = summarize_outputs(outputs)
        self._clear_prompt_artifacts(session)
        self._render_prompt_mask(session, frame_index, outputs)
        with session.lock:
            session.status = "prompted"
            session.message = "已确认当前帧遮罩，现在可以开始传播。"
            session.progress_current = 0
            session.progress_total = 0
            session.prompt_frame_index = frame_index
            session.prompt_mode = mode
            session.prompt_summary = prompt_summary
            session.last_prompt_outputs = outputs
            session.last_prompt_request = prompt_request
            session.keyframe_mode_enabled = use_keyframes
            if mode == "points":
                if reset_before_prompt or not use_keyframes:
                    session.prompt_keyframes.clear()
                    session.prompt_keyframe_requests.clear()
                session.prompt_keyframes[int(frame_index)] = outputs
                session.prompt_keyframe_requests[int(frame_index)] = prompt_request
            else:
                session.prompt_keyframes.clear()
                session.prompt_keyframe_requests.clear()
            session.error = None
            session.updated_at = time.time()
        console_info(
            f"提示已应用：frame={frame_index}, objects={prompt_summary['num_objects']}"
        )
        return {
            "session": self._snapshot(session),
            "prompt": {
                "frameIndex": frame_index,
                "summary": prompt_summary,
            },
        }

    def _render_prompt_mask(
        self, session: WebSession, frame_index: int, outputs: dict[str, Any]
    ) -> None:
        mask_frame = build_mask_frame(outputs, session.height, session.width)
        rgba_path = session.session_dir / "prompt_mask_rgba.png"
        bw_path = session.session_dir / "prompt_mask_bw.png"
        save_png(rgba_path, build_mask_rgba(mask_frame))
        save_png(bw_path, mask_frame)
        with session.lock:
            session.prompt_mask_rgba_path = rgba_path
            session.prompt_mask_bw_path = bw_path

    def delete_keyframe(self, session_id: str, frame_index: int) -> dict[str, Any]:
        session = self._get_session(session_id)
        if session.worker is not None and session.worker.is_alive():
            raise RuntimeError("传播运行中，无法删除关键帧。")
        if frame_index < 0 or frame_index >= session.frame_count:
            raise ValueError("帧序号超出范围。")

        keyframe_missing = False
        with session.lock:
            existed = (
                frame_index in session.prompt_keyframes
                or frame_index in session.prompt_keyframe_requests
            )
            if not existed:
                session.message = f"当前帧 {frame_index} 不是已确认关键帧。"
                session.updated_at = time.time()
                keyframe_missing = True
                current_prompt_frame = session.prompt_frame_index
                remaining_requests = []
            else:
                session.prompt_keyframes.pop(frame_index, None)
                session.prompt_keyframe_requests.pop(frame_index, None)
                remaining_requests = [
                    (idx, copy.deepcopy(request))
                    for idx, request in sorted(session.prompt_keyframe_requests.items())
                ]
                current_prompt_frame = session.prompt_frame_index

        if keyframe_missing:
            return {
                "session": self._snapshot(session),
                "deleted": False,
                "frameIndex": current_prompt_frame,
            }

        self._clear_prompt_artifacts(session)

        if not remaining_requests:
            predictor_session_id = self._ensure_predictor_session(
                session,
                restore_prompts=False,
            )
            self._run_inference(
                lambda predictor: predictor.handle_request(
                    {
                        "type": "reset_session",
                        "session_id": predictor_session_id,
                    }
                )
            )
            with session.lock:
                session.status = "idle"
                session.message = "关键帧已删除，当前没有已确认提示。"
                session.progress_current = 0
                session.progress_total = 0
                session.prompt_frame_index = None
                session.prompt_mode = None
                session.prompt_summary = {}
                session.last_prompt_outputs = None
                session.last_prompt_request = None
                session.prompt_keyframes.clear()
                session.prompt_keyframe_requests.clear()
                session.prompt_mask_rgba_path = None
                session.prompt_mask_bw_path = None
                session.overlay_preview_path = None
                session.overlay_source_path = None
                session.mask_preview_path = None
                session.mask_source_path = None
                session.error = None
                session.updated_at = time.time()
            return {
                "session": self._snapshot(session),
                "deleted": True,
                "frameIndex": None,
            }

        predictor_session_id = self._ensure_predictor_session(
            session,
            restore_prompts=False,
        )

        def _rebuild_keyframes(predictor):
            predictor.handle_request(
                {
                    "type": "reset_session",
                    "session_id": predictor_session_id,
                }
            )
            rebuilt: dict[int, dict[str, Any]] = {}
            for keyframe_index, request in remaining_requests:
                response = predictor.handle_request(
                    self._bind_request_to_predictor_session(
                        request,
                        predictor_session_id,
                    )
                )
                rebuilt[keyframe_index] = response["outputs"]
            return rebuilt

        rebuilt_outputs = self._run_inference(_rebuild_keyframes)
        selected_frame = max(rebuilt_outputs)
        selected_outputs = rebuilt_outputs[selected_frame]
        prompt_summary = summarize_outputs(selected_outputs)
        self._render_prompt_mask(session, selected_frame, selected_outputs)

        with session.lock:
            session.status = "prompted"
            session.message = f"已删除关键帧 {frame_index}。"
            session.progress_current = 0
            session.progress_total = 0
            session.prompt_frame_index = selected_frame
            session.prompt_mode = "points"
            session.prompt_summary = prompt_summary
            session.last_prompt_outputs = selected_outputs
            session.last_prompt_request = copy.deepcopy(
                session.prompt_keyframe_requests[selected_frame]
            )
            session.prompt_keyframes = rebuilt_outputs
            session.keyframe_mode_enabled = True
            session.overlay_preview_path = None
            session.overlay_source_path = None
            session.mask_preview_path = None
            session.mask_source_path = None
            session.error = None
            session.updated_at = time.time()
        return {
            "session": self._snapshot(session),
            "deleted": True,
            "frameIndex": selected_frame,
        }

    def start_propagation(
        self,
        session_id: str,
        preview_bitrate: str | None = None,
        mask_postprocess: dict[str, Any] | None = None,
        keyframe_enabled: bool | None = None,
    ) -> dict[str, Any]:
        session = self._get_session(session_id)
        bitrate = validate_bitrate(preview_bitrate)
        postprocess_options = validate_mask_postprocess_options(mask_postprocess)
        with session.lock:
            if session.last_prompt_outputs is None or session.prompt_frame_index is None:
                raise RuntimeError("传播前请先应用提示。")
            if session.worker is not None and session.worker.is_alive():
                raise RuntimeError("传播已经在运行。")
            use_keyframes = (
                session.keyframe_mode_enabled
                if keyframe_enabled is None
                else bool(keyframe_enabled)
            )
            use_keyframes = bool(
                use_keyframes and session.prompt_mode == "points" and session.prompt_keyframes
            )
            keyframe_count = len(session.prompt_keyframes) if use_keyframes else 0
            session.status = "propagating"
            session.message = (
                f"正在使用 {keyframe_count} 个关键帧传播遮罩..."
                if use_keyframes
                else "正在整段视频中传播遮罩..."
            )
            session.progress_current = 0
            session.progress_total = session.frame_count
            session.mask_postprocess_options = postprocess_options
            session.keyframe_mode_enabled = use_keyframes
            session.error = None
            session.updated_at = time.time()
            worker = threading.Thread(
                target=self._propagate_worker,
                args=(
                    session.session_id,
                    session.prompt_frame_index,
                    bitrate,
                    postprocess_options,
                    use_keyframes,
                ),
                daemon=True,
            )
            session.worker = worker
        console_info(
            f"开始传播：session={session_id}, keyframes={'on' if use_keyframes else 'off'}, "
            f"postprocess={postprocess_options.mode}, bitrate={bitrate}"
        )
        worker.start()
        return self._snapshot(session)

    def _propagate_worker(
        self,
        session_id: str,
        start_frame_index: int,
        preview_bitrate: str,
        postprocess_options: MaskPostprocessOptions,
        use_keyframes: bool,
    ) -> None:
        session = self._get_session(session_id)
        propagate_channel = f"propagate:{session_id}"
        render_channel = f"render:{session_id}"

        try:
            start_frame_index, outputs_by_frame = self._prepare_propagation_prompts(
                session, start_frame_index, use_keyframes
            )
            console_update(
                f"正在传播遮罩... 0/{session.frame_count}",
                channel=propagate_channel,
            )

            def _collect_outputs(predictor):
                collected: dict[int, dict[str, Any]] = {}
                for item in predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": session.predictor_session_id,
                        "propagation_direction": "both",
                        "start_frame_index": int(start_frame_index),
                    }
                ):
                    frame_index = int(item["frame_index"])
                    collected[frame_index] = item["outputs"]
                    with session.lock:
                        session.progress_current = min(
                            session.frame_count, len(outputs_by_frame) + len(collected)
                        )
                        session.message = (
                            f"正在传播... {session.progress_current}/{session.frame_count}"
                        )
                        session.updated_at = time.time()
                        console_update(
                            f"正在传播遮罩... {session.progress_current}/{session.frame_count}",
                            channel=propagate_channel,
                        )
                return collected

            outputs_by_frame.update(self._run_inference(_collect_outputs))
            console_finish(
                f"遮罩传播完成：{session.frame_count} 帧",
                channel=propagate_channel,
            )

            with session.lock:
                session.status = "rendering"
                session.progress_current = 0
                session.progress_total = session.frame_count
                if postprocess_options.mode == MASK_POSTPROCESS_VIDEOMAMA:
                    mask_label = "VideoMaMa 精修遮罩预览"
                elif postprocess_options.mode == MASK_POSTPROCESS_VITMATTE:
                    mask_label = "ViTMatte 精修遮罩预览"
                else:
                    mask_label = "黑白遮罩预览"
                session.message = f"正在渲染彩色叠加预览和{mask_label}..."
                session.updated_at = time.time()
            console_update(
                f"正在渲染预览... 0/{session.frame_count}",
                channel=render_channel,
            )

            self._render_preview_videos(
                session,
                outputs_by_frame,
                preview_bitrate,
                postprocess_options,
            )
            console_finish("预览视频渲染完成。", channel=render_channel)

            with session.lock:
                session.status = "completed"
                session.progress_current = session.frame_count
                session.progress_total = session.frame_count
                session.message = "预览视频已准备好。"
                session.updated_at = time.time()
            console_info(f"传播流程完成：session={session_id}")
        except Exception as exc:
            console_error(f"传播流程失败：session={session_id}，{type(exc).__name__}: {exc}")
            with session.lock:
                session.status = "error"
                session.error = str(exc)
                session.message = str(exc)
                session.updated_at = time.time()
        finally:
            with session.lock:
                session.worker = None

    def _prepare_propagation_prompts(
        self, session: WebSession, start_frame_index: int, use_keyframes: bool
    ) -> tuple[int, dict[int, dict[str, Any]]]:
        with session.lock:
            last_request = copy.deepcopy(session.last_prompt_request)
            last_outputs = session.last_prompt_outputs
            keyframe_outputs = dict(session.prompt_keyframes)
            prompt_mode = session.prompt_mode
            prompt_frame_index = session.prompt_frame_index

        if prompt_frame_index is None or last_outputs is None:
            raise RuntimeError("传播前请先应用提示。")

        predictor_session_id = self._ensure_predictor_session(
            session,
            restore_prompts=True,
        )

        if use_keyframes and prompt_mode == "points" and keyframe_outputs:
            with session.lock:
                session.keyframe_mode_enabled = True
            return int(start_frame_index), keyframe_outputs

        if len(keyframe_outputs) <= 1:
            return int(prompt_frame_index), {int(prompt_frame_index): last_outputs}

        if last_request is None:
            return int(start_frame_index), {int(prompt_frame_index): last_outputs}

        def _reset_and_reapply(predictor):
            predictor.handle_request(
                {
                    "type": "reset_session",
                    "session_id": predictor_session_id,
                }
            )
            return predictor.handle_request(
                self._bind_request_to_predictor_session(
                    last_request,
                    predictor_session_id,
                )
            )

        response = self._run_inference(_reset_and_reapply)
        outputs = response["outputs"]
        prompt_summary = summarize_outputs(outputs)
        frame_index = int(response.get("frame_index", prompt_frame_index))
        with session.lock:
            session.keyframe_mode_enabled = False
            session.prompt_frame_index = frame_index
            session.prompt_summary = prompt_summary
            session.last_prompt_outputs = outputs
            session.prompt_keyframes.clear()
            if session.prompt_mode == "points":
                session.prompt_keyframes[frame_index] = outputs
            session.updated_at = time.time()
        return frame_index, {frame_index: outputs}

    def _render_preview_videos(
        self,
        session: WebSession,
        outputs_by_frame: dict[int, dict[str, Any]],
        preview_bitrate: str,
        postprocess_options: MaskPostprocessOptions,
    ) -> None:
        render_channel = f"render:{session.session_id}"
        overlay_source = session.session_dir / "overlay_preview_source.mp4"
        overlay_preview = session.session_dir / "overlay_preview.mp4"
        mask_source = session.session_dir / "mask_preview_source.mp4"
        mask_preview = session.session_dir / "mask_preview.mp4"

        for file_path in [overlay_source, overlay_preview, mask_source, mask_preview]:
            file_path.unlink(missing_ok=True)

        cap = cv2.VideoCapture(str(session.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{session.video_path}")

        overlay_writer = cv2.VideoWriter(
            str(overlay_source),
            cv2.VideoWriter_fourcc(*"mp4v"),
            session.fps,
            (session.width, session.height),
        )
        mask_writer = cv2.VideoWriter(
            str(mask_source),
            cv2.VideoWriter_fourcc(*"mp4v"),
            session.fps,
            (session.width, session.height),
            isColor=False,
        )

        if not overlay_writer.isOpened():
            raise RuntimeError(f"创建彩色叠加视频失败：{overlay_source}")
        if not mask_writer.isOpened():
            raise RuntimeError(f"创建遮罩视频失败：{mask_source}")

        try:
            for frame_idx in range(session.frame_count):
                ok, frame_bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"读取第 {frame_idx} 帧时失败。")
                outputs = outputs_by_frame.get(frame_idx)
                if outputs is None:
                    outputs = empty_outputs(session.height, session.width)
                overlay_writer.write(draw_outputs(frame_bgr, outputs))
                mask_frame = build_mask_frame(outputs, session.height, session.width)
                refined_mask_frame = self._vitmatte_refiner.refine(
                    frame_bgr,
                    mask_frame,
                    postprocess_options,
                )
                mask_writer.write(refined_mask_frame)
                if frame_idx == session.frame_count - 1 or (
                    frame_idx > 0 and frame_idx % 10 == 0
                ):
                    with session.lock:
                        session.progress_current = frame_idx + 1
                        session.message = (
                            f"正在渲染预览... {frame_idx + 1}/{session.frame_count}"
                        )
                        session.updated_at = time.time()
                    if current_count == session.frame_count or current_count % 10 == 0:
                        console_update(
                        f"正在渲染预览... {frame_idx + 1}/{session.frame_count}",
                        channel=render_channel,
                        )
        finally:
            cap.release()
            overlay_writer.release()
            mask_writer.release()

        with session.lock:
            session.progress_current = session.frame_count
            session.progress_total = session.frame_count
            session.message = "正在进行 H.264 重编码..."
            session.updated_at = time.time()
        console_update("正在进行 H.264 重编码...", channel=render_channel)
        reencode_h264(overlay_source, overlay_preview, preview_bitrate)
        reencode_h264(mask_source, mask_preview, preview_bitrate)
        with session.lock:
            session.overlay_source_path = overlay_source
            session.overlay_preview_path = overlay_preview
            session.mask_source_path = mask_source
            session.mask_preview_path = mask_preview
            session.progress_current = session.frame_count
            session.progress_total = session.frame_count
            session.message = "预览视频已准备好。"
            session.updated_at = time.time()

    def _propagate_worker(
        self,
        session_id: str,
        start_frame_index: int,
        preview_bitrate: str,
        postprocess_options: MaskPostprocessOptions,
        use_keyframes: bool,
    ) -> None:
        session = self._get_session(session_id)
        propagate_channel = f"propagate:{session_id}"
        render_channel = f"render:{session_id}"

        try:
            self._clear_propagation_cache(session)
            cache_dir = self._propagation_cache_dir(session)
            start_frame_index, seed_outputs = self._prepare_propagation_prompts(
                session, start_frame_index, use_keyframes
            )
            batch_frame_limit, batch_byte_limit = choose_propagation_batch_limits(
                session.frame_count,
                session.sam_width,
                session.sam_height,
            )
            written_frames = set(seed_outputs)
            if seed_outputs:
                self._flush_propagation_cache_batch(cache_dir, dict(seed_outputs))
            console_update(
                f"正在传播遮罩... 0/{session.frame_count}",
                channel=propagate_channel,
            )

            predictor_session_id = self._ensure_predictor_session(
                session,
                restore_prompts=True,
            )

            def _collect_outputs(predictor):
                pending_batch: dict[int, dict[str, Any]] = {}
                pending_bytes = 0
                for item in predictor.handle_stream_request(
                    {
                        "type": "propagate_in_video",
                        "session_id": predictor_session_id,
                        "propagation_direction": "both",
                        "start_frame_index": int(start_frame_index),
                    }
                ):
                    frame_index = int(item["frame_index"])
                    pending_batch[frame_index] = item["outputs"]
                    pending_bytes = sum(
                        estimate_outputs_bytes(outputs)
                        for outputs in pending_batch.values()
                    )
                    current_count = len(written_frames.union(pending_batch.keys()))
                    with session.lock:
                        session.progress_current = min(session.frame_count, current_count)
                        session.message = (
                            f"正在传播... {session.progress_current}/{session.frame_count}"
                        )
                        session.updated_at = time.time()
                    if current_count == session.frame_count or current_count % 10 == 0:
                        console_update(
                        f"正在传播遮罩... {current_count}/{session.frame_count}",
                        channel=propagate_channel,
                        )
                    if (
                        len(pending_batch) >= batch_frame_limit
                        or pending_bytes >= batch_byte_limit
                    ):
                        written_frames.update(pending_batch.keys())
                        self._flush_propagation_cache_batch(cache_dir, pending_batch)
                        pending_bytes = 0
                if pending_batch:
                    written_frames.update(pending_batch.keys())
                    self._flush_propagation_cache_batch(cache_dir, pending_batch)
                return len(written_frames)

            cached_frame_count = self._run_inference(_collect_outputs)
            console_finish(
                f"遮罩传播完成：{cached_frame_count} 帧",
                channel=propagate_channel,
            )

            with session.lock:
                session.status = "rendering"
                session.progress_current = 0
                session.progress_total = session.frame_count
                if postprocess_options.mode == MASK_POSTPROCESS_VIDEOMAMA:
                    mask_label = "VideoMaMa 精修遮罩预览"
                elif postprocess_options.mode == MASK_POSTPROCESS_VITMATTE:
                    mask_label = "ViTMatte 精修遮罩预览"
                else:
                    mask_label = "黑白遮罩预览"
                session.message = f"正在渲染彩色叠加预览和{mask_label}..."
                session.updated_at = time.time()
            console_update(
                f"正在渲染预览... 0/{session.frame_count}",
                channel=render_channel,
            )

            self._render_preview_videos(
                session,
                cache_dir,
                preview_bitrate,
                postprocess_options,
            )
            self._clear_propagation_cache(session)
            console_finish("预览视频渲染完成。", channel=render_channel)

            with session.lock:
                session.status = "completed"
                session.progress_current = session.frame_count
                session.progress_total = session.frame_count
                session.message = "预览视频已准备好。"
                session.updated_at = time.time()
            console_info(f"传播流程完成：session={session_id}")
        except Exception as exc:
            console_error(
                f"传播流程失败：session={session_id}，{type(exc).__name__}: {exc}"
            )
            with session.lock:
                session.status = "error"
                session.error = str(exc)
                session.message = str(exc)
                session.updated_at = time.time()
        finally:
            with session.lock:
                session.worker = None

    def _render_preview_videos(
        self,
        session: WebSession,
        cache_dir: Path,
        preview_bitrate: str,
        postprocess_options: MaskPostprocessOptions,
    ) -> None:
        render_channel = f"render:{session.session_id}"
        overlay_source, overlay_preview, mask_source, mask_preview = (
            self._allocate_preview_paths(session)
        )

        cap = cv2.VideoCapture(str(session.video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频：{session.video_path}")
        sam_cap = None
        videomama_frames: list[np.ndarray] = []
        videomama_masks: list[np.ndarray] = []

        def _write_refined_mask(refined_sam_mask: np.ndarray) -> None:
            if session.sam_resized:
                if postprocess_options.uses_soft_matting:
                    refined_mask_frame = resize_gray_frame_lanczos(
                        refined_sam_mask,
                        session.width,
                        session.height,
                    )
                else:
                    refined_mask_frame = (
                        cv2.resize(
                            refined_sam_mask,
                            (session.width, session.height),
                            interpolation=cv2.INTER_NEAREST,
                        )
                        .clip(0, 255)
                        .astype(np.uint8)
                    )
            else:
                refined_mask_frame = refined_sam_mask
            mask_writer.write(refined_mask_frame)

        def _update_render_progress(current_count: int, message: str) -> None:
            with session.lock:
                session.progress_current = min(session.frame_count, current_count)
                session.message = message
                session.updated_at = time.time()
            console_update(message, channel=render_channel)
        if session.sam_resized:
            sam_cap = cv2.VideoCapture(str(session.sam_video_path))
            if not sam_cap.isOpened():
                cap.release()
                raise RuntimeError(f"无法打开 SAM 推理视频：{session.sam_video_path}")

        overlay_writer = cv2.VideoWriter(
            str(overlay_source),
            cv2.VideoWriter_fourcc(*"mp4v"),
            session.fps,
            (session.width, session.height),
        )
        mask_writer = cv2.VideoWriter(
            str(mask_source),
            cv2.VideoWriter_fourcc(*"mp4v"),
            session.fps,
            (session.width, session.height),
            isColor=False,
        )

        if not overlay_writer.isOpened():
            raise RuntimeError(f"创建彩色叠加视频失败：{overlay_source}")
        if not mask_writer.isOpened():
            raise RuntimeError(f"创建遮罩视频失败：{mask_source}")

        try:
            for frame_idx in range(session.frame_count):
                ok, frame_bgr = cap.read()
                if not ok:
                    raise RuntimeError(f"读取第 {frame_idx} 帧时失败。")
                if sam_cap is not None:
                    ok_sam, sam_frame_bgr = sam_cap.read()
                    if not ok_sam:
                        raise RuntimeError(f"读取第 {frame_idx} 帧的 SAM 视频时失败。")
                else:
                    sam_frame_bgr = frame_bgr

                outputs = load_cached_outputs(
                    cache_dir,
                    frame_idx,
                    session.sam_height,
                    session.sam_width,
                )
                overlay_writer.write(draw_outputs(frame_bgr, outputs))
                sam_mask_frame = build_mask_frame(
                    outputs,
                    session.sam_height,
                    session.sam_width,
                )
                if postprocess_options.uses_videomama:
                    videomama_frames.append(sam_frame_bgr.copy())
                    videomama_masks.append(sam_mask_frame.copy())
                else:
                    refined_sam_mask = self._vitmatte_refiner.refine(
                        sam_frame_bgr,
                        sam_mask_frame,
                        postprocess_options,
                    )
                    _write_refined_mask(refined_sam_mask)

                if (
                    postprocess_options.uses_videomama
                    and frame_idx == session.frame_count - 1
                ):
                    before_free = get_available_gpu_memory_bytes()
                    released_sam = self._inference_worker.release_predictor()
                    if released_sam:
                        self._invalidate_predictor_sessions()
                    after_free = get_available_gpu_memory_bytes()
                    if after_free is not None:
                        freed_gib = (
                            (after_free - before_free) / float(1024**3)
                            if before_free is not None
                            else 0.0
                        )
                        console_info(
                            "VideoMaMa 前显存整理完成："
                            f"释放SAM={'yes' if released_sam else 'no'}, "
                            f"可用显存={after_free / float(1024**3):.2f} GiB, "
                            f"增加={freed_gib:.2f} GiB"
                        )
                    _update_render_progress(
                        0,
                        f"正在使用 VideoMaMa 精修遮罩... 0/{session.frame_count}",
                    )

                    def _videomama_progress(done: int, total: int) -> None:
                        _update_render_progress(
                            done,
                            f"正在使用 VideoMaMa 精修遮罩... {done}/{total}",
                        )

                    def _videomama_plan(plan: dict[str, Any]) -> None:
                        starts = plan.get("starts") or []
                        start_text = ",".join(str(start) for start in starts[:12])
                        if len(starts) > 12:
                            start_text += ",..."
                        message = (
                            "正在使用 VideoMaMa 精修遮罩... "
                            f"{plan['target_width']}x{plan['target_height']}, "
                            f"{plan['chunk_frames']}帧/段, "
                            f"重叠{plan['overlap_frames']}帧"
                        )
                        console_info(
                            f"VideoMaMa 分段计划：{message}，starts=[{start_text}]"
                        )
                        _update_render_progress(0, message)

                    refined_masks = self._videomama_refiner.refine_video(
                        videomama_frames,
                        videomama_masks,
                        postprocess_options,
                        progress_callback=_videomama_progress,
                        plan_callback=_videomama_plan,
                    )
                    for refined_sam_mask in refined_masks:
                        _write_refined_mask(refined_sam_mask)

                if frame_idx == session.frame_count - 1 or (
                    frame_idx > 0 and frame_idx % 10 == 0
                ):
                    with session.lock:
                        session.progress_current = frame_idx + 1
                        session.message = (
                            f"正在渲染预览... {frame_idx + 1}/{session.frame_count}"
                        )
                        session.updated_at = time.time()
                    console_update(
                        f"正在渲染预览... {frame_idx + 1}/{session.frame_count}",
                        channel=render_channel,
                    )
        finally:
            cap.release()
            if sam_cap is not None:
                sam_cap.release()
            overlay_writer.release()
            mask_writer.release()
            gc.collect()

        console_update("正在进行 H.264 重编码...", channel=render_channel)
        reencode_h264(overlay_source, overlay_preview, preview_bitrate)
        reencode_h264(mask_source, mask_preview, preview_bitrate)
        with session.lock:
            session.overlay_source_path = overlay_source
            session.overlay_preview_path = overlay_preview
            session.mask_source_path = mask_source
            session.mask_preview_path = mask_preview

    def export_mask_video(self, session_id: str, bitrate: str | None) -> dict[str, Any]:
        session = self._get_session(session_id)
        target_bitrate = validate_bitrate(bitrate)
        with session.lock:
            source_path = session.mask_source_path or session.mask_preview_path
            if source_path is None or not source_path.exists():
                raise RuntimeError("遮罩预览尚未准备好。")
        exports_dir = session.session_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        output_name = (
            f"{session.video_path.stem}_遮罩_{target_bitrate.lower()}_{int(time.time())}.mp4"
        )
        output_path = exports_dir / output_name
        console_info(
            f"正在导出遮罩视频：session={session_id}, bitrate={target_bitrate}"
        )
        reencode_h264(source_path, output_path, target_bitrate)
        console_info(f"遮罩视频已导出：{output_name}")
        return {
            "downloadUrl": self._media_url(session, output_path),
            "fileName": output_name,
            "bitrate": target_bitrate,
        }

    def get_status(self, session_id: str) -> dict[str, Any]:
        return self._snapshot(self._get_session(session_id))
