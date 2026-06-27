from __future__ import annotations

import contextlib
import gc
import inspect
import io
import os
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

PIL_LANCZOS = (
    Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS
)


MASK_POSTPROCESS_BINARY = "binary"
MASK_POSTPROCESS_VITMATTE = "vitmatte"
MASK_POSTPROCESS_VIDEOMAMA = "videomama"
DEFAULT_MASK_POSTPROCESS_MODE = MASK_POSTPROCESS_VIDEOMAMA
VITMATTE_DEVICE_GPU = "gpu"
VITMATTE_DEVICE_CPU = "cpu"
DEFAULT_TRIMAP_ERODE_PX = 12
DEFAULT_TRIMAP_DILATE_PX = 16
MAX_TRIMAP_WIDTH_PX = 256
DEFAULT_VIDEOMAMA_MAX_RESOLUTION = 1024
MIN_VIDEOMAMA_MAX_RESOLUTION = 256
MAX_VIDEOMAMA_MAX_RESOLUTION = 2048
DEFAULT_VIDEOMAMA_CHUNK_FRAMES = 0
MAX_VIDEOMAMA_CHUNK_FRAMES = 512
DEFAULT_VIDEOMAMA_OVERLAP_FRAMES = 4
DEFAULT_VIDEOMAMA_SEED = 42
DEFAULT_VIDEOMAMA_FPS = 7
DEFAULT_VIDEOMAMA_MOTION_BUCKET_ID = 127
DEFAULT_VIDEOMAMA_NOISE_AUG_STRENGTH = 0.0
DEFAULT_VIDEOMAMA_TARGET_VRAM_FRACTION = 0.90
DEFAULT_VIDEOMAMA_RESERVED_VRAM_MB = 2048
DEFAULT_VIDEOMAMA_FRAME_VRAM_MB_AT_1024 = 1024
DEFAULT_VIDEOMAMA_FALLBACK_CHUNK_FRAMES = 12


def _read_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _read_env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def _read_default_vitmatte_device() -> str:
    device = str(os.environ.get("SAM31_VITMATTE_DEVICE") or VITMATTE_DEVICE_GPU)
    device = device.strip().lower()
    if device == "cuda":
        device = VITMATTE_DEVICE_GPU
    if device not in {VITMATTE_DEVICE_GPU, VITMATTE_DEVICE_CPU}:
        return VITMATTE_DEVICE_GPU
    return device


DEFAULT_VITMATTE_DEVICE = _read_default_vitmatte_device()
VIDEOMAMA_TARGET_VRAM_FRACTION = _read_env_float(
    "SAM31_VIDEOMAMA_TARGET_VRAM_FRACTION",
    DEFAULT_VIDEOMAMA_TARGET_VRAM_FRACTION,
    0.50,
    0.98,
)
VIDEOMAMA_RESERVED_VRAM_BYTES = (
    _read_env_int(
        "SAM31_VIDEOMAMA_RESERVED_VRAM_MB",
        DEFAULT_VIDEOMAMA_RESERVED_VRAM_MB,
        512,
        8192,
    )
    * 1024
    * 1024
)
VIDEOMAMA_FRAME_VRAM_BYTES_AT_1024 = (
    _read_env_int(
        "SAM31_VIDEOMAMA_FRAME_VRAM_MB_AT_1024",
        DEFAULT_VIDEOMAMA_FRAME_VRAM_MB_AT_1024,
        256,
        4096,
    )
    * 1024
    * 1024
)


@dataclass(frozen=True)
class MaskPostprocessOptions:
    mode: str = DEFAULT_MASK_POSTPROCESS_MODE
    trimap_erode_px: int = DEFAULT_TRIMAP_ERODE_PX
    trimap_dilate_px: int = DEFAULT_TRIMAP_DILATE_PX
    vitmatte_device: str = DEFAULT_VITMATTE_DEVICE
    videomama_max_resolution: int = _read_env_int(
        "SAM31_VIDEOMAMA_MAX_RESOLUTION",
        DEFAULT_VIDEOMAMA_MAX_RESOLUTION,
        MIN_VIDEOMAMA_MAX_RESOLUTION,
        MAX_VIDEOMAMA_MAX_RESOLUTION,
    )
    videomama_chunk_frames: int = _read_env_int(
        "SAM31_VIDEOMAMA_CHUNK_FRAMES",
        DEFAULT_VIDEOMAMA_CHUNK_FRAMES,
        0,
        MAX_VIDEOMAMA_CHUNK_FRAMES,
    )
    videomama_overlap_frames: int = _read_env_int(
        "SAM31_VIDEOMAMA_OVERLAP_FRAMES",
        DEFAULT_VIDEOMAMA_OVERLAP_FRAMES,
        0,
        MAX_VIDEOMAMA_CHUNK_FRAMES,
    )

    @property
    def uses_vitmatte(self) -> bool:
        return self.mode == MASK_POSTPROCESS_VITMATTE

    @property
    def uses_videomama(self) -> bool:
        return self.mode == MASK_POSTPROCESS_VIDEOMAMA

    @property
    def uses_soft_matting(self) -> bool:
        return self.uses_vitmatte or self.uses_videomama

    @property
    def vitmatte_torch_device(self) -> str:
        return "cuda" if self.vitmatte_device == VITMATTE_DEVICE_GPU else "cpu"


def _coerce_int_range(
    value: Any,
    default: int,
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if value is None or value == "":
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _coerce_int(value: Any, default: int, name: str) -> int:
    return _coerce_int_range(value, default, name, 0, MAX_TRIMAP_WIDTH_PX)


def _coerce_vitmatte_device(value: Any, default: str) -> str:
    device = str(value or default).strip().lower()
    if device == "cuda":
        device = VITMATTE_DEVICE_GPU
    if device not in {VITMATTE_DEVICE_GPU, VITMATTE_DEVICE_CPU}:
        raise ValueError(
            f"ViTMatte device must be {VITMATTE_DEVICE_GPU} or {VITMATTE_DEVICE_CPU}."
        )
    return device


def validate_mask_postprocess_options(
    value: dict[str, Any] | None,
) -> MaskPostprocessOptions:
    if not value:
        return MaskPostprocessOptions()

    mode = str(value.get("mode") or DEFAULT_MASK_POSTPROCESS_MODE).strip().lower()
    if mode not in {
        MASK_POSTPROCESS_BINARY,
        MASK_POSTPROCESS_VITMATTE,
        MASK_POSTPROCESS_VIDEOMAMA,
    }:
        raise ValueError(f"Unsupported mask postprocess mode: {mode}")

    return MaskPostprocessOptions(
        mode=mode,
        trimap_erode_px=_coerce_int(
            value.get("trimapErodePx"),
            DEFAULT_TRIMAP_ERODE_PX,
            "Trimap erode width",
        ),
        trimap_dilate_px=_coerce_int(
            value.get("trimapDilatePx"),
            DEFAULT_TRIMAP_DILATE_PX,
            "Trimap dilate width",
        ),
        vitmatte_device=_coerce_vitmatte_device(
            value.get("vitmatteDevice"),
            DEFAULT_VITMATTE_DEVICE,
        ),
        videomama_max_resolution=_coerce_int_range(
            value.get("videomamaMaxResolution"),
            DEFAULT_VIDEOMAMA_MAX_RESOLUTION,
            "VideoMaMa max resolution",
            MIN_VIDEOMAMA_MAX_RESOLUTION,
            MAX_VIDEOMAMA_MAX_RESOLUTION,
        ),
        videomama_chunk_frames=_coerce_int_range(
            value.get("videomamaChunkFrames"),
            DEFAULT_VIDEOMAMA_CHUNK_FRAMES,
            "VideoMaMa chunk frames",
            0,
            MAX_VIDEOMAMA_CHUNK_FRAMES,
        ),
        videomama_overlap_frames=_coerce_int_range(
            value.get("videomamaOverlapFrames"),
            DEFAULT_VIDEOMAMA_OVERLAP_FRAMES,
            "VideoMaMa overlap frames",
            0,
            MAX_VIDEOMAMA_CHUNK_FRAMES,
        ),
    )


def _ellipse_kernel(radius_px: int) -> np.ndarray | None:
    if radius_px <= 0:
        return None
    size = radius_px * 2 + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def build_vitmatte_trimap(
    mask_frame: np.ndarray,
    erode_px: int,
    dilate_px: int,
) -> np.ndarray:
    mask = np.asarray(mask_frame) > 127
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.uint8)
    if mask.all():
        return np.full(mask.shape, 255, dtype=np.uint8)

    mask_u8 = mask.astype(np.uint8)
    erode_kernel = _ellipse_kernel(erode_px)
    dilate_kernel = _ellipse_kernel(dilate_px)
    confident_foreground = (
        cv2.erode(mask_u8, erode_kernel, iterations=1).astype(bool)
        if erode_kernel is not None
        else mask
    )
    possible_foreground = (
        cv2.dilate(mask_u8, dilate_kernel, iterations=1).astype(bool)
        if dilate_kernel is not None
        else mask
    )

    trimap = np.zeros(mask.shape, dtype=np.uint8)
    trimap[possible_foreground] = 128
    trimap[confident_foreground] = 255
    return trimap


class VitMatteRefiner:
    def __init__(self, model_dir: Path) -> None:
        self.model_dir = Path(model_dir)
        self._lock = threading.Lock()
        self._model = None
        self._processor = None
        self._device: torch.device | None = None

    def refine(
        self,
        frame_bgr: np.ndarray,
        mask_frame: np.ndarray,
        options: MaskPostprocessOptions,
    ) -> np.ndarray:
        if not options.uses_vitmatte:
            return mask_frame
        if not np.any(mask_frame):
            return np.zeros_like(mask_frame, dtype=np.uint8)
        if np.all(mask_frame > 127):
            return np.full_like(mask_frame, 255, dtype=np.uint8)

        trimap = build_vitmatte_trimap(
            mask_frame,
            options.trimap_erode_px,
            options.trimap_dilate_px,
        )
        with self._lock:
            self._ensure_loaded(options.vitmatte_torch_device)
            return self._run_model(frame_bgr, trimap)

    def _ensure_loaded(self, device_name: str) -> None:
        target_device = torch.device(device_name)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("ViTMatte is set to GPU, but CUDA is not available.")

        if self._model is not None and self._processor is not None:
            if self._device is not None and self._device.type != target_device.type:
                self._model.to(target_device)
                if self._device.type == "cuda" and target_device.type != "cuda":
                    torch.cuda.empty_cache()
                self._device = target_device
            return
        if not self.model_dir.is_dir():
            raise FileNotFoundError(f"ViTMatte model directory not found: {self.model_dir}")
        try:
            from transformers import VitMatteForImageMatting, VitMatteImageProcessor
        except ImportError as exc:
            raise RuntimeError(
                "ViTMatte postprocess requires the transformers package."
            ) from exc

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            processor = VitMatteImageProcessor.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
            )
            model = VitMatteForImageMatting.from_pretrained(
                str(self.model_dir),
                local_files_only=True,
            )
        model.to(target_device)
        model.eval()

        self._processor = processor
        self._model = model
        self._device = target_device

    def _run_model(self, frame_bgr: np.ndarray, trimap: np.ndarray) -> np.ndarray:
        if self._model is None or self._processor is None or self._device is None:
            raise RuntimeError("ViTMatte model is not loaded.")

        height, width = trimap.shape
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        trimap_image = Image.fromarray(trimap)
        inputs = self._processor(
            images=image,
            trimaps=trimap_image,
            return_tensors="pt",
        )
        inputs = {
            key: value.to(self._device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

        with torch.inference_mode():
            outputs = self._model(**inputs)

        alpha = self._postprocess_alpha(outputs, height, width)
        alpha[trimap == 0] = 0.0
        alpha[trimap == 255] = 1.0
        return np.clip(np.rint(alpha * 255.0), 0, 255).astype(np.uint8)

    def _postprocess_alpha(self, outputs: Any, height: int, width: int) -> np.ndarray:
        assert self._processor is not None
        if hasattr(self._processor, "post_process_image_matting"):
            alpha = self._processor.post_process_image_matting(
                outputs,
                target_sizes=[(height, width)],
            )[0]
            if isinstance(alpha, torch.Tensor):
                alpha = alpha.detach().cpu().numpy()
            return self._normalize_alpha(np.asarray(alpha, dtype=np.float32))

        alpha_tensor = outputs.alphas if hasattr(outputs, "alphas") else outputs[0]
        if alpha_tensor.ndim == 3:
            alpha_tensor = alpha_tensor[:, None, :, :]
        alpha_tensor = F.interpolate(
            alpha_tensor,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )
        alpha = alpha_tensor[0, 0].detach().cpu().numpy().astype(np.float32)
        return self._normalize_alpha(alpha)

    @staticmethod
    def _normalize_alpha(alpha: np.ndarray) -> np.ndarray:
        alpha = np.nan_to_num(alpha, nan=0.0, posinf=1.0, neginf=0.0)
        if alpha.size and float(np.nanmax(alpha)) > 1.0:
            alpha = alpha / 255.0
        return np.clip(alpha, 0.0, 1.0)


class VideoMaMaRefiner:
    def __init__(
        self,
        repo_dir: Path,
        base_model_dir: Path,
        checkpoint_dir: Path,
    ) -> None:
        self.repo_dir = Path(repo_dir)
        self.base_model_dir = Path(base_model_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        self._lock = threading.Lock()
        self._pipeline = None
        self._run_accepts_mask_cond_mode = False

    def refine_video(
        self,
        frames_bgr: list[np.ndarray],
        mask_frames: list[np.ndarray],
        options: MaskPostprocessOptions,
        progress_callback: Any | None = None,
        plan_callback: Any | None = None,
    ) -> list[np.ndarray]:
        if not options.uses_videomama:
            return [np.asarray(mask, dtype=np.uint8) for mask in mask_frames]
        if len(frames_bgr) != len(mask_frames):
            raise ValueError("VideoMaMa frame and mask counts must match.")
        if not frames_bgr:
            return []

        total = len(frames_bgr)
        overrides = [self._mask_override(mask_frame) for mask_frame in mask_frames]
        if all(mask is not None for mask in overrides):
            return [np.asarray(mask, dtype=np.uint8) for mask in overrides]

        height, width = mask_frames[0].shape[:2]
        target_width, target_height = self._compute_target_size(
            width,
            height,
            options.videomama_max_resolution,
        )
        self._clear_cuda_cache()
        with self._lock:
            self._ensure_loaded()
        self._clear_cuda_cache()

        requested_chunk_frames = options.videomama_chunk_frames or total
        requested_chunk_frames = max(1, min(total, requested_chunk_frames))
        chunk_frames = self._select_chunk_frames(
            total,
            requested_chunk_frames,
            target_width,
            target_height,
        )
        overlap_frames = max(
            0,
            min(options.videomama_overlap_frames, chunk_frames - 1),
        )

        if plan_callback is not None:
            starts = (
                self._build_chunk_starts(total, chunk_frames, overlap_frames)
                if chunk_frames < total and overlap_frames > 0
                else list(range(0, total, chunk_frames))
            )
            plan_callback(
                {
                    "chunk_frames": chunk_frames,
                    "overlap_frames": overlap_frames,
                    "target_width": target_width,
                    "target_height": target_height,
                    "starts": starts,
                }
            )

        if chunk_frames >= total or overlap_frames <= 0:
            refined: list[np.ndarray] = []
            for start in range(0, total, chunk_frames):
                end = min(total, start + chunk_frames)
                refined.extend(
                    self._run_chunk(
                        frames_bgr[start:end],
                        mask_frames[start:end],
                        options,
                    )
                )
                if progress_callback is not None:
                    progress_callback(end, total)
            return refined

        starts = self._build_chunk_starts(total, chunk_frames, overlap_frames)
        accumulators: list[np.ndarray | None] = [None] * total
        weight_sums = [0.0] * total
        for start in starts:
            end = min(total, start + chunk_frames)
            chunk_result = self._run_chunk(
                frames_bgr[start:end],
                mask_frames[start:end],
                options,
            )
            weights = self._blend_weights(
                len(chunk_result),
                start=start,
                end=end,
                total=total,
                overlap_frames=overlap_frames,
            )
            for local_index, (mask, weight) in enumerate(zip(chunk_result, weights)):
                frame_index = start + local_index
                if accumulators[frame_index] is None:
                    accumulators[frame_index] = np.zeros(
                        mask.shape,
                        dtype=np.float32,
                    )
                accumulators[frame_index] += mask.astype(np.float32) * weight
                weight_sums[frame_index] += weight
            if progress_callback is not None:
                progress_callback(end, total)

        blended: list[np.ndarray] = []
        for frame_index, accumulator in enumerate(accumulators):
            if accumulator is None or weight_sums[frame_index] <= 0.0:
                blended.append(np.asarray(mask_frames[frame_index], dtype=np.uint8))
                continue
            alpha = accumulator / weight_sums[frame_index]
            blended.append(np.clip(np.rint(alpha), 0, 255).astype(np.uint8))
        return blended

    @staticmethod
    def _build_chunk_starts(
        total: int,
        chunk_frames: int,
        overlap_frames: int,
    ) -> list[int]:
        if total <= chunk_frames:
            return [0]
        stride = max(1, chunk_frames - overlap_frames)
        starts = [0]
        while starts[-1] + chunk_frames < total:
            next_start = starts[-1] + stride
            if next_start + chunk_frames >= total:
                final_start = max(0, total - chunk_frames)
                if (
                    len(starts) >= 2
                    and final_start > starts[-1]
                    and final_start < starts[-2] + chunk_frames
                ):
                    starts[-1] = final_start
                    break
                next_start = final_start
            if next_start <= starts[-1]:
                break
            starts.append(next_start)
        return starts

    @staticmethod
    def _blend_weights(
        length: int,
        *,
        start: int,
        end: int,
        total: int,
        overlap_frames: int,
    ) -> np.ndarray:
        weights = np.ones(length, dtype=np.float32)
        if overlap_frames <= 0:
            return weights
        ramp = min(overlap_frames, length)
        if start > 0:
            for index in range(ramp):
                weights[index] *= float(index + 1) / float(ramp + 1)
        if end < total:
            for index in range(length - ramp, length):
                weights[index] *= float(length - index) / float(ramp + 1)
        return weights

    @staticmethod
    def _select_chunk_frames(
        total: int,
        requested_chunk_frames: int,
        target_width: int,
        target_height: int,
    ) -> int:
        if not torch.cuda.is_available():
            return max(1, min(total, requested_chunk_frames))

        try:
            free_bytes, _total_bytes = torch.cuda.mem_get_info()
        except Exception:
            return max(
                1,
                min(total, requested_chunk_frames, DEFAULT_VIDEOMAMA_FALLBACK_CHUNK_FRAMES),
            )

        pixel_scale = max(
            0.125,
            (target_width * target_height) / float(1024 * 576),
        )
        estimated_frame_bytes = max(
            1,
            int(VIDEOMAMA_FRAME_VRAM_BYTES_AT_1024 * pixel_scale),
        )
        usable_bytes = min(
            int(free_bytes * VIDEOMAMA_TARGET_VRAM_FRACTION),
            max(0, int(free_bytes) - VIDEOMAMA_RESERVED_VRAM_BYTES),
        )
        automatic_chunk_frames = max(1, usable_bytes // estimated_frame_bytes)
        return max(
            1,
            min(total, requested_chunk_frames, int(automatic_chunk_frames)),
        )

    @staticmethod
    def _clear_cuda_cache() -> None:
        gc.collect()
        if not torch.cuda.is_available():
            return
        try:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        except Exception:
            torch.cuda.empty_cache()

    def _ensure_loaded(self) -> None:
        if self._pipeline is not None:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("VideoMaMa postprocess requires CUDA.")
        if not self.repo_dir.is_dir():
            raise FileNotFoundError(f"VideoMaMa repository not found: {self.repo_dir}")
        if not self.base_model_dir.is_dir():
            raise FileNotFoundError(
                f"VideoMaMa base SVD model directory not found: {self.base_model_dir}"
            )
        if not (self.checkpoint_dir / "unet").is_dir():
            raise FileNotFoundError(
                f"VideoMaMa checkpoint directory not found: {self.checkpoint_dir}"
            )

        repo_path = str(self.repo_dir)
        if repo_path not in sys.path:
            sys.path.insert(0, repo_path)
        try:
            from pipeline_svd_mask import VideoInferencePipeline
        except ImportError as exc:
            raise RuntimeError("VideoMaMa postprocess requires diffusers.") from exc

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
            io.StringIO()
        ):
            pipeline = VideoInferencePipeline(
                base_model_path=str(self.base_model_dir),
                unet_checkpoint_path=str(self.checkpoint_dir),
                device="cuda",
                weight_dtype=torch.float16,
            )

        self._run_accepts_mask_cond_mode = (
            "mask_cond_mode" in inspect.signature(pipeline.run).parameters
        )
        self._pipeline = pipeline

    def _run_chunk(
        self,
        frames_bgr: list[np.ndarray],
        mask_frames: list[np.ndarray],
        options: MaskPostprocessOptions,
    ) -> list[np.ndarray]:
        overrides = [self._mask_override(mask_frame) for mask_frame in mask_frames]
        if all(mask is not None for mask in overrides):
            return [np.asarray(mask, dtype=np.uint8) for mask in overrides]

        height, width = mask_frames[0].shape[:2]
        target_width, target_height = self._compute_target_size(
            width,
            height,
            options.videomama_max_resolution,
        )
        cond_images: list[Image.Image] = []
        mask_images: list[Image.Image] = []
        for frame_bgr, mask_frame in zip(frames_bgr, mask_frames):
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frame_image = Image.fromarray(frame_rgb)
            if frame_image.size != (target_width, target_height):
                frame_image = frame_image.resize(
                    (target_width, target_height),
                    resample=PIL_LANCZOS,
                )
            cond_images.append(frame_image)

            mask_u8 = (np.asarray(mask_frame) > 127).astype(np.uint8) * 255
            mask_image = Image.fromarray(mask_u8, mode="L")
            if mask_image.size != (target_width, target_height):
                mask_image = mask_image.resize(
                    (target_width, target_height),
                    resample=PIL_LANCZOS,
                )
            mask_images.append(mask_image)

        with self._lock:
            self._ensure_loaded()
            assert self._pipeline is not None
            run_kwargs = {
                "cond_frames": cond_images,
                "mask_frames": mask_images,
                "seed": DEFAULT_VIDEOMAMA_SEED,
                "fps": DEFAULT_VIDEOMAMA_FPS,
                "motion_bucket_id": DEFAULT_VIDEOMAMA_MOTION_BUCKET_ID,
                "noise_aug_strength": DEFAULT_VIDEOMAMA_NOISE_AUG_STRENGTH,
            }
            if self._run_accepts_mask_cond_mode:
                run_kwargs["mask_cond_mode"] = "vae"
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ):
                generated_frames = self._pipeline.run(**run_kwargs)

        output: list[np.ndarray] = []
        for generated_frame, override in zip(generated_frames, overrides):
            if override is not None:
                output.append(np.asarray(override, dtype=np.uint8))
                continue
            alpha_image = generated_frame.convert("L")
            if alpha_image.size != (width, height):
                alpha_image = alpha_image.resize((width, height), resample=PIL_LANCZOS)
            output.append(np.asarray(alpha_image, dtype=np.uint8))

        self._clear_cuda_cache()
        return output

    @staticmethod
    def _mask_override(mask_frame: np.ndarray) -> np.ndarray | None:
        mask_u8 = np.asarray(mask_frame, dtype=np.uint8)
        if not np.any(mask_u8):
            return np.zeros_like(mask_u8, dtype=np.uint8)
        if np.all(mask_u8 > 127):
            return np.full_like(mask_u8, 255, dtype=np.uint8)
        return None

    @staticmethod
    def _compute_target_size(
        width: int,
        height: int,
        max_resolution: int,
    ) -> tuple[int, int]:
        if width <= 0 or height <= 0:
            raise ValueError("VideoMaMa input frames must have positive dimensions.")
        if width >= height:
            target_width = max_resolution
            target_height = int(height * max_resolution / width)
        else:
            target_height = max_resolution
            target_width = int(width * max_resolution / height)
        target_width = max(8, (target_width // 8) * 8)
        target_height = max(8, (target_height // 8) * 8)
        return target_width, target_height
