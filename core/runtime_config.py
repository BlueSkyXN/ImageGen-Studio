"""Runtime configuration shared by the web UI, MCP API, and engine."""

from __future__ import annotations

import os
from dataclasses import dataclass


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class RuntimeConfig:
    """Bounded settings with safe defaults for one shared GPU process."""

    # ComfyUI nodes, caches, and model state are process-global. A single
    # process is therefore always serialized; scale with separate replicas.
    gpu_concurrency: int = _env_int("IMAGEGEN_GPU_CONCURRENCY", 1, 1, 1)
    queue_max_size: int = _env_int("IMAGEGEN_QUEUE_MAX_SIZE", 24, 1, 256)
    mcp_max_pending: int = _env_int("IMAGEGEN_MCP_MAX_PENDING", 16, 1, 256)
    mcp_task_retention: int = _env_int("IMAGEGEN_MCP_TASK_RETENTION", 200, 20, 5000)
    max_batch_size: int = _env_int("IMAGEGEN_MAX_BATCH_SIZE", 4, 1, 16)
    # ZeroGPU is best used as one bounded job at a time.  The environment
    # overrides are intentionally available for dedicated hardware, while the
    # defaults keep the public Space predictable and affordable.
    max_pk_models: int = _env_int("IMAGEGEN_MAX_PK_MODELS", 2, 2, 4)
    max_multi_images: int = _env_int("IMAGEGEN_MAX_MULTI_IMAGES", 4, 1, 10)
    max_plan_jobs: int = _env_int("IMAGEGEN_MAX_PLAN_JOBS", 4, 2, 12)
    max_plan_outputs: int = _env_int("IMAGEGEN_MAX_PLAN_OUTPUTS", 8, 2, 16)
    output_retention: int = _env_int("IMAGEGEN_OUTPUT_RETENTION", 80, 10, 1000)
    max_input_megapixels: float = _env_float(
        "IMAGEGEN_MAX_INPUT_MEGAPIXELS", 4.2, 0.25, 64.0
    )
    max_reference_megapixels: float = _env_float(
        "IMAGEGEN_MAX_REFERENCE_MEGAPIXELS", 12.0, 1.0, 64.0
    )
    max_reference_images: int = _env_int(
        "IMAGEGEN_MAX_REFERENCE_IMAGES", 10, 1, 20
    )
    max_output_megapixels: float = _env_float(
        "IMAGEGEN_MAX_OUTPUT_MEGAPIXELS", 16.0, 1.0, 128.0
    )
    min_free_disk_gb: float = _env_float(
        "IMAGEGEN_MIN_FREE_DISK_GB", 3.0, 0.5, 50.0
    )
    enable_mcp: bool = _env_bool("IMAGEGEN_ENABLE_MCP", True)
    enable_startup_gpu_probe: bool = _env_bool("IMAGEGEN_STARTUP_GPU_PROBE", False)
    default_language: str = os.getenv("IMAGEGEN_DEFAULT_LANGUAGE", "zh-CN")


CONFIG = RuntimeConfig()


def estimate_gpu_duration(inputs: dict) -> int:
    """Estimate a ZeroGPU reservation without asking ordinary users to guess.

    This is deliberately conservative. A caller can still pass 60/90/120 via
    ``zero_gpu_duration``; 0 or an empty value means automatic mode.
    """

    explicit = inputs.get("zero_gpu_duration")
    try:
        explicit_value = int(explicit or 0)
    except (TypeError, ValueError):
        explicit_value = 0
    if explicit_value > 0:
        return max(30, min(120, explicit_value))

    model = str(inputs.get("model_display_name", "")).lower()
    steps = max(1, int(inputs.get("num_inference_steps") or 20))
    batch = max(1, int(inputs.get("batch_size") or 1))
    width = max(256, int(inputs.get("width") or 1024))
    height = max(256, int(inputs.get("height") or 1024))
    megapixels = (width * height) / 1_000_000

    fast_tokens = ("turbo", "lightning", "fast", "distilled", "schnell", "4b")
    base = 35 if any(token in model for token in fast_tokens) else 60
    work = steps * batch * max(0.5, megapixels)

    if work >= 55 or batch >= 3 or megapixels >= 2.5:
        return 120
    if work >= 28:
        return 90
    return max(45, base)
