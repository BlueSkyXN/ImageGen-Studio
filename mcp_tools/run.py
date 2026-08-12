"""
MCP Tool: run
Unified image generation task submission and execution interface.
"""

import time
import uuid
from core.task_scheduler import QueueFullError, submit_background
from core.runtime_config import CONFIG
from .common import (
    _load_yaml,
    _MODEL_LIST_PATH,
    _TASK_DEFINITIONS,
    _TASKS_DB,
    _TASKS_LOCK,
    _get_task_snapshot,
    _execute_imagegen_pipeline,
)
from .error_schema import make_error, make_validation_error, make_not_found_error


def handle_run(params: dict) -> dict:
    """Unified image generation task execution interface."""
    if not isinstance(params, dict):
        return make_validation_error("Request params must be an object.")

    missing = []
    for req_field in ["task_type", "model", "prompt"]:
        if req_field not in params or not params[req_field]:
            missing.append(req_field)
    if missing:
        return make_validation_error(
            f"Missing required parameter(s): {', '.join(missing)}",
            missing_fields=missing,
        )

    task_type = params["task_type"]
    valid_tasks = [t["task_type"] for t in _TASK_DEFINITIONS]
    if task_type not in valid_tasks:
        return make_validation_error(
            f"Invalid task_type '{task_type}'. Must be one of {valid_tasks}.",
            invalid_fields={"task_type": f"Must be in {valid_tasks}"},
        )

    model_list = _load_yaml(_MODEL_LIST_PATH)
    checkpoints = model_list.get("Checkpoint", {})
    all_models = set()
    for arch_name, arch_data in checkpoints.items():
        if isinstance(arch_data, dict):
            for m in arch_data.get("models", []):
                all_models.add(m.get("display_name"))

    if params["model"] not in all_models:
        return make_not_found_error("model", params["model"])

    task_requirements = {
        "txt2img": ("width", "height"),
        "img2img": ("image",),
        "inpaint": ("image",),
        "outpaint": ("image", "pad_left", "pad_right", "pad_top", "pad_bottom"),
        "hires_fix": ("image", "upscale_by"),
    }
    missing_task_fields = [
        field
        for field in task_requirements.get(task_type, ())
        if field not in params or params[field] is None or params[field] == ""
    ]
    if missing_task_fields:
        return make_validation_error(
            f"Missing required parameter(s) for {task_type}: {', '.join(missing_task_fields)}",
            missing_fields=missing_task_fields,
        )

    try:
        batch_size = int(params.get("batch_size", 1))
    except (TypeError, ValueError):
        batch_size = 0
    if not 1 <= batch_size <= CONFIG.max_batch_size:
        return make_validation_error(
            f"batch_size must be between 1 and {CONFIG.max_batch_size}.",
            invalid_fields={"batch_size": f"Expected 1..{CONFIG.max_batch_size}"},
        )

    with _TASKS_LOCK:
        if len(_TASKS_DB) >= CONFIG.mcp_task_retention:
            finished = sorted(
                (
                    (old_task_id, task)
                    for old_task_id, task in _TASKS_DB.items()
                    if task.get("status") in {"completed", "failed"}
                ),
                key=lambda pair: pair[1].get("completed_at", pair[1].get("failed_at", 0)),
            )
            remove_count = max(1, len(_TASKS_DB) - CONFIG.mcp_task_retention + 1)
            for old_task_id, _ in finished[:remove_count]:
                _TASKS_DB.pop(old_task_id, None)
            if len(_TASKS_DB) >= CONFIG.mcp_task_retention:
                return make_error(
                    "QUEUE_FULL",
                    "任务记录已满且当前任务均未结束，请稍后再试。",
                )

        task_id = f"img_task_{uuid.uuid4().hex[:10]}"
        created_at = int(time.time())
        _TASKS_DB[task_id] = {
            "task_id": task_id,
            "status": "queued",
            "progress": 0,
            "created_at": created_at,
        }

    async_exec = params.get("async_execution", False)

    if async_exec:
        try:
            submit_background(_execute_imagegen_pipeline, task_id, params)
        except QueueFullError as exc:
            with _TASKS_LOCK:
                _TASKS_DB.pop(task_id, None)
            return make_error("QUEUE_FULL", str(exc))
        return {
            "status": "queued",
            "task_id": task_id,
            "poll_interval_ms": 2000,
            "message": "Task queued successfully. Poll get_task_status for results.",
        }
    else:
        _execute_imagegen_pipeline(task_id, params)
        return _get_task_snapshot(task_id)
