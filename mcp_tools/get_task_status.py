"""
MCP Tool: get_task_status
Query the processing progress and final results of an async image generation task.
"""

from .common import _get_task_snapshot
from .error_schema import make_validation_error, make_not_found_error


def handle_get_task_status(task_id: str) -> dict:
    """Query the processing progress and final results of an async image generation task."""
    if not task_id:
        return make_validation_error(
            "Parameter 'task_id' is required.",
            missing_fields=["task_id"],
        )

    task = _get_task_snapshot(task_id)
    if task is None:
        return make_not_found_error("task", task_id)

    return task
