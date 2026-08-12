"""
MCP Tool Handlers — Backward-compatible aggregation entry point.
Core logic has been split into individual files (get_*.py and run.py).
"""

from .get_task_list import handle_get_task_list
from .get_model_architecture_list import handle_get_model_architecture_list
from .get_model_list import handle_get_model_list
from .get_feature_list import handle_get_feature_list
from .get_model_features import handle_get_model_features
from .run import handle_run
from .get_task_status import handle_get_task_status
from .get_chain_schema import handle_get_chain_schema
from .common import (
    _TASK_DEFINITIONS,
    _TASKS_DB,
    _load_yaml,
    _execute_imagegen_pipeline,
)

__all__ = [
    "handle_get_task_list",
    "handle_get_model_architecture_list",
    "handle_get_model_list",
    "handle_get_feature_list",
    "handle_get_model_features",
    "handle_run",
    "handle_get_task_status",
    "handle_get_chain_schema",
]
