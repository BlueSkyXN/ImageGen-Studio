def __getattr__(name):
    if name in ("types", "server", "client", "shared"):
        raise ImportError(f"No module named 'mcp.{name}' in local mcp package")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

from .get_task_list import handle_get_task_list
from .get_model_architecture_list import handle_get_model_architecture_list
from .get_model_list import handle_get_model_list
from .get_feature_list import handle_get_feature_list
from .get_model_features import handle_get_model_features
from .run import handle_run
from .get_task_status import handle_get_task_status
from .get_chain_schema import handle_get_chain_schema
from .error_schema import make_error, make_validation_error, make_not_found_error
from .mcp_gradio_integration import (
    register_high_level_mcp_apis,
    cleanup_dependencies_api_names,
    patch_gradio_api_suppression,
    HIGH_LEVEL_MCP_API_NAMES,
)

MCP_FUNCTIONS = [
    handle_get_task_list,
    handle_get_model_architecture_list,
    handle_get_model_list,
    handle_get_feature_list,
    handle_get_model_features,
    handle_run,
    handle_get_task_status,
    handle_get_chain_schema,
]

__all__ = [
    "handle_get_task_list",
    "handle_get_model_architecture_list",
    "handle_get_model_list",
    "handle_get_feature_list",
    "handle_get_model_features",
    "handle_run",
    "handle_get_task_status",
    "handle_get_chain_schema",
    "make_error",
    "make_validation_error",
    "make_not_found_error",
    "register_high_level_mcp_apis",
    "cleanup_dependencies_api_names",
    "patch_gradio_api_suppression",
    "HIGH_LEVEL_MCP_API_NAMES",
    "MCP_FUNCTIONS",
]
