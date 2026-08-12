import spaces
import importlib.util
import os
import sys
import site

from core.runtime_config import CONFIG

sage_mode = os.getenv("IMAGEGEN_USE_SAGE_ATTENTION", "auto").strip().lower()
sage_available = importlib.util.find_spec("sageattention") is not None
use_sage_attention = sage_mode in {"1", "true", "yes", "on"} or (
    sage_mode == "auto" and sage_available
)
if use_sage_attention and "--use-sage-attention" not in sys.argv:
    sys.argv.append("--use-sage-attention")
    print("🚀 [SageAttention] Injected '--use-sage-attention' into sys.argv.")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)
    print(f"✅ Added project root '{APP_DIR}' to sys.path.")

# Keep ComfyUI code isolated while pointing its model/input/output directories
# at this Space. These arguments are consumed when ComfyUI is imported.
if "--base-directory" not in sys.argv:
    sys.argv.extend(["--base-directory", APP_DIR])

SAGE_PATCH_APPLIED = False

def apply_sage_attention_patch():
    global SAGE_PATCH_APPLIED
    if SAGE_PATCH_APPLIED:
        return "SageAttention patch already applied."
    if not use_sage_attention:
        return "SageAttention disabled or unavailable; using the default attention backend."

    try:
        from comfy import model_management
        import sageattention

        print("--- [Runtime Patch] sageattention package found. Applying patch... ---")
        model_management.sage_attention_enabled = lambda: True
        model_management.pytorch_attention_enabled = lambda: False

        SAGE_PATCH_APPLIED = True
        return "✅ Successfully enabled SageAttention."
    except ImportError:
        SAGE_PATCH_APPLIED = False
        msg = "--- [Runtime Patch] ⚠️ sageattention package not found. Continuing with default attention. ---"
        print(msg)
        return msg
    except Exception as e:
        SAGE_PATCH_APPLIED = False
        msg = f"--- [Runtime Patch] ❌ An error occurred while applying SageAttention patch: {e} ---"
        print(msg)
        return msg

@spaces.GPU
def dummy_gpu_for_startup():
    print("--- [GPU Startup] Dummy function for startup check initiated. ---")
    patch_result = apply_sage_attention_patch()
    print(f"--- [GPU Startup] {patch_result} ---")
    print("--- [GPU Startup] Startup check passed. ---")
    return "Startup check passed."


def main():
    os.chdir(APP_DIR)
    from comfy_integration import setup as setup_comfyui
    from imagegen_utils.app_utils import load_ipadapter_presets

    print("--- [Setup] Starting ComfyUI initialization ---")
    setup_comfyui.initialize_comfyui()

    print("--- [Setup] Applying SageAttention Runtime Patch ---")
    patch_result = apply_sage_attention_patch()
    print(f"--- [Setup] {patch_result} ---")

    print("--- [Setup] Reloading site-packages to detect newly installed packages... ---")
    try:
        site.main()
        print("--- [Setup] ✅ Site-packages reloaded. ---")
    except Exception as e:
        print(f"--- [Setup] ⚠️  Warning: Could not fully reload site-packages: {e} ---")

    if CONFIG.enable_startup_gpu_probe:
        print("--- Initiating optional GPU startup check ---")
        try:
            dummy_gpu_for_startup()
        except BaseException as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            print(f"--- [GPU Startup] ⚠️ Warning: Startup check failed: {err_msg} ---")

    print("--- Starting Application Setup ---")

    print("--- Loading IPAdapter presets ---")
    load_ipadapter_presets()
    print("--- ✅ IPAdapter setup complete. ---")


    print("--- Environment configured. Proceeding with module imports. ---")
    from ui.layout import build_ui
    from ui.events import attach_event_handlers
    if CONFIG.enable_mcp:
        import mcp_tools as mcp
        print(f"✅ Loaded MCP module with tools: {[fn.__name__ for fn in mcp.MCP_FUNCTIONS]}")
    else:
        print("ℹ️ MCP is disabled by IMAGEGEN_ENABLE_MCP.")

    print(f"✅ Working directory is stable: {os.getcwd()}")

    demo = build_ui(attach_event_handlers)

    print(
        "--- Launching Gradio Interface "
        f"(GPU concurrency={CONFIG.gpu_concurrency}, queue={CONFIG.queue_max_size}, "
        f"MCP={CONFIG.enable_mcp}) ---"
    )
    demo.queue(
        default_concurrency_limit=1,
        max_size=CONFIG.queue_max_size,
        status_update_rate="auto",
    ).launch(mcp_server=CONFIG.enable_mcp)


if __name__ == "__main__":
    main()
