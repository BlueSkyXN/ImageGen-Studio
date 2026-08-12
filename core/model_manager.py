import gc
from typing import List
import gradio as gr
from imagegen_utils.app_utils import _ensure_model_downloaded
from core.settings import ALL_MODEL_MAP

class ModelManager:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelManager, cls).__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self):
        if hasattr(self, 'initialized'):
            return
        self.initialized = True
        print("✅ ModelManager initialized.")

    def ensure_models_downloaded(self, required_models: List[str], progress):
        print(f"--- [ModelManager] Ensuring models are downloaded: {required_models} ---")
        for i, display_name in enumerate(required_models):
            if progress and hasattr(progress, '__call__'):
                progress(i / max(len(required_models), 1), desc=f"Checking file: {display_name}")
            try:
                _ensure_model_downloaded(display_name, progress)
            except Exception as e:
                raise gr.Error(f"模型“{display_name}”下载失败：{e}")
        print(f"--- [ModelManager] ✅ All required models are present on disk. ---")

model_manager = ModelManager()


def release_loaded_models() -> bool:
    """Best-effort release of ComfyUI model state while GPU access is active."""

    released = False
    try:
        from comfy import model_management

        unload = getattr(model_management, "unload_all_models", None)
        if callable(unload):
            unload()
            released = True

        cleanup = getattr(model_management, "cleanup_models", None)
        if callable(cleanup):
            cleanup()

        gc.collect()
        empty_cache = getattr(model_management, "soft_empty_cache", None)
        if callable(empty_cache):
            try:
                empty_cache(force=True)
            except TypeError:
                empty_cache()
        print("✅ Released ComfyUI model state after a model switch/error.")
    except Exception as exc:
        # Cleanup must never hide the original generation result or exception.
        gc.collect()
        print(f"Warning: Could not fully release ComfyUI model state: {exc}")
    return released
