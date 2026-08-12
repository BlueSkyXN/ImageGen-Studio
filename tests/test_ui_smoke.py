from __future__ import annotations

import os
import sys
import types
import unittest
from unittest import mock

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

try:
    import gradio
except ImportError:
    gradio = None


@unittest.skipIf(gradio is None, "Gradio is not installed")
class UnifiedUiSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch_stub = types.ModuleType("torch")
        torch_stub.Tensor = object
        torch_stub.cuda = types.SimpleNamespace(
            empty_cache=lambda: None, is_available=lambda: False
        )
        sys.modules.setdefault("torch", torch_stub)

        nodes_stub = types.ModuleType("comfy_integration.nodes")
        nodes_stub.SAMPLER_CHOICES = ["euler", "dpmpp_2m_sde_gpu"]
        nodes_stub.SCHEDULER_CHOICES = ["simple", "normal", "karras"]
        sys.modules["comfy_integration.nodes"] = nodes_stub

        generation_stub = types.ModuleType("core.generation_logic")
        generation_stub.generate_image_wrapper = lambda inputs, progress=None: []
        sys.modules["core.generation_logic"] = generation_stub

    def test_builds_one_workspace_and_only_high_level_public_apis(self):
        from ui.events import attach_event_handlers
        from ui.layout import build_ui

        app = build_ui(attach_event_handlers)
        config = app.get_config_file()
        self.assertLess(len(config.get("components", [])), 1000)
        self.assertLess(len(config.get("dependencies", [])), 150)
        public_names = {
            dependency.get("api_name")
            for dependency in config.get("dependencies", [])
            if dependency.get("show_api") and dependency.get("api_name")
        }
        self.assertEqual(
            public_names,
            {
                "get_task_list",
                "get_model_architecture_list",
                "get_model_list",
                "get_feature_list",
                "get_model_features",
                "run",
                "get_task_status",
                "run_imagegen",
                "get_chain_schema",
            },
        )

    def test_reference_budget_rejects_oversized_advanced_image(self):
        from PIL import Image

        from core.pipelines.pipeline_input_processor import (
            _validate_reference_image_budget,
        )

        with self.assertRaises(gradio.Error):
            _validate_reference_image_budget(
                {"reference_latent_data": [Image.new("1", (3000, 2000))]}
            )

    def test_model_download_preflight_preserves_disk_reserve(self):
        from imagegen_utils.app_utils import _assert_download_space

        usage = types.SimpleNamespace(total=10, used=9, free=1)
        with (
            mock.patch("shutil.disk_usage", return_value=usage),
            self.assertRaises(gradio.Error),
        ):
            _assert_download_space(100, "model.safetensors", "/tmp")


if __name__ == "__main__":
    unittest.main()
