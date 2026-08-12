from __future__ import annotations

import unittest

from PIL import Image

from mcp_tools.common import _parse_image_param, _validate_public_image_url
from mcp_tools.get_chain_schema import handle_get_chain_schema
from mcp_tools.get_feature_list import handle_get_feature_list
from mcp_tools.get_model_features import handle_get_model_features
from mcp_tools.run import handle_run


class McpValidationTests(unittest.TestCase):
    def test_requires_common_fields(self):
        result = handle_run({})
        self.assertEqual(result["error"]["code"], "INVALID_PARAMS")

    def test_requires_task_specific_fields_before_execution(self):
        result = handle_run(
            {
                "task_type": "txt2img",
                "model": "Krea-2-Turbo",
                "prompt": "测试",
            }
        )
        self.assertEqual(result["error"]["code"], "INVALID_PARAMS")
        self.assertEqual(set(result["error"]["details"]["missing_fields"]), {"width", "height"})

    def test_batch_limit_is_enforced(self):
        result = handle_run(
            {
                "task_type": "txt2img",
                "model": "Krea-2-Turbo",
                "prompt": "测试",
                "width": 1024,
                "height": 1024,
                "batch_size": 999,
            }
        )
        self.assertEqual(result["error"]["code"], "INVALID_PARAMS")

    def test_private_image_urls_are_rejected(self):
        with self.assertRaises(ValueError):
            _validate_public_image_url("http://127.0.0.1/private.png")

    def test_decoded_image_pixel_budget_is_enforced(self):
        with self.assertRaisesRegex(ValueError, "maximum"):
            _parse_image_param(Image.new("RGB", (3000, 2000)))

    def test_fluxus_chain_schema_contract(self):
        missing = handle_get_chain_schema("")
        self.assertEqual(missing["error"]["code"], "INVALID_PARAMS")
        self.assertEqual(
            missing["error"]["details"]["missing_fields"], ["chain_type"]
        )
        self.assertEqual(handle_get_chain_schema("lora")["feature_name"], "lora")
        self.assertEqual(
            handle_get_chain_schema("style")["canonical_feature_name"], "flux1_style"
        )

    def test_fluxus_feature_discovery_can_return_full_schemas(self):
        full = handle_get_feature_list("", include_schema_on_empty=True)
        compact = handle_get_feature_list("")
        self.assertTrue(full)
        self.assertTrue(all("parameters_schema" in item for item in full))
        self.assertTrue(all("parameters_schema" not in item for item in compact))

    def test_checkpoint_level_edit_capability_is_not_overreported(self):
        regular = handle_get_model_features(
            "lightx2v/Qwen-Image-2512-Lightning"
        )
        editing = handle_get_model_features(
            "lightx2v/Qwen-Image-Edit-2511-Lightning"
        )
        self.assertNotIn("qwen_image_edit", regular["supported_features"])
        self.assertIn("qwen_image_edit", editing["supported_features"])


if __name__ == "__main__":
    unittest.main()
