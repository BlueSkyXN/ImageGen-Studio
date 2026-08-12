from __future__ import annotations

import threading
import unittest

from PIL import Image

from core.execution_plan import (
    MODE_MODEL_PK,
    MODE_MULTI_INDEPENDENT,
    MODE_MULTI_MODEL_GRID,
    MODE_MULTI_REFERENCE,
    ExecutionPlanError,
    PlannedGeneration,
    build_execution_plan,
    execute_generation_plan,
)


def base_inputs(**overrides):
    values = {
        "task_type": "txt2img",
        "model_display_name": "Krea-2-Turbo",
        "positive_prompt": "一只猫",
        "negative_prompt": "",
        "seed": -1,
        "batch_size": 1,
        "num_inference_steps": 13,
        "guidance_scale": 2.5,
        "sampler": "euler",
        "scheduler": "simple",
        "width": 1024,
        "height": 1024,
    }
    values.update(overrides)
    return values


class ExecutionPlanTests(unittest.TestCase):
    def test_pk_resolves_one_seed_and_marks_only_the_model_boundary(self):
        plan = build_execution_plan(
            base_inputs(lora_data=["Civitai", "123", 1.0, None]),
            mode=MODE_MODEL_PK,
            extra_models=["Krea-2-Raw"],
            use_model_defaults=True,
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual(plan[0].inputs["seed"], plan[1].inputs["seed"])
        self.assertGreaterEqual(plan[0].inputs["seed"], 0)
        self.assertEqual(plan[0].inputs["num_inference_steps"], 8)
        self.assertEqual(plan[1].inputs["num_inference_steps"], 52)
        self.assertTrue(plan[0].inputs["_release_models_after_run"])
        self.assertNotIn("_release_models_after_run", plan[1].inputs)
        self.assertEqual(plan[0].inputs["lora_data"], [])
        self.assertEqual(plan[1].inputs["lora_data"], [])

    def test_independent_images_keep_user_sampling_values(self):
        images = [Image.new("RGB", (32, 32)), Image.new("RGB", (48, 32))]
        plan = build_execution_plan(
            base_inputs(task_type="img2img"),
            mode=MODE_MULTI_INDEPENDENT,
            images=images,
            use_model_defaults=True,
        )
        self.assertEqual(len(plan), 2)
        self.assertEqual([item.inputs["img2img_image"] for item in plan], images)
        self.assertTrue(all(item.inputs["num_inference_steps"] == 13 for item in plan))

    def test_image_model_grid_groups_inputs_by_model(self):
        images = [Image.new("RGB", (32, 32)), Image.new("RGB", (48, 32))]
        plan = build_execution_plan(
            base_inputs(task_type="img2img"),
            mode=MODE_MULTI_MODEL_GRID,
            extra_models=["Krea-2-Raw"],
            images=images,
        )
        self.assertEqual(
            [item.inputs["model_display_name"] for item in plan],
            ["Krea-2-Turbo", "Krea-2-Turbo", "Krea-2-Raw", "Krea-2-Raw"],
        )
        self.assertNotIn("_release_models_after_run", plan[0].inputs)
        self.assertTrue(plan[1].inputs["_release_models_after_run"])

    def test_reference_fusion_uses_edit_checkpoint_chain(self):
        images = [Image.new("RGB", (32, 32)), Image.new("RGB", (32, 32))]
        edit_model = "lightx2v/Qwen-Image-Edit-2511-Lightning"
        plan = build_execution_plan(
            base_inputs(model_display_name=edit_model),
            mode=MODE_MULTI_REFERENCE,
            images=images,
        )
        self.assertEqual(plan[0].inputs["qwen_image_edit_data"], images)

        with self.assertRaisesRegex(ExecutionPlanError, "编辑/多模态模型"):
            build_execution_plan(
                base_inputs(
                    model_display_name="lightx2v/Qwen-Image-2512-Lightning"
                ),
                mode=MODE_MULTI_REFERENCE,
                images=images,
            )

    def test_reference_fusion_is_not_img2img(self):
        with self.assertRaisesRegex(ExecutionPlanError, "文生图"):
            build_execution_plan(
                base_inputs(task_type="img2img"),
                mode=MODE_MULTI_REFERENCE,
                images=[Image.new("RGB", (32, 32))],
            )

    def test_executor_keeps_partial_results_and_failure_context(self):
        plan = [
            PlannedGeneration({"id": 1}, "模型一"),
            PlannedGeneration({"id": 2}, "模型二"),
        ]

        def generate(inputs, _progress):
            if inputs["id"] == 2:
                raise RuntimeError("显存不足")
            return ["one.png"]

        gallery, summary = execute_generation_plan(plan, generate)
        self.assertEqual(gallery, [("one.png", "模型一")])
        self.assertIn("模型二：显存不足", summary)

    def test_executor_stops_before_next_case_after_cancel(self):
        cancellation = threading.Event()
        cancellation.set()
        with self.assertRaisesRegex(ExecutionPlanError, "已取消"):
            execute_generation_plan(
                [PlannedGeneration({}, "任务")],
                lambda *_: ["unexpected.png"],
                cancel_event=cancellation,
            )

    def test_cancel_after_success_keeps_partial_gallery(self):
        cancellation = threading.Event()
        called = []

        def generate(inputs, _progress):
            called.append(inputs["id"])
            cancellation.set()
            return ["kept.png"]

        gallery, summary = execute_generation_plan(
            [
                PlannedGeneration({"id": 1}, "任务一"),
                PlannedGeneration({"id": 2}, "任务二"),
            ],
            generate,
            cancel_event=cancellation,
        )
        self.assertEqual(called, [1])
        self.assertEqual(gallery, [("kept.png", "任务一")])
        self.assertIn("已保留成功结果", summary)


if __name__ == "__main__":
    unittest.main()
