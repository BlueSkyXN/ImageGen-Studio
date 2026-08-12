from __future__ import annotations

import threading
import time
import unittest
from pathlib import Path

import yaml

from core.runtime_config import CONFIG, estimate_gpu_duration
from core.settings import CHECKPOINT_DIR, INPUT_DIR, OUTPUT_DIR
from core.task_scheduler import (
    QueueFullError,
    TaskCancelledError,
    generation_guard,
    generation_slot,
    submit_background,
)


ROOT = Path(__file__).resolve().parents[1]


class RuntimeConfigTests(unittest.TestCase):
    def test_duration_estimation_is_bounded(self):
        self.assertEqual(estimate_gpu_duration({"zero_gpu_duration": 999}), 120)
        self.assertEqual(estimate_gpu_duration({"zero_gpu_duration": 1}), 30)
        self.assertEqual(
            estimate_gpu_duration(
                {
                    "model_display_name": "example-lightning",
                    "num_inference_steps": 4,
                    "batch_size": 1,
                    "width": 1024,
                    "height": 1024,
                }
            ),
            45,
        )
        self.assertEqual(
            estimate_gpu_duration(
                {
                    "model_display_name": "large-model",
                    "num_inference_steps": 40,
                    "batch_size": 3,
                    "width": 2048,
                    "height": 2048,
                }
            ),
            120,
        )

    def test_generation_slots_respect_configured_limit(self):
        active = 0
        maximum = 0
        state_lock = threading.Lock()

        def worker():
            nonlocal active, maximum
            with generation_slot():
                with state_lock:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.02)
                with state_lock:
                    active -= 1

        threads = [threading.Thread(target=worker) for _ in range(CONFIG.gpu_concurrency + 2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertLessEqual(maximum, CONFIG.gpu_concurrency)

    def test_cancelled_job_stops_before_guarded_execution(self):
        reached_function = False

        @generation_guard
        def guarded(ui_inputs):
            nonlocal reached_function
            reached_function = True

        cancel_event = threading.Event()
        cancel_event.set()
        with self.assertRaises(TaskCancelledError):
            guarded({"_cancel_event": cancel_event})
        self.assertFalse(reached_function)

    def test_waiting_cancelled_job_leaves_gate_before_gpu_is_free(self):
        holder_started = threading.Event()
        release_holder = threading.Event()
        cancellation = threading.Event()
        errors = []

        def holder():
            with generation_slot():
                holder_started.set()
                release_holder.wait(2)

        @generation_guard
        def waiting_job(ui_inputs):
            raise AssertionError("cancelled job must not execute")

        def wait_then_cancel():
            try:
                waiting_job({"_cancel_event": cancellation})
            except BaseException as exc:
                errors.append(exc)

        holder_thread = threading.Thread(target=holder)
        holder_thread.start()
        self.assertTrue(holder_started.wait(1))

        waiting_thread = threading.Thread(target=wait_then_cancel)
        waiting_thread.start()
        cancellation.set()
        waiting_thread.join(1)

        self.assertFalse(waiting_thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], TaskCancelledError)

        release_holder.set()
        holder_thread.join(1)
        self.assertFalse(holder_thread.is_alive())

    def test_mcp_pending_queue_is_bounded(self):
        release = threading.Event()
        futures = [
            submit_background(lambda: release.wait(2))
            for _ in range(CONFIG.mcp_max_pending)
        ]
        with self.assertRaises(QueueFullError):
            submit_background(lambda: None)
        release.set()
        for future in futures:
            self.assertTrue(future.result(timeout=3))


class RegistryTests(unittest.TestCase):
    def test_runtime_directories_are_project_absolute(self):
        for configured in (CHECKPOINT_DIR, INPUT_DIR, OUTPUT_DIR):
            path = Path(configured)
            self.assertTrue(path.is_absolute())
            self.assertTrue(path.is_relative_to(ROOT))

    def test_quick_presets_exist(self):
        registry = yaml.safe_load((ROOT / "yaml" / "model_list.yaml").read_text("utf-8"))
        names = {
            model["display_name"]
            for architecture in registry["Checkpoint"].values()
            for model in architecture.get("models", [])
        }
        expected = {
            "Krea-2-Turbo",
            "lightx2v/Qwen-Image-2512-Lightning",
            "circlestone-labs/Anima-Turbo-v1.0",
            "lightx2v/Qwen-Image-Edit-2511-Lightning",
            "CagliostroLab/Animagine XL 4.0",
        }
        self.assertTrue(expected.issubset(names))

    def test_vendor_revisions_are_full_commits(self):
        lock = yaml.safe_load((ROOT / "vendor.lock.yaml").read_text("utf-8"))
        entries = [lock["comfyui"], *lock["custom_nodes"].values()]
        for entry in entries:
            revision = entry["revision"]
            self.assertEqual(len(revision), 40)
            int(revision, 16)

    def test_concurrency_regressions_are_absent(self):
        mcp_run = (ROOT / "mcp_tools" / "run.py").read_text("utf-8")
        input_processor = (
            ROOT / "core" / "pipelines" / "pipeline_input_processor.py"
        ).read_text("utf-8")
        studio = (ROOT / "ui" / "shared" / "studio_ui.py").read_text("utf-8")
        requirements = (ROOT / "requirements.txt").read_text("utf-8")
        self.assertNotIn("threading.Thread", mcp_run)
        self.assertIn("uuid.uuid4().hex", input_processor)
        self.assertIn('"_task_prefixes": [(prefix, None)]', studio)
        self.assertIn("onnxruntime-gpu==", requirements)


if __name__ == "__main__":
    unittest.main()
