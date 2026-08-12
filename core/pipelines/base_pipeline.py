from abc import ABC, abstractmethod
from typing import List, Any, Dict
import gradio as gr
import spaces
import tempfile
import imageio
import numpy as np
import sys
import os

class BasePipeline(ABC):
    def __init__(self):
        from core.model_manager import model_manager
        self.model_manager = model_manager

    @abstractmethod
    def get_required_models(self, **kwargs) -> List[str]:
        pass

    @abstractmethod
    def run(self, *args, progress: gr.Progress, **kwargs) -> Any:
        pass

    def _ensure_models_downloaded(self, progress: gr.Progress, **kwargs):
        """Ensures model files are downloaded before requesting GPU."""
        required_models = self.get_required_models(**kwargs)
        self.model_manager.ensure_models_downloaded(required_models, progress=progress)

    def _execute_gpu_logic(self, gpu_function: callable, duration: int, default_duration: int, task_name: str, *args, **kwargs):
        final_duration = default_duration
        try:
            if duration is not None and int(duration) > 0:
                final_duration = int(duration)
        except (ValueError, TypeError):
            print(f"Invalid ZeroGPU duration input for {task_name}. Using default {default_duration}s.")
            pass

        print(f"Requesting ZeroGPU for {task_name} with duration: {final_duration} seconds.")
        gpu_runner = spaces.GPU(duration=final_duration)(gpu_function)

        return gpu_runner(*args, **kwargs)

    def _encode_video_from_frames(self, frames_tensor_cpu: 'torch.Tensor', fps: int, progress: gr.Progress) -> str:
        progress(0.9, desc="Encoding video on CPU...")
        frames_np = (frames_tensor_cpu.numpy() * 255.0).astype(np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video_file:
            video_path = temp_video_file.name
            writer = imageio.get_writer(video_path, fps=fps, codec='libx264', quality=8)
            for frame in frames_np:
                writer.append_data(frame)
            writer.close()

        progress(1.0, desc="Done!")
        return video_path
