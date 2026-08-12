from typing import Any, Dict
import gradio as gr

from core.pipelines.sd_image_pipeline import SdImagePipeline

sd_image_pipeline = SdImagePipeline()


def generate_image_wrapper(ui_inputs: dict, progress=gr.Progress(track_tqdm=True)):
    return sd_image_pipeline.run(ui_inputs=ui_inputs, progress=progress)
