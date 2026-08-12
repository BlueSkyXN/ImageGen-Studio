import os
import random
import shutil
import torch
import uuid
import gradio as gr
from PIL import Image
from typing import List, Dict, Any

from .base_pipeline import BasePipeline
from core.settings import *
from core.model_capabilities import supports_chain_for_model
from imagegen_utils.app_utils import sanitize_prompt
from core.workflow_assembler import WorkflowAssembler
from core.runtime_config import CONFIG, estimate_gpu_duration
from core.task_scheduler import TaskCancelledError, generation_guard
from .workflow_executor import WorkflowExecutor
from .pipeline_input_processor import process_pipeline_inputs

class SdImagePipeline(BasePipeline):
    CHAIN_INPUT_KEYS = {
        "lora": ("lora_data",),
        "controlnet": ("controlnet_data",),
        "anima_controlnet_lllite": ("anima_controlnet_lllite_data",),
        "diffsynth_controlnet": ("diffsynth_controlnet_data",),
        "krea2_controlnet": ("krea2_controlnet_data",),
        "ipadapter": ("ipadapter_data",),
        "flux1_ipadapter": ("flux1_ipadapter_data",),
        "sd3_ipadapter": ("sd3_ipadapter_chain",),
        "style": ("style_data",),
        "embedding": ("embedding_data",),
        "conditioning": ("conditioning_data",),
        "reference_latent": ("reference_latent_data",),
        "hidream_o1_reference": ("hidream_o1_reference_data",),
        "joyai_image": ("joyai_reference_data",),
        "krea2_identity_edit": ("krea2_identity_edit_data",),
        "krea2_style_reference": ("krea2_reference_edit_data",),
        "qwen_image_edit": ("qwen_image_edit_data",),
        "boogu_image_edit": ("boogu_edit_data",),
        "reference_image": ("reference_image_data",),
    }
    NATIVE_REFERENCE_CHAINS = {
        "reference_latent",
        "hidream_o1_reference",
        "joyai_image",
        "krea2_identity_edit",
        "krea2_style_reference",
        "qwen_image_edit",
        "boogu_image_edit",
        "reference_image",
    }

    def get_required_models(self, model_display_name: str, **kwargs) -> List[str]:
        model_info = ALL_MODEL_MAP.get(model_display_name)
        if not model_info:
            return [model_display_name]

        path_or_components = model_info[1]
        if isinstance(path_or_components, dict):
            return [v for v in path_or_components.values() if v and v != "pixel_space"]
        else:
            return [model_display_name]

    def _gpu_logic(self, ui_inputs: Dict, loras_string: str, workflow: Dict[str, Any], assembler: WorkflowAssembler, progress=gr.Progress(track_tqdm=True)):
        model_display_name = ui_inputs['model_display_name']
        succeeded = False
        try:
            progress(0.4, desc="正在执行工作流…")

            initial_objects = {}
            decoded_images_tensor = WorkflowExecutor.execute_workflow(
                workflow, initial_objects=initial_objects
            )

            output_images = []
            raw_seed = ui_inputs.get('seed')
            start_seed = int(raw_seed) if (raw_seed is not None and raw_seed != -1) else random.randint(0, 2**64 - 1)
            image_count = int(decoded_images_tensor.shape[0])
            for i in range(image_count):
                img_tensor = decoded_images_tensor[i]
                pil_image = Image.fromarray((img_tensor.cpu().numpy() * 255.0).astype("uint8"))

                width_for_meta = ui_inputs.get('width', 'N/A')
                height_for_meta = ui_inputs.get('height', 'N/A')

                params_string = f"{ui_inputs['positive_prompt']}\nNegative prompt: {ui_inputs['negative_prompt']}\n"
                params_string += f"Steps: {ui_inputs['num_inference_steps']}, Sampler: {ui_inputs['sampler']}, Scheduler: {ui_inputs['scheduler']}, CFG scale: {ui_inputs['guidance_scale']}, Seed: {start_seed}, Size: {width_for_meta}x{height_for_meta}, Base Model: {model_display_name}"
                if image_count > 1:
                    params_string += f", Batch index: {i + 1}/{image_count}"
                if ui_inputs['task_type'] != 'txt2img': params_string += f", Denoise: {ui_inputs['denoise']}"
                if ui_inputs.get('clip_skip') and ui_inputs['clip_skip'] != 1: params_string += f", Clip skip: {abs(ui_inputs['clip_skip'])}"
                if loras_string: params_string += f", {loras_string}"

                pil_image.info = {'parameters': params_string.strip()}
                output_images.append(pil_image)

            succeeded = True
            return output_images
        finally:
            if not succeeded or ui_inputs.get("_release_models_after_run"):
                from core.model_manager import release_loaded_models

                release_loaded_models()

    @generation_guard
    def run(self, ui_inputs: Dict, progress):
        progress(0, desc="正在准备模型…")

        task_type = ui_inputs['task_type']
        model_display_name = ui_inputs['model_display_name']
        model_type = MODEL_TYPE_MAP.get(model_display_name, 'sdxl')

        architectures_dict = ARCHITECTURES_CONFIG.get('architectures', {})
        workflow_model_type = architectures_dict.get(model_type, {}).get("model_type", model_type.lower().replace(" ", "").replace(".", ""))

        enabled_chains = set(
            FEATURES_CONFIG.get(workflow_model_type, {}).get("enabled_chains", [])
        )
        if task_type != "txt2img":
            has_native_references = any(
                any(ui_inputs.get(input_key) or [])
                for chain_name in self.NATIVE_REFERENCE_CHAINS
                for input_key in self.CHAIN_INPUT_KEYS[chain_name]
            )
            if has_native_references:
                raise gr.Error(
                    "原生多图参考编辑目前只支持“文生图 / 多图融合”。"
                    "普通逐张重绘请使用图生图；不要同时叠加源图 latent 与参考图链。"
                )
        for chain_name, input_keys in self.CHAIN_INPUT_KEYS.items():
            if (
                chain_name not in enabled_chains
                or not supports_chain_for_model(model_display_name, chain_name)
            ):
                for input_key in input_keys:
                    ui_inputs[input_key] = []
        if "pid" not in enabled_chains:
            ui_inputs["pid_settings"] = "OFF"
        if "vae" not in enabled_chains:
            ui_inputs["vae_source"] = None
            ui_inputs["vae_id"] = None
            ui_inputs["vae_file"] = None

        try:
            batch_size = int(ui_inputs.get("batch_size") or 1)
        except (TypeError, ValueError):
            batch_size = 1
        if batch_size < 1 or batch_size > CONFIG.max_batch_size:
            raise gr.Error(f"单次生成数量需为 1–{CONFIG.max_batch_size}。")
        ui_inputs["batch_size"] = batch_size

        ui_inputs['positive_prompt'] = sanitize_prompt(ui_inputs.get('positive_prompt', ''))
        ui_inputs['negative_prompt'] = sanitize_prompt(ui_inputs.get('negative_prompt', ''))

        if 'clip_skip' in ui_inputs and ui_inputs['clip_skip'] is not None:
             ui_inputs['clip_skip'] = -int(ui_inputs['clip_skip'])
        else:
             ui_inputs['clip_skip'] = -1

        required_models = self.get_required_models(model_display_name=model_display_name)

        is_pid_enabled = (ui_inputs.get('pid_settings', 'OFF') == 'ON' and task_type == 'txt2img')
        if is_pid_enabled:
            import yaml
            pid_config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'yaml', 'pid.yaml')
            pid_unet_name = "pid_flux1_1024_to_4096_4step_mxfp8.safetensors"
            try:
                with open(pid_config_path, 'r', encoding='utf-8') as f:
                    pid_config = yaml.safe_load(f) or {}
                pid_items = pid_config.get("PiD", [])
                for item in pid_items:
                    archs = item.get("architectures", [])
                    if workflow_model_type in archs:
                        pid_unet_name = item.get("filepath")
                        break
            except Exception as e:
                print(f"Error loading PiD config for download: {e}")

            if pid_unet_name not in required_models:
                required_models.append(pid_unet_name)
            if "gemma_2_2b_it_elm_fp8_scaled.safetensors" not in required_models:
                required_models.append("gemma_2_2b_it_elm_fp8_scaled.safetensors")

        self.model_manager.ensure_models_downloaded(required_models, progress=progress)
        cancel_event = ui_inputs.get("_cancel_event")
        if cancel_event is not None and cancel_event.is_set():
            raise TaskCancelledError("任务已在模型准备完成后取消，未进入 GPU 生成。")

        temp_files_to_clean = []
        try:
            processed = process_pipeline_inputs(ui_inputs, progress, workflow_model_type)
            temp_files_to_clean.extend(processed["temp_files_to_clean"])

            active_loras_for_gpu = processed["active_loras_for_gpu"]
            active_loras_for_meta = processed["active_loras_for_meta"]
            active_controlnets = processed["active_controlnets"]
            active_anima_controlnets = processed["active_anima_controlnets"]
            active_diffsynth_controlnets = processed["active_diffsynth_controlnets"]
            active_krea2_controlnets = processed.get("active_krea2_controlnets", [])
            active_ipadapters = processed["active_ipadapters"]
            active_flux1_ipadapters = processed["active_flux1_ipadapters"]
            active_sd3_ipadapters = processed["active_sd3_ipadapters"]
            active_styles = processed["active_styles"]
            active_reference_latents = processed["active_reference_latents"]
            active_hidream_o1_reference = processed["active_hidream_o1_reference"]
            active_joyai_reference = processed.get("active_joyai_reference", [])
            active_krea2_identity_edit = processed.get("active_krea2_identity_edit", [])
            active_krea2_reference_edit = processed.get("active_krea2_reference_edit", [])
            active_qwen_image_edit = processed.get("active_qwen_image_edit", [])
            active_boogu_edit = processed.get("active_boogu_edit", [])
            active_reference_images = processed.get("active_reference_images", [])
            active_conditioning = processed["active_conditioning"]

            loras_string = f"LoRAs: [{', '.join(active_loras_for_meta)}]" if active_loras_for_meta else ""

            progress(0.8, desc="正在组装工作流…")

            seed_val = ui_inputs.get('seed')
            if seed_val is None or seed_val == -1:
                ui_inputs['seed'] = random.randint(0, 2**32 - 1)

            model_info = ALL_MODEL_MAP[model_display_name]
            path_or_components = model_info[1]
            latent_type = model_info[3] if len(model_info) > 3 and model_info[3] else 'latent'
            latent_generator_template = "EmptyLatentImage"
            if latent_type == 'sd3_latent':
                latent_generator_template = "EmptySD3LatentImage"
            elif latent_type == 'chroma_radiance_latent':
                latent_generator_template = "EmptyChromaRadianceLatentImage"
            elif latent_type == 'hunyuan_latent':
                latent_generator_template = "EmptyHunyuanImageLatent"

            dynamic_values = {
                'task_type': ui_inputs['task_type'],
                'model_type': workflow_model_type,
                'latent_type': latent_type,
                'latent_generator_template': latent_generator_template
            }

            recipe_path = os.path.join(os.path.dirname(__file__), "workflow_recipes", "unified_recipe.yaml")
            assembler = WorkflowAssembler(recipe_path, dynamic_values=dynamic_values)

            hidream_o1_smoothing_data = []
            if workflow_model_type == 'hidream-o1' and model_display_name == "HiDream-O1-Image":
                hidream_o1_smoothing_data.append({})

            workflow_inputs = {
                **ui_inputs,
                "positive_prompt": ui_inputs['positive_prompt'], "negative_prompt": ui_inputs['negative_prompt'],
                "seed": ui_inputs['seed'], "steps": ui_inputs['num_inference_steps'], "cfg": ui_inputs['guidance_scale'],
                "sampler_name": ui_inputs['sampler'], "scheduler": ui_inputs['scheduler'],
                "batch_size": ui_inputs['batch_size'],
                "clip_skip": ui_inputs['clip_skip'],
                "denoise": ui_inputs['denoise'],
                "vae_name": ui_inputs.get('vae_name'),
                "guidance": ui_inputs.get('guidance', 3.5),
                "lora_chain": active_loras_for_gpu,
                "controlnet_chain": active_controlnets if not active_anima_controlnets else [],
                "anima_controlnet_lllite_chain": active_anima_controlnets,
                "diffsynth_controlnet_chain": active_diffsynth_controlnets,
                "krea2_controlnet_chain": active_krea2_controlnets,
                "ipadapter_chain": active_ipadapters,
                "flux1_ipadapter_chain": active_flux1_ipadapters,
                "sd3_ipadapter_chain": active_sd3_ipadapters,
                "style_chain": active_styles,
                "conditioning_chain": active_conditioning,
                "reference_latent_chain": active_reference_latents,
                "hidream_o1_reference_chain": active_hidream_o1_reference,
                "joyai_image_chain": active_joyai_reference,
                "krea2_identity_edit_chain": active_krea2_identity_edit,
                "krea2_style_reference_chain": active_krea2_reference_edit,
                "qwen_image_edit_chain": active_qwen_image_edit,
                "boogu_image_edit_chain": active_boogu_edit,
                "reference_image_chain": active_reference_images,
                "vae_chain": [ui_inputs.get('vae_name')] if ui_inputs.get('vae_name') else [],
                "hidream_o1_smoothing_chain": hidream_o1_smoothing_data,
                "pid_chain": [ui_inputs.get('pid_settings', 'OFF')] if is_pid_enabled else [],
                "scheduler_width": ui_inputs.get('width', 1024),
                "scheduler_height": ui_inputs.get('height', 1024),
            }

            if isinstance(path_or_components, dict):
                workflow_inputs.update({
                    'unet_name': path_or_components.get('unet'),
                    'unet_uncond_name': path_or_components.get('unet_uncond'),
                    'vae_name': ui_inputs.get('vae_name') or path_or_components.get('vae'),
                    'clip_name': path_or_components.get('clip'),
                    'clip1_name': path_or_components.get('clip1'),
                    'clip2_name': path_or_components.get('clip2'),
                    'clip3_name': path_or_components.get('clip3'),
                    'clip4_name': path_or_components.get('clip4'),
                    'lora_name': path_or_components.get('lora'),
                })
            else:
                workflow_inputs['model_name'] = path_or_components

            if task_type == 'txt2img':
                workflow_inputs['width'] = ui_inputs['width']
                workflow_inputs['height'] = ui_inputs['height']

            workflow = assembler.assemble(workflow_inputs)

            if cancel_event is not None and cancel_event.is_set():
                raise TaskCancelledError("任务已在进入 GPU 前取消。")

            gpu_duration = estimate_gpu_duration(ui_inputs)
            progress(1.0, desc=f"模型已就绪，正在申请 GPU（预计上限 {gpu_duration} 秒）…")

            results = self._execute_gpu_logic(
                self._gpu_logic,
                duration=gpu_duration,
                default_duration=60,
                task_name=f"ImageGen ({task_type})",
                ui_inputs=ui_inputs,
                loras_string=loras_string,
                workflow=workflow,
                assembler=assembler,
                progress=progress
            )

            import json
            import glob
            from PIL import PngImagePlugin

            prompt_json = json.dumps(workflow)

            out_dir = os.path.abspath(OUTPUT_DIR)
            os.makedirs(out_dir, exist_ok=True)

            try:
                existing_files = glob.glob(os.path.join(out_dir, "gen_*.png"))
                existing_files.sort(key=os.path.getmtime)
                keep_existing = max(0, CONFIG.output_retention - len(results))
                while len(existing_files) > keep_existing:
                    os.remove(existing_files.pop(0))
            except Exception as e:
                print(f"Warning: Failed to cleanup output dir: {e}")

            final_results = []
            for img in results:
                if not isinstance(img, Image.Image):
                    final_results.append(img)
                    continue

                metadata = PngImagePlugin.PngInfo()
                params_string = img.info.get("parameters", "")
                if params_string:
                    metadata.add_text("parameters", params_string)
                metadata.add_text("prompt", prompt_json)

                filename = f"gen_{uuid.uuid4().hex}.png"
                filepath = os.path.join(out_dir, filename)
                img.save(filepath, "PNG", pnginfo=metadata)
                final_results.append(filepath)

            results = final_results

        finally:
            for temp_file in temp_files_to_clean:
                if temp_file and os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"✅ Cleaned up temp file: {temp_file}")

        return results
