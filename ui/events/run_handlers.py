import threading

import gradio as gr
from core.execution_plan import (
    ExecutionPlanError,
    MODE_MODEL_PK,
    MODE_MULTI_INDEPENDENT,
    MODE_MULTI_MODEL_GRID,
    MODE_MULTI_REFERENCE,
    build_execution_plan,
    execute_generation_plan,
    load_uploaded_images,
)
from core.generation_logic import generate_image_wrapper
from core.runtime_config import CONFIG
from core.task_scheduler import TaskCancelledError


_CANCEL_LOCK = threading.Lock()
_CANCEL_EVENTS = {}


def _session_key(request: gr.Request) -> str:
    return getattr(request, "session_hash", None) or "anonymous"

def create_run_event(prefix: str, task_type: str, ui_components: dict):
    run_inputs_map = {
        'model_display_name': ui_components[f'base_model_{prefix}'],
        'positive_prompt': ui_components.get(f'prompt_{prefix}') or ui_components.get(f'{prefix}_positive_prompt'),
        'negative_prompt': ui_components.get(f'neg_prompt_{prefix}') or ui_components.get(f'{prefix}_negative_prompt'),
        'seed': ui_components.get(f'seed_{prefix}') or ui_components.get(f'{prefix}_seed'),
        'batch_size': ui_components.get(f'batch_size_{prefix}') or ui_components.get(f'{prefix}_batch_size'),
        'guidance_scale': ui_components.get(f'cfg_{prefix}') or ui_components.get(f'{prefix}_cfg'),
        'num_inference_steps': ui_components.get(f'steps_{prefix}') or ui_components.get(f'{prefix}_steps'),
        'sampler': ui_components.get(f'sampler_{prefix}') or ui_components.get(f'{prefix}_sampler_name'),
        'scheduler': ui_components.get(f'scheduler_{prefix}') or ui_components.get(f'{prefix}_scheduler'),
        'zero_gpu_duration': ui_components.get(f'zero_gpu_{prefix}'),

        'clip_skip': ui_components.get(f'clip_skip_{prefix}'),
        'guidance': ui_components.get(f'guidance_{prefix}'),
        'task_type': gr.State(task_type)
    }

    if ui_components.get(f'pid_settings_{prefix}'):
        run_inputs_map['pid_settings'] = ui_components[f'pid_settings_{prefix}']

    if task_type not in ['img2img', 'inpaint']:
        run_inputs_map.update({
            'width': ui_components.get(f'width_{prefix}') or ui_components.get(f'{prefix}_width'),
            'height': ui_components.get(f'height_{prefix}') or ui_components.get(f'{prefix}_height')
        })

    task_specific_map = {
        'img2img': {'img2img_image': f'input_image_{prefix}', 'img2img_denoise': f'denoise_{prefix}'},
        'inpaint': {'inpaint_image_dict': f'input_image_dict_{prefix}', 'grow_mask_by': f'grow_mask_by_{prefix}', 'inpaint_denoise': f'denoise_{prefix}'},
        'outpaint': {'outpaint_image': f'input_image_{prefix}', 'left': f'left_{prefix}', 'top': f'top_{prefix}', 'right': f'right_{prefix}', 'bottom': f'bottom_{prefix}', 'feathering': f'feathering_{prefix}'},
        'hires_fix': {'hires_image': f'input_image_{prefix}', 'hires_upscaler': f'hires_upscaler_{prefix}', 'hires_scale_by': f'hires_scale_by_{prefix}', 'hires_denoise': f'denoise_{prefix}'}
    }
    if task_type in task_specific_map:
        for key, comp_name in task_specific_map[task_type].items():
            if comp_name in ui_components:
                run_inputs_map[key] = ui_components[comp_name]

    lora_data_components = ui_components.get(f'all_lora_components_flat_{prefix}', [])
    controlnet_data_components = ui_components.get(f'all_controlnet_components_flat_{prefix}', [])
    anima_controlnet_lllite_data_components = ui_components.get(f'all_anima_controlnet_lllite_components_flat_{prefix}', [])
    diffsynth_controlnet_data_components = ui_components.get(f'all_diffsynth_controlnet_components_flat_{prefix}', [])
    krea2_controlnet_data_components = ui_components.get(f'all_krea2_controlnet_components_flat_{prefix}', [])
    ipadapter_data_components = ui_components.get(f'all_ipadapter_components_flat_{prefix}', [])
    sd3_ipadapter_data_components = ui_components.get(f'all_sd3_ipadapter_components_flat_{prefix}', [])
    flux1_ipadapter_data_components = ui_components.get(f'all_flux1_ipadapter_components_flat_{prefix}', [])
    style_data_components = ui_components.get(f'all_style_components_flat_{prefix}', [])
    embedding_data_components = ui_components.get(f'all_embedding_components_flat_{prefix}', [])
    conditioning_data_components = ui_components.get(f'all_conditioning_components_flat_{prefix}', [])
    reference_latent_data_components = ui_components.get(f'all_reference_latent_components_flat_{prefix}', [])
    hidream_o1_reference_data_components = ui_components.get(f'all_hidream_o1_reference_components_flat_{prefix}', [])
    joyai_reference_data_components = ui_components.get(f'all_joyai_reference_components_flat_{prefix}', [])
    krea2_identity_edit_data_components = ui_components.get(f'all_krea2_identity_edit_components_flat_{prefix}', [])
    krea2_reference_edit_data_components = ui_components.get(f'all_krea2_reference_edit_components_flat_{prefix}', [])
    qwen_image_edit_data_components = ui_components.get(f'all_qwen_image_edit_components_flat_{prefix}', [])
    boogu_edit_data_components = ui_components.get(f'all_boogu_edit_components_flat_{prefix}', [])
    reference_image_data_components = ui_components.get(f'all_reference_image_components_flat_{prefix}', [])

    run_inputs_map['vae_source'] = ui_components.get(f'vae_source_{prefix}')
    run_inputs_map['vae_id'] = ui_components.get(f'vae_id_{prefix}')
    run_inputs_map['vae_file'] = ui_components.get(f'vae_file_{prefix}')

    input_keys = list(run_inputs_map.keys())
    input_list_flat = [v for v in run_inputs_map.values() if v is not None]
    all_chains = [
        lora_data_components, controlnet_data_components, anima_controlnet_lllite_data_components, diffsynth_controlnet_data_components, krea2_controlnet_data_components, ipadapter_data_components,
        sd3_ipadapter_data_components, flux1_ipadapter_data_components, style_data_components,
        embedding_data_components, conditioning_data_components, reference_latent_data_components, hidream_o1_reference_data_components, joyai_reference_data_components, krea2_identity_edit_data_components, krea2_reference_edit_data_components, qwen_image_edit_data_components, boogu_edit_data_components, reference_image_data_components
    ]
    for chain in all_chains:
        if chain:
            input_list_flat.extend(chain)

    def create_ui_inputs_dict(*args):
        valid_keys = [k for k in input_keys if run_inputs_map[k] is not None]
        ui_dict = dict(zip(valid_keys, args[:len(valid_keys)]))
        arg_idx = len(valid_keys)

        def assign_chain_data(chain_key, components_list):
            nonlocal arg_idx
            if components_list:
                ui_dict[chain_key] = list(args[arg_idx : arg_idx + len(components_list)])
                arg_idx += len(components_list)

        assign_chain_data('lora_data', lora_data_components)
        assign_chain_data('controlnet_data', controlnet_data_components)
        assign_chain_data('anima_controlnet_lllite_data', anima_controlnet_lllite_data_components)
        assign_chain_data('diffsynth_controlnet_data', diffsynth_controlnet_data_components)
        assign_chain_data('krea2_controlnet_data', krea2_controlnet_data_components)
        assign_chain_data('ipadapter_data', ipadapter_data_components)
        assign_chain_data('sd3_ipadapter_chain', sd3_ipadapter_data_components)
        assign_chain_data('flux1_ipadapter_data', flux1_ipadapter_data_components)
        assign_chain_data('style_data', style_data_components)
        assign_chain_data('embedding_data', embedding_data_components)
        assign_chain_data('conditioning_data', conditioning_data_components)
        assign_chain_data('reference_latent_data', reference_latent_data_components)
        assign_chain_data('hidream_o1_reference_data', hidream_o1_reference_data_components)
        assign_chain_data('joyai_reference_data', joyai_reference_data_components)
        assign_chain_data('krea2_identity_edit_data', krea2_identity_edit_data_components)
        assign_chain_data('krea2_reference_edit_data', krea2_reference_edit_data_components)
        assign_chain_data('qwen_image_edit_data', qwen_image_edit_data_components)
        assign_chain_data('boogu_edit_data', boogu_edit_data_components)
        assign_chain_data('reference_image_data', reference_image_data_components)

        return ui_dict

    run_btn = ui_components.get(f'run_{prefix}') or ui_components.get(f'{prefix}_run_button')
    res_gal = ui_components.get(f'result_{prefix}') or ui_components.get(f'{prefix}_output_gallery')
    if run_btn and res_gal:
        run_btn.click(
            fn=lambda *args, progress=gr.Progress(track_tqdm=True): generate_image_wrapper(create_ui_inputs_dict(*args), progress),
            inputs=input_list_flat,
            outputs=[res_gal]
        )


def create_unified_run_event(prefix: str, ui_components: dict):
    """Attach one generation event shared by all five task modes."""

    run_inputs_map = {
        "task_type": ui_components[f"task_type_{prefix}"],
        "model_display_name": ui_components[f"base_model_{prefix}"],
        "positive_prompt": ui_components[f"prompt_{prefix}"],
        "negative_prompt": ui_components[f"neg_prompt_{prefix}"],
        "seed": ui_components[f"seed_{prefix}"],
        "batch_size": ui_components[f"batch_size_{prefix}"],
        "guidance_scale": ui_components[f"cfg_{prefix}"],
        "num_inference_steps": ui_components[f"steps_{prefix}"],
        "sampler": ui_components[f"sampler_{prefix}"],
        "scheduler": ui_components[f"scheduler_{prefix}"],
        "zero_gpu_duration": ui_components[f"zero_gpu_{prefix}"],
        "clip_skip": ui_components.get(f"clip_skip_{prefix}"),
        "guidance": ui_components.get(f"guidance_{prefix}"),
        "width": ui_components[f"width_{prefix}"],
        "height": ui_components[f"height_{prefix}"],
        "source_image": ui_components[f"source_image_{prefix}"],
        "inpaint_image_dict": ui_components[f"inpaint_image_dict_{prefix}"],
        "img2img_denoise": ui_components[f"img2img_denoise_{prefix}"],
        "inpaint_denoise": ui_components[f"inpaint_denoise_{prefix}"],
        "grow_mask_by": ui_components[f"grow_mask_by_{prefix}"],
        "left": ui_components[f"left_{prefix}"],
        "top": ui_components[f"top_{prefix}"],
        "right": ui_components[f"right_{prefix}"],
        "bottom": ui_components[f"bottom_{prefix}"],
        "feathering": ui_components[f"feathering_{prefix}"],
        "hires_upscaler": ui_components[f"hires_upscaler_{prefix}"],
        "hires_scale_by": ui_components[f"hires_scale_by_{prefix}"],
        "hires_denoise": ui_components[f"hires_denoise_{prefix}"],
        "pid_settings": ui_components.get(f"pid_settings_{prefix}"),
        "vae_source": ui_components.get(f"vae_source_{prefix}"),
        "vae_id": ui_components.get(f"vae_id_{prefix}"),
        "vae_file": ui_components.get(f"vae_file_{prefix}"),
        "_run_mode": ui_components[f"run_mode_{prefix}"],
        "_pk_models": ui_components[f"pk_models_{prefix}"],
        "_batch_images": ui_components[f"batch_images_{prefix}"],
        "_reference_role": ui_components[f"reference_role_{prefix}"],
        "_pk_model_defaults": ui_components[f"pk_model_defaults_{prefix}"],
    }
    run_inputs_map = {key: value for key, value in run_inputs_map.items() if value is not None}
    input_keys = list(run_inputs_map)
    input_components = list(run_inputs_map.values())

    chain_specs = [
        ("lora_data", "all_lora_components_flat"),
        ("controlnet_data", "all_controlnet_components_flat"),
        ("anima_controlnet_lllite_data", "all_anima_controlnet_lllite_components_flat"),
        ("diffsynth_controlnet_data", "all_diffsynth_controlnet_components_flat"),
        ("krea2_controlnet_data", "all_krea2_controlnet_components_flat"),
        ("ipadapter_data", "all_ipadapter_components_flat"),
        ("sd3_ipadapter_chain", "all_sd3_ipadapter_components_flat"),
        ("flux1_ipadapter_data", "all_flux1_ipadapter_components_flat"),
        ("style_data", "all_style_components_flat"),
        ("embedding_data", "all_embedding_components_flat"),
        ("conditioning_data", "all_conditioning_components_flat"),
        ("reference_latent_data", "all_reference_latent_components_flat"),
        ("hidream_o1_reference_data", "all_hidream_o1_reference_components_flat"),
        ("joyai_reference_data", "all_joyai_reference_components_flat"),
        ("krea2_identity_edit_data", "all_krea2_identity_edit_components_flat"),
        ("krea2_reference_edit_data", "all_krea2_reference_edit_components_flat"),
        ("qwen_image_edit_data", "all_qwen_image_edit_components_flat"),
        ("boogu_edit_data", "all_boogu_edit_components_flat"),
        ("reference_image_data", "all_reference_image_components_flat"),
    ]
    chain_components = []
    for chain_key, component_key in chain_specs:
        values = ui_components.get(f"{component_key}_{prefix}", [])
        chain_components.append((chain_key, values))
        input_components.extend(values)

    def create_inputs(*args):
        values = dict(zip(input_keys, args[: len(input_keys)]))
        index = len(input_keys)
        for chain_key, components in chain_components:
            if components:
                values[chain_key] = list(args[index : index + len(components)])
                index += len(components)

        task_type = values["task_type"]
        run_mode = values.get("_run_mode", "single")
        source_image = values.pop("source_image", None)
        batch_driven = run_mode in {MODE_MULTI_INDEPENDENT, MODE_MULTI_MODEL_GRID}
        if task_type == "img2img" and not batch_driven:
            if source_image is None:
                raise gr.Error("图生图需要先上传源图片。")
            values["img2img_image"] = source_image
        elif task_type == "inpaint" and not batch_driven:
            image_dict = values.get("inpaint_image_dict")
            if not image_dict or not image_dict.get("background"):
                raise gr.Error("局部重绘需要先上传图片并涂抹蒙版。")
        elif task_type == "outpaint" and not batch_driven:
            if source_image is None:
                raise gr.Error("扩图需要先上传源图片。")
            values["outpaint_image"] = source_image
        elif task_type == "hires_fix" and not batch_driven:
            if source_image is None:
                raise gr.Error("高清修复需要先上传源图片。")
            values["hires_image"] = source_image
        return values

    run_button = ui_components[f"run_{prefix}"]
    result_gallery = ui_components[f"result_{prefix}"]
    run_summary = ui_components[f"run_summary_{prefix}"]

    def change_run_mode(mode):
        is_pk = mode in {MODE_MODEL_PK, MODE_MULTI_MODEL_GRID}
        uses_files = mode in {
            MODE_MULTI_INDEPENDENT,
            MODE_MULTI_MODEL_GRID,
            MODE_MULTI_REFERENCE,
        }
        return (
            gr.update(visible=is_pk),
            gr.update(visible=uses_files),
            gr.update(visible=mode == MODE_MULTI_REFERENCE),
            gr.update(visible=is_pk),
        )

    ui_components[f"run_mode_{prefix}"].change(
        fn=change_run_mode,
        inputs=[ui_components[f"run_mode_{prefix}"]],
        outputs=[
            ui_components[f"pk_models_{prefix}"],
            ui_components[f"batch_images_{prefix}"],
            ui_components[f"reference_role_{prefix}"],
            ui_components[f"pk_model_defaults_{prefix}"],
        ],
        queue=False,
        show_progress="hidden",
        api_name=False,
        show_api=False,
    )

    def execute(
        request: gr.Request,
        progress=gr.Progress(track_tqdm=True),
        *args,
    ):
        key = _session_key(request)
        cancel_event = threading.Event()
        with _CANCEL_LOCK:
            _CANCEL_EVENTS[key] = cancel_event
        try:
            values = create_inputs(*args)
            values["_cancel_event"] = cancel_event
            run_mode = values.pop("_run_mode", "single")
            pk_models = values.pop("_pk_models", []) or []
            uploaded_files = values.pop("_batch_images", []) or []
            reference_role = values.pop("_reference_role", "auto") or "auto"
            use_model_defaults = bool(values.pop("_pk_model_defaults", True))
            images = (
                load_uploaded_images(uploaded_files)
                if run_mode
                in {
                    MODE_MULTI_INDEPENDENT,
                    MODE_MULTI_MODEL_GRID,
                    MODE_MULTI_REFERENCE,
                }
                else []
            )
            plan = build_execution_plan(
                values,
                mode=run_mode,
                extra_models=pk_models,
                images=images,
                reference_role=reference_role,
                use_model_defaults=use_model_defaults,
            )
            return execute_generation_plan(
                plan,
                generate_image_wrapper,
                progress=progress,
                cancel_event=cancel_event,
            )
        except (TaskCancelledError, ExecutionPlanError) as exc:
            raise gr.Error(str(exc)) from exc
        finally:
            with _CANCEL_LOCK:
                if _CANCEL_EVENTS.get(key) is cancel_event:
                    _CANCEL_EVENTS.pop(key, None)

    def cancel_waiting(request: gr.Request):
        with _CANCEL_LOCK:
            cancel_event = _CANCEL_EVENTS.get(_session_key(request))
            if cancel_event is not None:
                cancel_event.set()

    run_event = run_button.click(
        fn=execute,
        inputs=input_components,
        outputs=[result_gallery, run_summary],
        concurrency_id="gpu_generation",
        concurrency_limit=CONFIG.gpu_concurrency,
        trigger_mode="once",
        show_progress="full",
        show_progress_on=[result_gallery],
        scroll_to_output=True,
        api_name=False,
        show_api=False,
    )

    ui_components[f"cancel_{prefix}"].click(
        fn=cancel_waiting,
        cancels=[run_event],
        queue=False,
        api_name=False,
        show_api=False,
    )
    ui_components[f"clear_result_{prefix}"].click(
        fn=lambda: ([], "已清空本次结果。"),
        outputs=[result_gallery, run_summary],
        queue=False,
        show_progress="hidden",
        api_name=False,
        show_api=False,
    )
