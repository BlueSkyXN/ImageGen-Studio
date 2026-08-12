"""Event wiring for the unified ImageGen Studio workspace."""

from __future__ import annotations

import gradio as gr

from ui.guidance import model_hint, recommended_params, task_help
from .chain_handlers import (
    create_anima_controlnet_lllite_event_handlers,
    create_boogu_edit_event_handlers,
    create_conditioning_event_handlers,
    create_controlnet_event_handlers,
    create_diffsynth_controlnet_event_handlers,
    create_embedding_event_handlers,
    create_flux1_ipadapter_event_handlers,
    create_hidream_o1_reference_event_handlers,
    create_ipadapter_event_handlers,
    create_joyai_reference_event_handlers,
    create_krea2_controlnet_event_handlers,
    create_krea2_identity_edit_event_handlers,
    create_krea2_reference_edit_event_handlers,
    create_lora_event_handlers,
    create_qwen_image_edit_event_handlers,
    create_reference_image_event_handlers,
    create_reference_latent_event_handlers,
    create_sd3_ipadapter_event_handlers,
    create_style_event_handlers,
)
from .change_handlers import (
    make_model_change_fn,
    make_update_fn,
    on_aspect_ratio_change,
    run_on_load,
)
from .run_handlers import create_run_event, create_unified_run_event


CHAIN_EVENT_FACTORIES = (
    create_lora_event_handlers,
    create_controlnet_event_handlers,
    create_anima_controlnet_lllite_event_handlers,
    create_diffsynth_controlnet_event_handlers,
    create_krea2_controlnet_event_handlers,
    create_ipadapter_event_handlers,
    create_embedding_event_handlers,
    create_conditioning_event_handlers,
    create_flux1_ipadapter_event_handlers,
    create_sd3_ipadapter_event_handlers,
    create_style_event_handlers,
    create_reference_latent_event_handlers,
    create_hidream_o1_reference_event_handlers,
    create_joyai_reference_event_handlers,
    create_krea2_identity_edit_event_handlers,
    create_krea2_reference_edit_event_handlers,
    create_qwen_image_edit_event_handlers,
    create_boogu_edit_event_handlers,
    create_reference_image_event_handlers,
)


def _append_if(target, component):
    if component is not None:
        target.append(component)


def _task_switch_values(task_type):
    uses_source = task_type in {"img2img", "outpaint", "hires_fix"}
    labels = {
        "img2img": "源图片（图生图）",
        "outpaint": "源图片（扩图）",
        "hires_fix": "源图片（高清修复）",
    }
    run_labels = {
        "txt2img": "开始生成",
        "img2img": "开始重绘",
        "inpaint": "开始局部重绘",
        "outpaint": "开始扩图",
        "hires_fix": "开始高清修复",
    }
    return (
        task_help(task_type),
        gr.update(visible=uses_source),
        gr.update(label=labels.get(task_type, "源图片")),
        gr.update(visible=task_type == "inpaint"),
        gr.update(visible=task_type == "img2img"),
        gr.update(visible=task_type == "outpaint"),
        gr.update(visible=task_type == "hires_fix"),
        gr.update(visible=task_type == "txt2img"),
        gr.update(value=run_labels.get(task_type, "开始生成")),
    )


def _select_quick_preset(preset):
    if preset == "__manual__":
        return gr.update()
    return gr.update(value=preset)


def _prefixes(ui_components):
    return ui_components.get(
        "_task_prefixes",
        [
            ("txt2img", "txt2img"),
            ("img2img", "img2img"),
            ("inpaint", "inpaint"),
            ("outpaint", "outpaint"),
            ("hires_fix", "hires_fix"),
        ],
    )


def attach_event_handlers(ui_components, demo):
    task_prefixes = _prefixes(ui_components)

    for prefix, task_type in task_prefixes:
        arch_comp = ui_components.get(f"model_arch_{prefix}")
        cat_comp = ui_components.get(f"model_cat_{prefix}")
        model_comp = ui_components.get(f"base_model_{prefix}")
        clip_skip_comp = ui_components.get(f"clip_skip_{prefix}") or ui_components.get(f"{prefix}_clip_skip")
        guidance_comp = ui_components.get(f"guidance_{prefix}") or ui_components.get(f"{prefix}_guidance")
        aspect_ratio_comp = ui_components.get(f"aspect_ratio_{prefix}") or ui_components.get(f"{prefix}_aspect_ratio_dropdown")
        width_comp = ui_components.get(f"width_{prefix}") or ui_components.get(f"{prefix}_width")
        height_comp = ui_components.get(f"height_{prefix}") or ui_components.get(f"{prefix}_height")

        cn_types = ui_components.get(f"controlnet_types_{prefix}", [])
        cn_series = ui_components.get(f"controlnet_series_{prefix}", [])
        cn_filepaths = ui_components.get(f"controlnet_filepaths_{prefix}", [])
        anima_types = ui_components.get(f"anima_controlnet_lllite_types_{prefix}", [])
        anima_series = ui_components.get(f"anima_controlnet_lllite_series_{prefix}", [])
        anima_filepaths = ui_components.get(f"anima_controlnet_lllite_filepaths_{prefix}", [])
        diffsynth_types = ui_components.get(f"diffsynth_controlnet_types_{prefix}", [])
        diffsynth_series = ui_components.get(f"diffsynth_controlnet_series_{prefix}", [])
        diffsynth_filepaths = ui_components.get(f"diffsynth_controlnet_filepaths_{prefix}", [])
        krea2_types = ui_components.get(f"krea2_controlnet_types_{prefix}", [])
        krea2_series = ui_components.get(f"krea2_controlnet_series_{prefix}", [])
        krea2_filepaths = ui_components.get(f"krea2_controlnet_filepaths_{prefix}", [])

        accordion_names = {
            "lora": "lora_accordion",
            "cn": "controlnet_accordion",
            "anima": "anima_controlnet_lllite_accordion",
            "diffsynth": "diffsynth_controlnet_accordion",
            "krea2": "krea2_controlnet_accordion",
            "ipa": "ipadapter_accordion",
            "sd3_ipa": "sd3_ipadapter_accordion",
            "flux1_ipa": "flux1_ipadapter_accordion",
            "style": "style_accordion",
            "embed": "embedding_accordion",
            "cond": "conditioning_accordion",
            "ref_latent": "reference_latent_accordion",
            "hidream": "hidream_o1_reference_accordion",
            "joyai": "joyai_reference_accordion",
            "krea2_identity": "krea2_identity_edit_accordion",
            "krea2_reference": "krea2_reference_edit_accordion",
            "qwen_edit": "qwen_image_edit_accordion",
            "boogu": "boogu_edit_accordion",
            "ref_img": "reference_image_accordion",
            "pid": "pid_accordion",
            "vae": "vae_accordion",
        }
        acc = {
            key: ui_components.get(f"{component_name}_{prefix}")
            for key, component_name in accordion_names.items()
        }
        ipa_preset = ui_components.get(f"ipadapter_final_preset_{prefix}")

        prompt_comp = ui_components.get(f"prompt_{prefix}") or ui_components.get(f"{prefix}_positive_prompt")
        neg_prompt_comp = ui_components.get(f"neg_prompt_{prefix}") or ui_components.get(f"{prefix}_negative_prompt")
        steps_comp = ui_components.get(f"steps_{prefix}") or ui_components.get(f"{prefix}_steps")
        cfg_comp = ui_components.get(f"cfg_{prefix}") or ui_components.get(f"{prefix}_cfg")
        sampler_comp = ui_components.get(f"sampler_{prefix}") or ui_components.get(f"{prefix}_sampler_name")
        scheduler_comp = ui_components.get(f"scheduler_{prefix}") or ui_components.get(f"{prefix}_scheduler")
        auto_params_comp = ui_components.get(f"auto_model_params_{prefix}")
        parameter_outputs = [
            item
            for item in (
                prompt_comp,
                neg_prompt_comp,
                steps_comp,
                cfg_comp,
                sampler_comp,
                scheduler_comp,
                width_comp,
                height_comp,
            )
            if item is not None
        ]

        if arch_comp is not None and cat_comp is not None and model_comp is not None:
            filter_outputs = [model_comp, cat_comp]
            for component in (clip_skip_comp, guidance_comp, aspect_ratio_comp):
                _append_if(filter_outputs, component)
            filter_outputs.extend(
                cn_types + cn_series + cn_filepaths
                + anima_types + anima_series + anima_filepaths
                + diffsynth_types + diffsynth_series + diffsynth_filepaths
                + krea2_types + krea2_series + krea2_filepaths
            )
            for component in acc.values():
                _append_if(filter_outputs, component)
            _append_if(filter_outputs, ipa_preset)
            filter_outputs.extend(parameter_outputs)

            filter_fn = make_update_fn(
                model_comp, cat_comp, clip_skip_comp, aspect_ratio_comp, width_comp, height_comp,
                cn_types, cn_series, cn_filepaths,
                anima_types, anima_series, anima_filepaths,
                diffsynth_types, diffsynth_series, diffsynth_filepaths,
                krea2_types, krea2_series, krea2_filepaths,
                ipa_preset, acc["lora"], acc["cn"], acc["anima"], acc["diffsynth"],
                acc["krea2"], acc["ipa"], acc["sd3_ipa"], acc["flux1_ipa"],
                acc["style"], acc["embed"], acc["cond"], acc["ref_latent"],
                acc["hidream"], guidance_comp, prompt_comp, neg_prompt_comp, steps_comp,
                cfg_comp, sampler_comp, scheduler_comp, pid_acc=acc["pid"],
                vae_acc=acc["vae"], joyai_ref_acc=acc["joyai"],
                krea2_identity_edit_acc=acc["krea2_identity"],
                krea2_reference_edit_acc=acc["krea2_reference"],
                qwen_image_edit_acc=acc["qwen_edit"], boogu_edit_acc=acc["boogu"],
                ref_img_acc=acc["ref_img"], auto_params_comp=auto_params_comp,
            )
            filter_inputs = [arch_comp, cat_comp, model_comp]
            if aspect_ratio_comp is not None:
                filter_inputs.append(aspect_ratio_comp)
            if auto_params_comp is not None:
                filter_inputs.append(auto_params_comp)
            arch_comp.change(filter_fn, filter_inputs, filter_outputs, show_progress="hidden")
            cat_comp.change(filter_fn, filter_inputs, filter_outputs, show_progress="hidden")

        if model_comp is not None:
            model_outputs = []
            for component in (arch_comp, cat_comp, clip_skip_comp, guidance_comp, aspect_ratio_comp):
                _append_if(model_outputs, component)
            model_outputs.extend(
                cn_types + cn_series + cn_filepaths
                + anima_types + anima_series + anima_filepaths
                + diffsynth_types + diffsynth_series + diffsynth_filepaths
                + krea2_types + krea2_series + krea2_filepaths
            )
            for component in acc.values():
                _append_if(model_outputs, component)
            _append_if(model_outputs, ipa_preset)
            model_outputs.extend(parameter_outputs)

            model_inputs = [model_comp]
            for component in (arch_comp, cat_comp, aspect_ratio_comp):
                _append_if(model_inputs, component)
            _append_if(model_inputs, auto_params_comp)
            model_fn = make_model_change_fn(
                cat_comp, clip_skip_comp, aspect_ratio_comp, width_comp, height_comp,
                cn_types, cn_series, cn_filepaths,
                anima_types, anima_series, anima_filepaths,
                diffsynth_types, diffsynth_series, diffsynth_filepaths,
                krea2_types, krea2_series, krea2_filepaths,
                arch_comp, ipa_preset, acc["lora"], acc["cn"], acc["anima"],
                acc["diffsynth"], acc["krea2"], acc["ipa"], acc["sd3_ipa"],
                acc["flux1_ipa"], acc["style"], acc["embed"], acc["cond"],
                acc["ref_latent"], acc["hidream"], guidance_comp, prompt_comp,
                neg_prompt_comp, steps_comp, cfg_comp, sampler_comp, scheduler_comp,
                pid_acc=acc["pid"], vae_acc=acc["vae"], joyai_ref_acc=acc["joyai"],
                krea2_identity_edit_acc=acc["krea2_identity"],
                krea2_reference_edit_acc=acc["krea2_reference"],
                qwen_image_edit_acc=acc["qwen_edit"], boogu_edit_acc=acc["boogu"],
                ref_img_acc=acc["ref_img"], auto_params_comp=auto_params_comp,
            )
            model_comp.change(model_fn, model_inputs, model_outputs, show_progress="hidden")

        for factory in CHAIN_EVENT_FACTORIES:
            factory(prefix, ui_components)

        if task_type is None:
            create_unified_run_event(prefix, ui_components)
        else:
            create_run_event(prefix, task_type, ui_components)

        if all(component is not None for component in (aspect_ratio_comp, width_comp, height_comp, model_comp)):
            aspect_ratio_comp.change(
                on_aspect_ratio_change,
                [aspect_ratio_comp, model_comp],
                [width_comp, height_comp],
                show_progress="hidden",
                api_name=False,
                show_api=False,
            )

        if task_type is None:
            task_component = ui_components[f"task_type_{prefix}"]
            task_component.change(
                _task_switch_values,
                [task_component],
                [
                    ui_components[f"task_help_{prefix}"],
                    ui_components[f"source_panel_{prefix}"],
                    ui_components[f"source_image_{prefix}"],
                    ui_components[f"inpaint_panel_{prefix}"],
                    ui_components[f"img2img_panel_{prefix}"],
                    ui_components[f"outpaint_panel_{prefix}"],
                    ui_components[f"hires_panel_{prefix}"],
                    ui_components[f"size_panel_{prefix}"],
                    ui_components[f"run_{prefix}"],
                ],
                show_progress="hidden",
                api_name=False,
                show_api=False,
            )
            preset = ui_components[f"quick_preset_{prefix}"]
            preset.change(
                _select_quick_preset,
                [preset],
                [model_comp],
                show_progress="hidden",
                api_name=False,
                show_api=False,
            )
            model_comp.input(
                lambda: "__manual__",
                outputs=[preset],
                show_progress="hidden",
                api_name=False,
                show_api=False,
            )
            model_comp.change(
                model_hint,
                [model_comp],
                [ui_components[f"model_hint_{prefix}"]],
                show_progress="hidden",
                api_name=False,
                show_api=False,
            )
            ui_components[f"reset_model_params_{prefix}"].click(
                recommended_params,
                [model_comp],
                [steps_comp, cfg_comp, sampler_comp, scheduler_comp],
                show_progress="hidden",
                api_name=False,
                show_api=False,
            )

    load_outputs = []
    for prefix, _ in task_prefixes:
        for base_name in (
            "controlnet_types", "controlnet_series", "controlnet_filepaths",
            "anima_controlnet_lllite_types", "anima_controlnet_lllite_series",
            "anima_controlnet_lllite_filepaths", "diffsynth_controlnet_types",
            "diffsynth_controlnet_series", "diffsynth_controlnet_filepaths",
            "krea2_controlnet_types", "krea2_controlnet_series", "krea2_controlnet_filepaths",
        ):
            load_outputs.extend(ui_components.get(f"{base_name}_{prefix}", []))
        if f"ipadapter_final_preset_{prefix}" in ui_components:
            load_outputs.extend(ui_components.get(f"ipadapter_lora_strengths_{prefix}", []))
            load_outputs.append(ui_components[f"ipadapter_final_preset_{prefix}"])
            load_outputs.append(ui_components[f"ipadapter_final_lora_strength_{prefix}"])

    if load_outputs:
        demo.load(
            lambda: run_on_load(ui_components),
            outputs=load_outputs,
            show_progress="hidden",
            api_name=False,
            show_api=False,
        )
