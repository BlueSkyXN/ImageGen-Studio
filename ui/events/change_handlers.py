import gradio as gr
from core.settings import (
    MODEL_TYPE_MAP,
    MODEL_MAP_CHECKPOINT,
    FEATURES_CONFIG,
    ARCHITECTURES_CONFIG,
    MODEL_DEFAULTS_CONFIG,
    ARCH_CATEGORIES_MAP
)
from imagegen_utils.app_utils import get_model_generation_defaults
from core.model_capabilities import supports_chain_for_model
from ui.shared.ui_components import RESOLUTION_MAP
from .config_loaders import (
    get_cn_defaults,
    get_anima_cn_defaults,
    get_diffsynth_cn_defaults,
    get_krea2_cn_defaults,
    load_ipadapter_config
)

def make_update_fn(m_comp, cat_comp, cs_comp, ar_comp, width_comp, height_comp, cn_types, cn_series, cn_filepaths, anima_cn_types, anima_cn_series, anima_cn_filepaths, diffsynth_cn_types, diffsynth_cn_series, diffsynth_cn_filepaths, krea2_cn_types, krea2_cn_series, krea2_cn_filepaths, ipa_preset, lora_acc, cn_acc, anima_cn_acc, diffsynth_cn_acc, krea2_cn_acc, ipa_acc, sd3_ipa_acc, flux1_ipa_acc, style_acc, embed_acc, cond_acc, ref_latent_acc, hidream_o1_ref_acc, guidance_comp, prompt_comp, neg_prompt_comp, steps_comp, cfg_comp, sampler_comp, scheduler_comp, pid_acc=None, vae_acc=None, joyai_ref_acc=None, krea2_identity_edit_acc=None, krea2_reference_edit_acc=None, qwen_image_edit_acc=None, boogu_edit_acc=None, ref_img_acc=None, auto_params_comp=None):
    def update_fn(*args):
        arch = args[0]
        category = args[1]
        current_model = args[2] if len(args) > 2 else None
        idx = 3
        current_ar = args[idx] if ar_comp and len(args) > idx else None
        if ar_comp:
            idx += 1
        auto_params = args[idx] if auto_params_comp and len(args) > idx else True

        if arch == "ALL":
            valid_cats = list(set(cat for cats in ARCH_CATEGORIES_MAP.values() for cat in cats))
        else:
            valid_cats = ARCH_CATEGORIES_MAP.get(arch, [])

        cat_choices = ["ALL"] + sorted(valid_cats)
        new_category = category if category in cat_choices else "ALL"

        choices = []
        for name, info in MODEL_MAP_CHECKPOINT.items():
            m_arch = info[2]
            m_cat = info[4] if len(info) > 4 else None
            arch_match = (arch == "ALL" or m_arch == arch)
            cat_match = (new_category == "ALL" or m_cat == new_category)
            if arch_match and cat_match:
                choices.append(name)

        # Filtering must not silently switch a model that is still valid.
        val = current_model if current_model in choices else (choices[0] if choices else None)

        updates = {
            m_comp: gr.update(choices=choices, value=val),
            cat_comp: gr.update(choices=cat_choices, value=new_category)
        }

        m_type = MODEL_TYPE_MAP.get(val, "SDXL") if val else "SDXL"

        architectures_dict = ARCHITECTURES_CONFIG.get('architectures', {})
        arch_model_type = architectures_dict.get(m_type, {}).get("model_type", m_type.lower().replace(" ", "").replace(".", ""))

        arch_features = FEATURES_CONFIG.get(arch_model_type, {})
        enabled_chains = arch_features.get('enabled_chains', [])

        if lora_acc: updates[lora_acc] = gr.update(visible=('lora' in enabled_chains))
        if cn_acc: updates[cn_acc] = gr.update(visible=('controlnet' in enabled_chains))
        if anima_cn_acc: updates[anima_cn_acc] = gr.update(visible=('anima_controlnet_lllite' in enabled_chains))
        if diffsynth_cn_acc: updates[diffsynth_cn_acc] = gr.update(visible=('diffsynth_controlnet' in enabled_chains))
        if krea2_cn_acc: updates[krea2_cn_acc] = gr.update(visible=('krea2_controlnet' in enabled_chains))
        if ipa_acc: updates[ipa_acc] = gr.update(visible=('ipadapter' in enabled_chains))
        if flux1_ipa_acc: updates[flux1_ipa_acc] = gr.update(visible=('flux1_ipadapter' in enabled_chains))
        if sd3_ipa_acc: updates[sd3_ipa_acc] = gr.update(visible=('sd3_ipadapter' in enabled_chains))
        if style_acc: updates[style_acc] = gr.update(visible=('style' in enabled_chains))
        if embed_acc: updates[embed_acc] = gr.update(visible=('embedding' in enabled_chains))
        if cond_acc: updates[cond_acc] = gr.update(visible=('conditioning' in enabled_chains))
        if ref_latent_acc: updates[ref_latent_acc] = gr.update(visible=('reference_latent' in enabled_chains))
        if hidream_o1_ref_acc: updates[hidream_o1_ref_acc] = gr.update(visible=('hidream_o1_reference' in enabled_chains))
        if joyai_ref_acc: updates[joyai_ref_acc] = gr.update(visible=('joyai_image' in enabled_chains))
        if krea2_identity_edit_acc: updates[krea2_identity_edit_acc] = gr.update(visible=('krea2_identity_edit' in enabled_chains))
        if krea2_reference_edit_acc: updates[krea2_reference_edit_acc] = gr.update(visible=('krea2_style_reference' in enabled_chains))
        if qwen_image_edit_acc: updates[qwen_image_edit_acc] = gr.update(visible=('qwen_image_edit' in enabled_chains and supports_chain_for_model(val, 'qwen_image_edit')))
        if boogu_edit_acc: updates[boogu_edit_acc] = gr.update(visible=('boogu_image_edit' in enabled_chains and supports_chain_for_model(val, 'boogu_image_edit')))
        if ref_img_acc: updates[ref_img_acc] = gr.update(visible=('reference_image' in enabled_chains and supports_chain_for_model(val, 'reference_image')))
        if pid_acc: updates[pid_acc] = gr.update(visible=('pid' in enabled_chains))
        if vae_acc: updates[vae_acc] = gr.update(visible=('vae' in enabled_chains))

        if cs_comp:
            updates[cs_comp] = gr.update(visible=(arch_model_type == "sd15"))
        if guidance_comp:
            updates[guidance_comp] = gr.update(visible=(arch_model_type == "flux1"))

        if ar_comp:
            res_key = arch_model_type
            if res_key not in RESOLUTION_MAP:
                res_key = 'sdxl'
            res_map = RESOLUTION_MAP.get(res_key, {})
            target_ar = current_ar if current_ar in res_map else (list(res_map.keys())[0] if res_map else "1:1 (Square)")
            updates[ar_comp] = gr.update(choices=list(res_map.keys()), value=target_ar)
            if width_comp and height_comp and target_ar in res_map:
                updates[width_comp] = gr.update(value=res_map[target_ar][0])
                updates[height_comp] = gr.update(value=res_map[target_ar][1])

        controlnet_key = architectures_dict.get(m_type, {}).get("controlnet_key", m_type)

        all_types, default_type, series_choices, default_series, filepath = get_cn_defaults(controlnet_key)
        for t_comp in cn_types:
            updates[t_comp] = gr.update(choices=all_types, value=default_type)
        for s_comp in cn_series:
            updates[s_comp] = gr.update(choices=series_choices, value=default_series)
        for f_comp in cn_filepaths:
            updates[f_comp] = filepath

        anima_all_types, anima_default_type, anima_series_choices, anima_default_series, anima_filepath = get_anima_cn_defaults()
        for t_comp in anima_cn_types:
            updates[t_comp] = gr.update(choices=anima_all_types, value=anima_default_type)
        for s_comp in anima_cn_series:
            updates[s_comp] = gr.update(choices=anima_series_choices, value=anima_default_series)
        for f_comp in anima_cn_filepaths:
            updates[f_comp] = anima_filepath

        diffsynth_all_types, diffsynth_default_type, diffsynth_series_choices, diffsynth_default_series, diffsynth_filepath = get_diffsynth_cn_defaults(controlnet_key)
        for t_comp in diffsynth_cn_types:
            updates[t_comp] = gr.update(choices=diffsynth_all_types, value=diffsynth_default_type)
        for s_comp in diffsynth_cn_series:
            updates[s_comp] = gr.update(choices=diffsynth_series_choices, value=diffsynth_default_series)
        for f_comp in diffsynth_cn_filepaths:
            updates[f_comp] = diffsynth_filepath

        krea2_all_types, krea2_default_type, krea2_series_choices, krea2_default_series, krea2_filepath = get_krea2_cn_defaults()
        for t_comp in krea2_cn_types:
            updates[t_comp] = gr.update(choices=krea2_all_types, value=krea2_default_type)
        for s_comp in krea2_cn_series:
            updates[s_comp] = gr.update(choices=krea2_series_choices, value=krea2_default_series)
        for f_comp in krea2_cn_filepaths:
            updates[f_comp] = krea2_filepath

        if ipa_preset and (arch_model_type in ["sdxl", "sd15", "sd35"]):
            config = load_ipadapter_config()
            ipa_arch_key = "SDXL" if arch_model_type in ["sdxl", "sd35"] else "SD1.5"
            std_presets = config.get("IPAdapter_presets", {}).get(ipa_arch_key, [])
            face_presets = config.get("IPAdapter_FaceID_presets", {}).get(ipa_arch_key, [])
            all_ipa_presets = std_presets + face_presets
            default_ipa = all_ipa_presets[0] if all_ipa_presets else None
            updates[ipa_preset] = gr.update(choices=all_ipa_presets, value=default_ipa)

        defaults = get_model_generation_defaults(val, arch_model_type, MODEL_DEFAULTS_CONFIG)
        if steps_comp: updates[steps_comp] = gr.update(value=defaults.get('steps')) if auto_params else gr.update()
        if cfg_comp: updates[cfg_comp] = gr.update(value=defaults.get('cfg')) if auto_params else gr.update()
        if sampler_comp: updates[sampler_comp] = gr.update(value=defaults.get('sampler_name')) if auto_params else gr.update()
        if scheduler_comp: updates[scheduler_comp] = gr.update(value=defaults.get('scheduler')) if auto_params else gr.update()
        # Model/filter changes may tune sampling defaults, but user-authored text is
        # never overwritten. This is especially important when comparing models.
        if prompt_comp: updates[prompt_comp] = gr.update()
        if neg_prompt_comp: updates[neg_prompt_comp] = gr.update()

        return updates
    return update_fn


def make_model_change_fn(cat_comp_ref, cs_comp, ar_comp, width_comp, height_comp, cn_types, cn_series, cn_filepaths, anima_cn_types, anima_cn_series, anima_cn_filepaths, diffsynth_cn_types, diffsynth_cn_series, diffsynth_cn_filepaths, krea2_cn_types, krea2_cn_series, krea2_cn_filepaths, arch_comp_ref, ipa_preset, lora_acc, cn_acc, anima_cn_acc, diffsynth_cn_acc, krea2_cn_acc, ipa_acc, sd3_ipa_acc, flux1_ipa_acc, style_acc, embed_acc, cond_acc, ref_latent_acc, hidream_o1_ref_acc, guidance_comp, prompt_comp, neg_prompt_comp, steps_comp, cfg_comp, sampler_comp, scheduler_comp, pid_acc=None, vae_acc=None, joyai_ref_acc=None, krea2_identity_edit_acc=None, krea2_reference_edit_acc=None, qwen_image_edit_acc=None, boogu_edit_acc=None, ref_img_acc=None, auto_params_comp=None):
    def change_fn(*args):
        model_name = args[0]
        idx = 1
        current_arch = args[idx] if arch_comp_ref and idx < len(args) else None
        if arch_comp_ref: idx += 1
        current_cat = args[idx] if cat_comp_ref and idx < len(args) else None
        if cat_comp_ref: idx += 1
        current_ar = args[idx] if ar_comp and idx < len(args) else None
        if ar_comp:
            idx += 1
        auto_params = args[idx] if auto_params_comp and idx < len(args) else True

        m_type = MODEL_TYPE_MAP.get(model_name, "SDXL")

        m_info = MODEL_MAP_CHECKPOINT.get(model_name)
        m_cat = m_info[4] if m_info and len(m_info) > 4 else None
        if not m_cat: m_cat = "ALL"

        updates = {}
        target_arch = m_type
        if arch_comp_ref:
            if current_arch == "ALL":
                updates[arch_comp_ref] = gr.update()
                target_arch = "ALL"
            else:
                updates[arch_comp_ref] = m_type

        if cat_comp_ref:
            if target_arch == "ALL":
                valid_cats = list(set(cat for cats in ARCH_CATEGORIES_MAP.values() for cat in cats))
            else:
                valid_cats = ARCH_CATEGORIES_MAP.get(target_arch, [])
            cat_choices = ["ALL"] + sorted(valid_cats)

            if current_cat == "ALL":
                updates[cat_comp_ref] = gr.update(choices=cat_choices)
            else:
                updates[cat_comp_ref] = gr.update(choices=cat_choices, value=m_cat)

        architectures_dict = ARCHITECTURES_CONFIG.get('architectures', {})
        arch_model_type = architectures_dict.get(m_type, {}).get("model_type", m_type.lower().replace(" ", "").replace(".", ""))

        arch_features = FEATURES_CONFIG.get(arch_model_type, {})
        enabled_chains = arch_features.get('enabled_chains', [])

        if lora_acc: updates[lora_acc] = gr.update(visible=('lora' in enabled_chains))
        if cn_acc: updates[cn_acc] = gr.update(visible=('controlnet' in enabled_chains))
        if anima_cn_acc: updates[anima_cn_acc] = gr.update(visible=('anima_controlnet_lllite' in enabled_chains))
        if diffsynth_cn_acc: updates[diffsynth_cn_acc] = gr.update(visible=('diffsynth_controlnet' in enabled_chains))
        if krea2_cn_acc: updates[krea2_cn_acc] = gr.update(visible=('krea2_controlnet' in enabled_chains))
        if ipa_acc: updates[ipa_acc] = gr.update(visible=('ipadapter' in enabled_chains))
        if flux1_ipa_acc: updates[flux1_ipa_acc] = gr.update(visible=('flux1_ipadapter' in enabled_chains))
        if sd3_ipa_acc: updates[sd3_ipa_acc] = gr.update(visible=('sd3_ipadapter' in enabled_chains))
        if style_acc: updates[style_acc] = gr.update(visible=('style' in enabled_chains))
        if embed_acc: updates[embed_acc] = gr.update(visible=('embedding' in enabled_chains))
        if cond_acc: updates[cond_acc] = gr.update(visible=('conditioning' in enabled_chains))
        if ref_latent_acc: updates[ref_latent_acc] = gr.update(visible=('reference_latent' in enabled_chains))
        if hidream_o1_ref_acc: updates[hidream_o1_ref_acc] = gr.update(visible=('hidream_o1_reference' in enabled_chains))
        if joyai_ref_acc: updates[joyai_ref_acc] = gr.update(visible=('joyai_image' in enabled_chains))
        if krea2_identity_edit_acc: updates[krea2_identity_edit_acc] = gr.update(visible=('krea2_identity_edit' in enabled_chains))
        if krea2_reference_edit_acc: updates[krea2_reference_edit_acc] = gr.update(visible=('krea2_style_reference' in enabled_chains))
        if qwen_image_edit_acc: updates[qwen_image_edit_acc] = gr.update(visible=('qwen_image_edit' in enabled_chains and supports_chain_for_model(model_name, 'qwen_image_edit')))
        if boogu_edit_acc: updates[boogu_edit_acc] = gr.update(visible=('boogu_image_edit' in enabled_chains and supports_chain_for_model(model_name, 'boogu_image_edit')))
        if ref_img_acc: updates[ref_img_acc] = gr.update(visible=('reference_image' in enabled_chains and supports_chain_for_model(model_name, 'reference_image')))
        if pid_acc: updates[pid_acc] = gr.update(visible=('pid' in enabled_chains))
        if vae_acc: updates[vae_acc] = gr.update(visible=('vae' in enabled_chains))

        if cs_comp:
            updates[cs_comp] = gr.update(visible=(arch_model_type == "sd15"))
        if guidance_comp:
            updates[guidance_comp] = gr.update(visible=(arch_model_type == "flux1"))

        if ar_comp:
            res_key = arch_model_type
            if res_key not in RESOLUTION_MAP:
                res_key = 'sdxl'
            res_map = RESOLUTION_MAP.get(res_key, {})
            target_ar = current_ar if current_ar in res_map else (list(res_map.keys())[0] if res_map else "1:1 (Square)")
            updates[ar_comp] = gr.update(choices=list(res_map.keys()), value=target_ar)
            if width_comp and height_comp and target_ar in res_map:
                updates[width_comp] = gr.update(value=res_map[target_ar][0])
                updates[height_comp] = gr.update(value=res_map[target_ar][1])

        controlnet_key = architectures_dict.get(m_type, {}).get("controlnet_key", m_type)

        all_types, default_type, series_choices, default_series, filepath = get_cn_defaults(controlnet_key)
        for t_comp in cn_types:
            updates[t_comp] = gr.update(choices=all_types, value=default_type)
        for s_comp in cn_series:
            updates[s_comp] = gr.update(choices=series_choices, value=default_series)
        for f_comp in cn_filepaths:
            updates[f_comp] = filepath

        anima_all_types, anima_default_type, anima_series_choices, anima_default_series, anima_filepath = get_anima_cn_defaults()
        for t_comp in anima_cn_types:
            updates[t_comp] = gr.update(choices=anima_all_types, value=anima_default_type)
        for s_comp in anima_cn_series:
            updates[s_comp] = gr.update(choices=anima_series_choices, value=anima_default_series)
        for f_comp in anima_cn_filepaths:
            updates[f_comp] = anima_filepath

        diffsynth_all_types, diffsynth_default_type, diffsynth_series_choices, diffsynth_default_series, diffsynth_filepath = get_diffsynth_cn_defaults(controlnet_key)
        for t_comp in diffsynth_cn_types:
            updates[t_comp] = gr.update(choices=diffsynth_all_types, value=diffsynth_default_type)
        for s_comp in diffsynth_cn_series:
            updates[s_comp] = gr.update(choices=diffsynth_series_choices, value=diffsynth_default_series)
        for f_comp in diffsynth_cn_filepaths:
            updates[f_comp] = diffsynth_filepath

        krea2_all_types, krea2_default_type, krea2_series_choices, krea2_default_series, krea2_filepath = get_krea2_cn_defaults()
        for t_comp in krea2_cn_types:
            updates[t_comp] = gr.update(choices=krea2_all_types, value=krea2_default_type)
        for s_comp in krea2_cn_series:
            updates[s_comp] = gr.update(choices=krea2_series_choices, value=krea2_default_series)
        for f_comp in krea2_cn_filepaths:
            updates[f_comp] = krea2_filepath

        if ipa_preset and (arch_model_type in ["sdxl", "sd15", "sd35"]):
            config = load_ipadapter_config()
            ipa_arch_key = "SDXL" if arch_model_type in ["sdxl", "sd35"] else "SD1.5"
            std_presets = config.get("IPAdapter_presets", {}).get(ipa_arch_key, [])
            face_presets = config.get("IPAdapter_FaceID_presets", {}).get(ipa_arch_key, [])
            all_ipa_presets = std_presets + face_presets
            default_ipa = all_ipa_presets[0] if all_ipa_presets else None
            updates[ipa_preset] = gr.update(choices=all_ipa_presets, value=default_ipa)

        defaults = get_model_generation_defaults(model_name, arch_model_type, MODEL_DEFAULTS_CONFIG)
        if steps_comp: updates[steps_comp] = gr.update(value=defaults.get('steps')) if auto_params else gr.update()
        if cfg_comp: updates[cfg_comp] = gr.update(value=defaults.get('cfg')) if auto_params else gr.update()
        if sampler_comp: updates[sampler_comp] = gr.update(value=defaults.get('sampler_name')) if auto_params else gr.update()
        if scheduler_comp: updates[scheduler_comp] = gr.update(value=defaults.get('scheduler')) if auto_params else gr.update()
        if prompt_comp: updates[prompt_comp] = gr.update()
        if neg_prompt_comp: updates[neg_prompt_comp] = gr.update()

        return updates
    return change_fn


def initialize_all_cn_dropdowns(ui_components):
    default_model_name = list(MODEL_MAP_CHECKPOINT.keys())[0] if MODEL_MAP_CHECKPOINT else None
    default_m_type = MODEL_TYPE_MAP.get(default_model_name, "SDXL") if default_model_name else "SDXL"
    architectures_dict = ARCHITECTURES_CONFIG.get('architectures', {})
    controlnet_key = architectures_dict.get(default_m_type, {}).get("controlnet_key", default_m_type)

    all_types, default_type, series_choices, default_series, filepath = get_cn_defaults(controlnet_key)
    anima_all_types, anima_default_type, anima_series_choices, anima_default_series, anima_filepath = get_anima_cn_defaults()
    diffsynth_all_types, diffsynth_default_type, diffsynth_series_choices, diffsynth_default_series, diffsynth_filepath = get_diffsynth_cn_defaults(controlnet_key)
    krea2_all_types, krea2_default_type, krea2_series_choices, krea2_default_series, krea2_filepath = get_krea2_cn_defaults()

    updates = {}
    prefixes = [item[0] for item in ui_components.get("_task_prefixes", [])]
    if not prefixes:
        prefixes = ["txt2img", "img2img", "inpaint", "outpaint", "hires_fix"]
    for prefix in prefixes:
        if f'controlnet_types_{prefix}' in ui_components:
            for type_dd in ui_components[f'controlnet_types_{prefix}']:
                updates[type_dd] = gr.update(choices=all_types, value=default_type)
            for series_dd in ui_components[f'controlnet_series_{prefix}']:
                updates[series_dd] = gr.update(choices=series_choices, value=default_series)
            for filepath_state in ui_components[f'controlnet_filepaths_{prefix}']:
                updates[filepath_state] = filepath

        if f'anima_controlnet_lllite_types_{prefix}' in ui_components:
            for type_dd in ui_components[f'anima_controlnet_lllite_types_{prefix}']:
                updates[type_dd] = gr.update(choices=anima_all_types, value=anima_default_type)
            for series_dd in ui_components[f'anima_controlnet_lllite_series_{prefix}']:
                updates[series_dd] = gr.update(choices=anima_series_choices, value=anima_default_series)
            for filepath_state in ui_components[f'anima_controlnet_lllite_filepaths_{prefix}']:
                updates[filepath_state] = anima_filepath

        if f'diffsynth_controlnet_types_{prefix}' in ui_components:
            for type_dd in ui_components[f'diffsynth_controlnet_types_{prefix}']:
                updates[type_dd] = gr.update(choices=diffsynth_all_types, value=diffsynth_default_type)
            for series_dd in ui_components[f'diffsynth_controlnet_series_{prefix}']:
                updates[series_dd] = gr.update(choices=diffsynth_series_choices, value=default_series)
            for filepath_state in ui_components[f'diffsynth_controlnet_filepaths_{prefix}']:
                updates[filepath_state] = diffsynth_filepath

        if f'krea2_controlnet_types_{prefix}' in ui_components:
            for type_dd in ui_components[f'krea2_controlnet_types_{prefix}']:
                updates[type_dd] = gr.update(choices=krea2_all_types, value=krea2_default_type)
            for series_dd in ui_components[f'krea2_controlnet_series_{prefix}']:
                updates[series_dd] = gr.update(choices=krea2_series_choices, value=krea2_default_series)
            for filepath_state in ui_components[f'krea2_controlnet_filepaths_{prefix}']:
                updates[filepath_state] = krea2_filepath

    return updates


def initialize_all_ipa_dropdowns(ui_components):
    config = load_ipadapter_config()
    if not config: return {}

    default_model_name = list(MODEL_MAP_CHECKPOINT.keys())[0] if MODEL_MAP_CHECKPOINT else None
    default_m_type = MODEL_TYPE_MAP.get(default_model_name, "SDXL") if default_model_name else "SDXL"
    architectures_dict = ARCHITECTURES_CONFIG.get('architectures', {})
    arch_model_type = architectures_dict.get(default_m_type, {}).get("model_type", default_m_type.lower().replace(" ", "").replace(".", ""))
    ipa_arch_key = "SDXL" if arch_model_type in ["sdxl", "sd35"] else "SD1.5"

    unified_presets = config.get("IPAdapter_presets", {}).get(ipa_arch_key, [])
    faceid_presets = config.get("IPAdapter_FaceID_presets", {}).get(ipa_arch_key, [])

    all_presets = unified_presets + faceid_presets
    default_preset = all_presets[0] if all_presets else None
    is_faceid_default = default_preset in faceid_presets

    lora_strength_update = gr.update(visible=is_faceid_default)

    updates = {}
    prefixes = [item[0] for item in ui_components.get("_task_prefixes", [])]
    if not prefixes:
        prefixes = ["txt2img", "img2img", "inpaint", "outpaint", "hires_fix"]
    for prefix in prefixes:
        if f'ipadapter_final_preset_{prefix}' in ui_components:
            for lora_strength_slider in ui_components[f'ipadapter_lora_strengths_{prefix}']:
                updates[lora_strength_slider] = lora_strength_update
            updates[ui_components[f'ipadapter_final_preset_{prefix}']] = gr.update(choices=all_presets, value=default_preset)
            updates[ui_components[f'ipadapter_final_lora_strength_{prefix}']] = lora_strength_update
    return updates


def run_on_load(ui_components):
    cn_updates = initialize_all_cn_dropdowns(ui_components)
    ipa_updates = initialize_all_ipa_dropdowns(ui_components)
    return {**cn_updates, **ipa_updates}


def on_aspect_ratio_change(ratio_key, model_display_name):
    m_type = MODEL_TYPE_MAP.get(model_display_name, 'SDXL')
    architectures_dict = ARCHITECTURES_CONFIG.get('architectures', {})
    arch_model_type = architectures_dict.get(m_type, {}).get("model_type", m_type.lower().replace(" ", "").replace(".", ""))

    res_map = RESOLUTION_MAP.get(arch_model_type, RESOLUTION_MAP.get("sdxl", {}))
    w, h = res_map.get(ratio_key, (1024, 1024))
    return w, h
