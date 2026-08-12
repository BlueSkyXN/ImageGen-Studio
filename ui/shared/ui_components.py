import gradio as gr
from comfy_integration.nodes import SAMPLER_CHOICES, SCHEDULER_CHOICES
from core.settings import (
    MAX_LORAS, LORA_SOURCE_CHOICES, MAX_EMBEDDINGS, MAX_CONDITIONINGS,
    MAX_CONTROLNETS, MAX_IPADAPTERS, RESOLUTION_MAP, ARCHITECTURES_CONFIG,
    MODEL_MAP_CHECKPOINT, MODEL_TYPE_MAP, FEATURES_CONFIG, ARCH_CATEGORIES_MAP,
    VAE_DIR, MODEL_DEFAULTS_CONFIG
)
import yaml
import os
from functools import lru_cache
from imagegen_utils.app_utils import save_uploaded_file_with_hash

default_model_name = list(MODEL_MAP_CHECKPOINT.keys())[0] if MODEL_MAP_CHECKPOINT else None
default_m_type = MODEL_TYPE_MAP.get(default_model_name, "SDXL") if default_model_name else "SDXL"
default_architectures_dict = ARCHITECTURES_CONFIG.get('architectures', {})
default_arch_model_type = default_architectures_dict.get(default_m_type, {}).get("model_type", default_m_type.lower().replace(" ", "").replace(".", ""))
default_arch_features = FEATURES_CONFIG.get(default_arch_model_type, {})
default_enabled_chains = default_arch_features.get('enabled_chains', [])

default_vals = MODEL_DEFAULTS_CONFIG.get('Default', {})
DEFAULT_STEPS = default_vals.get('steps', 20)
DEFAULT_CFG = default_vals.get('cfg', 5.0)
DEFAULT_SAMPLER = default_vals.get('sampler_name', 'euler')
DEFAULT_SCHEDULER = default_vals.get('scheduler', 'simple')
DEFAULT_POS_PROMPT = default_vals.get('positive_prompt', '')
DEFAULT_NEG_PROMPT = default_vals.get('negative_prompt', '')

@lru_cache(maxsize=1)
def get_ipadapter_config_from_yaml():
    try:
        _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        _IPADAPTER_LIST_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'ipadapter.yaml')
        with open(_IPADAPTER_LIST_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Warning: Could not load ipadapter.yaml for UI components: {e}")
        return {}

def get_ipadapter_presets(arch="SDXL"):
    config = get_ipadapter_config_from_yaml()
    presets = []
    if config:
        std_presets = config.get("IPAdapter_presets", {}).get(arch, [])
        face_presets = config.get("IPAdapter_FaceID_presets", {}).get(arch, [])
        if std_presets:
            presets.extend(std_presets)
        if face_presets:
            presets.extend(face_presets)
    return presets if presets else ["STANDARD (medium strength)"]

def create_model_architecture_filter_ui(prefix):
    components = {}
    ordered_architectures = ARCHITECTURES_CONFIG.get("architecture_order", [])
    choices = ["ALL"] + ordered_architectures

    components[f'model_arch_{prefix}'] = gr.Dropdown(
        label="模型架构",
        choices=choices,
        value="ALL",
        interactive=True,
        visible=True,
        filterable=True,
        info="用于缩小模型范围；保持 ALL 可直接搜索全部模型。",
    )
    return components

def create_category_filter_ui(prefix):
    valid_cats = list(set(cat for cats in ARCH_CATEGORIES_MAP.values() for cat in cats))
    cat_choices = ["ALL"] + sorted(valid_cats)

    components = {}
    components[f'model_cat_{prefix}'] = gr.Dropdown(
        label="模型分类",
        choices=cat_choices,
        value="ALL",
        interactive=True,
        scale=1,
        allow_custom_value=False,
        info="主要用于 SDXL 的 NoobAI、Illustrious、Pony 等分类。",
    )
    return components

def create_base_parameter_ui(prefix, defaults=None):
    if defaults is None:
        defaults = {}

    components = {}

    with gr.Row():
        components[f'aspect_ratio_{prefix}'] = gr.Dropdown(
            label="Aspect Ratio",
            choices=list(RESOLUTION_MAP.get('sdxl', {}).keys()),
            value="1:1 (Square)",
            interactive=True,
            allow_custom_value=True
        )
    with gr.Row():
        components[f'width_{prefix}'] = gr.Number(label="Width", value=defaults.get('w', 1024), interactive=True)
        components[f'height_{prefix}'] = gr.Number(label="Height", value=defaults.get('h', 1024), interactive=True)
    with gr.Row():
        components[f'sampler_{prefix}'] = gr.Dropdown(
            label="Sampler",
            choices=SAMPLER_CHOICES,
            value=DEFAULT_SAMPLER if DEFAULT_SAMPLER in SAMPLER_CHOICES else (SAMPLER_CHOICES[0] if SAMPLER_CHOICES else 'euler')
        )
        components[f'scheduler_{prefix}'] = gr.Dropdown(
            label="Scheduler",
            choices=SCHEDULER_CHOICES,
            value=DEFAULT_SCHEDULER if DEFAULT_SCHEDULER in SCHEDULER_CHOICES else (SCHEDULER_CHOICES[0] if SCHEDULER_CHOICES else 'simple')
        )
    with gr.Row():
        components[f'steps_{prefix}'] = gr.Slider(label="Steps", minimum=1, maximum=100, step=1, value=DEFAULT_STEPS)
        components[f'cfg_{prefix}'] = gr.Slider(label="CFG Scale", minimum=1.0, maximum=20.0, step=0.1, value=DEFAULT_CFG)
    with gr.Row():
        components[f'seed_{prefix}'] = gr.Number(label="Seed (-1 for random)", value=-1, precision=0)
        components[f'batch_size_{prefix}'] = gr.Slider(label="Batch Size", minimum=1, maximum=16, step=1, value=1)
    with gr.Row():
        components[f'clip_skip_{prefix}'] = gr.Slider(label="Clip Skip", minimum=1, maximum=2, step=1, value=1, visible=False, interactive=True)
        components[f'guidance_{prefix}'] = gr.Slider(label="Guidance (FLUX)", minimum=1.0, maximum=10.0, step=0.1, value=3.5, visible=False, interactive=True)
        components[f'zero_gpu_{prefix}'] = gr.Number(label="ZeroGPU Duration (s)", value=None, placeholder="Default: 60s, Max: 120s", info="Optional: Set how long to reserve the GPU.")

    return components


def create_lora_settings_ui(prefix: str):
    components = {}

    lora_rows, lora_sources, lora_ids, lora_scales, lora_uploads = [], [], [], [], []

    with gr.Accordion("LoRA 设置", open=False, visible=('lora' in default_enabled_chains)) as lora_accordion:
        components[f'lora_accordion_{prefix}'] = lora_accordion
        gr.Markdown("💡 Civitai 请填写 **Version ID**，不要填 Model ID；Hugging Face 请填写 `repo_id/文件路径`。")
        components[f'lora_count_state_{prefix}'] = gr.State(1)

        for i in range(MAX_LORAS):
            with gr.Row(visible=i==0) as row:
                source = gr.Dropdown(label=f"LoRA 来源 {i+1}", choices=LORA_SOURCE_CHOICES, value=LORA_SOURCE_CHOICES[0], scale=1)
                lora_id = gr.Textbox(label="Civitai Version ID / HF 文件 / 已上传文件", scale=2, type="text")
                scale = gr.Slider(label="权重", minimum=0.0, maximum=2.0, step=0.05, value=1.0, scale=1)
                upload = gr.UploadButton(label="上传", file_types=[".safetensors"], scale=1)

            lora_rows.append(row)
            lora_sources.append(source)
            lora_ids.append(lora_id)
            lora_scales.append(scale)
            lora_uploads.append(upload)

        with gr.Row():
            components[f'add_lora_button_{prefix}'] = gr.Button("添加 LoRA", variant="secondary")
            components[f'delete_lora_button_{prefix}'] = gr.Button("移除 LoRA", variant="secondary", visible=False)

    components[f'lora_rows_{prefix}'] = lora_rows
    components[f'lora_sources_{prefix}'] = lora_sources
    components[f'lora_ids_{prefix}'] = lora_ids
    components[f'lora_scales_{prefix}'] = lora_scales
    components[f'lora_uploads_{prefix}'] = lora_uploads

    all_lora_components_flat = []
    for i in range(MAX_LORAS):
        all_lora_components_flat.extend([lora_sources[i], lora_ids[i], lora_scales[i], lora_uploads[i]])
    components[f'all_lora_components_flat_{prefix}'] = all_lora_components_flat

    return components

def create_controlnet_ui(prefix: str, max_units=MAX_CONTROLNETS):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("ControlNet 设置", open=False, visible=('controlnet' in default_enabled_chains)) as accordion:
        components[key('controlnet_accordion')] = accordion

        cn_rows, images, series, types, strengths, filepaths = [], [], [], [], [], []
        components.update({
            key('controlnet_rows'): cn_rows,
            key('controlnet_images'): images,
            key('controlnet_series'): series,
            key('controlnet_types'): types,
            key('controlnet_strengths'): strengths,
            key('controlnet_filepaths'): filepaths
        })

        for i in range(max_units):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"控制图 {i+1}", type="pil", sources=["upload"], height=256))
                with gr.Column(scale=2):
                    types.append(gr.Dropdown(label="类型", choices=[], interactive=True, allow_custom_value=True))
                    series.append(gr.Dropdown(label="系列", choices=[], interactive=True, allow_custom_value=True))
                    strengths.append(gr.Slider(label="强度", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True))
                    filepaths.append(gr.State(None))
                cn_rows.append(row)

        with gr.Row():
            components[key('add_controlnet_button')] = gr.Button("✚ 添加 ControlNet")
            components[key('delete_controlnet_button')] = gr.Button("➖ 移除 ControlNet", visible=False)
        components[key('controlnet_count_state')] = gr.State(1)

        all_cn_components_flat = []
        for i in range(max_units):
            all_cn_components_flat.extend([
                images[i], types[i], series[i], strengths[i], filepaths[i]
            ])
        components[key('all_controlnet_components_flat')] = all_cn_components_flat

    return components

def create_krea2_controlnet_ui(prefix: str, max_units=MAX_CONTROLNETS):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("Krea2 ControlNet 设置", open=False, visible=('krea2_controlnet' in default_enabled_chains)) as accordion:
        components[key('krea2_controlnet_accordion')] = accordion
        gr.Markdown("💡 由 [facok/comfyui-krea2-controlnet](https://github.com/facok/comfyui-krea2-controlnet) 节点处理。")

        cn_rows, images, series, types, strengths, filepaths = [], [], [], [], [], []
        components.update({
            key('krea2_controlnet_rows'): cn_rows,
            key('krea2_controlnet_images'): images,
            key('krea2_controlnet_series'): series,
            key('krea2_controlnet_types'): types,
            key('krea2_controlnet_strengths'): strengths,
            key('krea2_controlnet_filepaths'): filepaths
        })

        for i in range(max_units):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"控制图 {i+1}", type="pil", sources=["upload"], height=256))
                with gr.Column(scale=2):
                    types.append(gr.Dropdown(label="类型", choices=[], interactive=True, allow_custom_value=True))
                    series.append(gr.Dropdown(label="系列", choices=[], interactive=True, allow_custom_value=True))
                    strengths.append(gr.Slider(label="强度", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True))
                    filepaths.append(gr.State(None))
                cn_rows.append(row)

        with gr.Row():
            components[key('add_krea2_controlnet_button')] = gr.Button("✚ 添加 Krea2 ControlNet")
            components[key('delete_krea2_controlnet_button')] = gr.Button("➖ 移除 Krea2 ControlNet", visible=False)
        components[key('krea2_controlnet_count_state')] = gr.State(1)

        all_cn_components_flat = []
        for i in range(max_units):
            all_cn_components_flat.extend([
                images[i], types[i], series[i], strengths[i], filepaths[i]
            ])
        components[key('all_krea2_controlnet_components_flat')] = all_cn_components_flat

    return components

def create_anima_controlnet_lllite_ui(prefix: str, max_units=MAX_CONTROLNETS):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("Anima ControlNet LLLite 设置", open=False, visible=('anima_controlnet_lllite' in default_enabled_chains)) as accordion:
        components[key('anima_controlnet_lllite_accordion')] = accordion

        cn_rows, images, series, types, strengths, filepaths, start_percents, end_percents = [], [], [], [], [], [], [], []
        components.update({
            key('anima_controlnet_lllite_rows'): cn_rows,
            key('anima_controlnet_lllite_images'): images,
            key('anima_controlnet_lllite_series'): series,
            key('anima_controlnet_lllite_types'): types,
            key('anima_controlnet_lllite_strengths'): strengths,
            key('anima_controlnet_lllite_filepaths'): filepaths,
            key('anima_controlnet_lllite_start_percents'): start_percents,
            key('anima_controlnet_lllite_end_percents'): end_percents
        })

        for i in range(max_units):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"控制图 {i+1}", type="pil", sources=["upload"], height=256))
                with gr.Column(scale=2):
                    types.append(gr.Dropdown(label="类型", choices=[], interactive=True, allow_custom_value=True))
                    series.append(gr.Dropdown(label="系列", choices=[], interactive=True, allow_custom_value=True))
                    strengths.append(gr.Slider(label="强度", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True))
                    with gr.Row(visible=False):
                        start_percents.append(gr.State(0.0))
                        end_percents.append(gr.State(1.0))
                    filepaths.append(gr.State(None))
                cn_rows.append(row)

        with gr.Row():
            components[key('add_anima_controlnet_lllite_button')] = gr.Button("✚ 添加 LLLite")
            components[key('delete_anima_controlnet_lllite_button')] = gr.Button("➖ 移除 LLLite", visible=False)
        components[key('anima_controlnet_lllite_count_state')] = gr.State(1)

        all_cn_components_flat = []
        for i in range(max_units):
            all_cn_components_flat.extend([
                images[i], types[i], series[i], strengths[i], filepaths[i], start_percents[i], end_percents[i]
            ])
        components[key('all_anima_controlnet_lllite_components_flat')] = all_cn_components_flat

    return components

def create_diffsynth_controlnet_ui(prefix: str, max_units=MAX_CONTROLNETS):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("DiffSynth ControlNet 设置", open=False, visible=('diffsynth_controlnet' in default_enabled_chains)) as accordion:
        components[key('diffsynth_controlnet_accordion')] = accordion

        cn_rows, images, series, types, strengths, filepaths = [], [], [], [], [], []
        components.update({
            key('diffsynth_controlnet_rows'): cn_rows,
            key('diffsynth_controlnet_images'): images,
            key('diffsynth_controlnet_series'): series,
            key('diffsynth_controlnet_types'): types,
            key('diffsynth_controlnet_strengths'): strengths,
            key('diffsynth_controlnet_filepaths'): filepaths
        })

        for i in range(max_units):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"控制图 {i+1}", type="pil", sources=["upload"], height=256))
                with gr.Column(scale=2):
                    types.append(gr.Dropdown(label="类型", choices=[], interactive=True, allow_custom_value=True))
                    series.append(gr.Dropdown(label="系列", choices=[], interactive=True, allow_custom_value=True))
                    strengths.append(gr.Slider(label="强度", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True))
                    filepaths.append(gr.State(None))
                cn_rows.append(row)

        with gr.Row():
            components[key('add_diffsynth_controlnet_button')] = gr.Button("✚ 添加 DiffSynth ControlNet")
            components[key('delete_diffsynth_controlnet_button')] = gr.Button("➖ 移除 DiffSynth ControlNet", visible=False)
        components[key('diffsynth_controlnet_count_state')] = gr.State(1)

        all_cn_components_flat = []
        for i in range(max_units):
            all_cn_components_flat.extend([
                images[i], types[i], series[i], strengths[i], filepaths[i]
            ])
        components[key('all_diffsynth_controlnet_components_flat')] = all_cn_components_flat

    return components

def create_ipadapter_ui(prefix: str, max_units=MAX_IPADAPTERS):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    sdxl_presets = get_ipadapter_presets("SDXL")
    default_preset = sdxl_presets[0] if sdxl_presets else None

    with gr.Accordion("IP-Adapter 设置", open=False, visible=('ipadapter' in default_enabled_chains)) as accordion:
        components[key('ipadapter_accordion')] = accordion
        gr.Markdown("💡 由 [cubiq/ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) 节点处理。")

        with gr.Row():
            components[key('ipadapter_final_preset')] = gr.Dropdown(
                label="预设（应用于全部参考图）",
                choices=sdxl_presets,
                value=default_preset,
                interactive=True,
                allow_custom_value=True
            )
            components[key('ipadapter_embeds_scaling')] = gr.Dropdown(
                label="特征缩放方式",
                choices=['V only', 'K+V', 'K+V w/ C penalty', 'K+mean(V) w/ C penalty'],
                value='V only',
                interactive=True
            )

        with gr.Row():
            components[key('ipadapter_combine_method')] = gr.Dropdown(
                label="合并方式",
                choices=["concat", "add", "subtract", "average", "norm average", "max", "min"],
                value="concat",
                interactive=True
            )
            components[key('ipadapter_final_weight')] = gr.Slider(label="最终权重", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True)
            components[key('ipadapter_final_lora_strength')] = gr.Slider(label="最终 LoRA 强度", minimum=0.0, maximum=2.0, step=0.05, value=0.6, interactive=True, visible=False)

        gr.Markdown("---")

        ipa_rows, images, weights, lora_strengths = [], [], [], []
        components.update({
            key('ipadapter_rows'): ipa_rows,
            key('ipadapter_images'): images,
            key('ipadapter_weights'): weights,
            key('ipadapter_lora_strengths'): lora_strengths
        })

        for i in range(max_units):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"IP-Adapter 参考图 {i+1}", type="pil", sources=["upload"], height=256))
                with gr.Column(scale=2):
                    weights.append(gr.Slider(label="权重", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True))
                    lora_strengths.append(gr.Slider(label="LoRA 强度", minimum=0.0, maximum=2.0, step=0.05, value=0.6, interactive=True, visible=False))
                ipa_rows.append(row)

        with gr.Row():
            components[key('add_ipadapter_button')] = gr.Button("✚ 添加 IP-Adapter")
            components[key('delete_ipadapter_button')] = gr.Button("➖ 移除 IP-Adapter", visible=False)
        components[key('ipadapter_count_state')] = gr.State(1)

        all_ipa_components_flat = images + weights + lora_strengths
        all_ipa_components_flat += [
            components[key('ipadapter_final_preset')],
            components[key('ipadapter_final_weight')],
            components[key('ipadapter_final_lora_strength')],
            components[key('ipadapter_embeds_scaling')],
            components[key('ipadapter_combine_method')],
        ]
        components[key('all_ipadapter_components_flat')] = all_ipa_components_flat

    return components

def create_flux1_ipadapter_ui(prefix: str, max_units=MAX_IPADAPTERS):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("FLUX.1 IP-Adapter 设置", open=False, visible=('flux1_ipadapter' in default_enabled_chains)) as accordion:
        components[key('flux1_ipadapter_accordion')] = accordion
        gr.Markdown("💡 由 [Shakker-Labs/ComfyUI-IPAdapter-Flux](https://github.com/Shakker-Labs/ComfyUI-IPAdapter-Flux) 节点处理。")

        ipa_rows, images, weights, start_percents, end_percents = [], [], [], [], []
        components.update({
            key('flux1_ipadapter_rows'): ipa_rows,
            key('flux1_ipadapter_images'): images,
            key('flux1_ipadapter_weights'): weights,
            key('flux1_ipadapter_start_percents'): start_percents,
            key('flux1_ipadapter_end_percents'): end_percents,
        })

        for i in range(max_units):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"IP-Adapter 参考图 {i+1}", type="pil", sources=["upload"], height=256))
                with gr.Column(scale=2):
                    weights.append(gr.Slider(label="权重", minimum=0.0, maximum=2.0, step=0.05, value=0.6, interactive=True))
                    with gr.Row():
                        start_percents.append(gr.Slider(label="开始位置", minimum=0.0, maximum=1.0, step=0.01, value=0.0, interactive=True))
                        end_percents.append(gr.Slider(label="结束位置", minimum=0.0, maximum=1.0, step=0.01, value=0.6, interactive=True))
                ipa_rows.append(row)

        with gr.Row():
            components[key('add_flux1_ipadapter_button')] = gr.Button("✚ 添加 FLUX IP-Adapter")
            components[key('delete_flux1_ipadapter_button')] = gr.Button("➖ 移除 FLUX IP-Adapter", visible=False)
        components[key('flux1_ipadapter_count_state')] = gr.State(1)

        all_flux1_ipa_components_flat = images + weights + start_percents + end_percents
        components[key('all_flux1_ipadapter_components_flat')] = all_flux1_ipa_components_flat

    return components

def create_sd3_ipadapter_ui(prefix: str, max_units=MAX_IPADAPTERS):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("SD3 IP-Adapter 设置", open=False, visible=('sd3_ipadapter' in default_enabled_chains)) as accordion:
        components[key('sd3_ipadapter_accordion')] = accordion
        gr.Markdown("💡 由 [Slickytail/ComfyUI-InstantX-IPAdapter-SD3](https://github.com/Slickytail/ComfyUI-InstantX-IPAdapter-SD3) 节点处理。")

        ipa_rows, images, weights, start_percents, end_percents = [], [], [], [], []
        components.update({
            key('sd3_ipadapter_rows'): ipa_rows,
            key('sd3_ipadapter_images'): images,
            key('sd3_ipadapter_weights'): weights,
            key('sd3_ipadapter_start_percents'): start_percents,
            key('sd3_ipadapter_end_percents'): end_percents,
        })

        for i in range(max_units):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"IP-Adapter 参考图 {i+1}", type="pil", sources=["upload"], height=256))
                with gr.Column(scale=2):
                    weights.append(gr.Slider(label="权重", minimum=0.0, maximum=2.0, step=0.05, value=0.5, interactive=True))
                    with gr.Row():
                        start_percents.append(gr.Slider(label="开始位置", minimum=0.0, maximum=1.0, step=0.01, value=0.0, interactive=True))
                        end_percents.append(gr.Slider(label="结束位置", minimum=0.0, maximum=1.0, step=0.01, value=1.0, interactive=True))
                ipa_rows.append(row)

        with gr.Row():
            components[key('add_sd3_ipadapter_button')] = gr.Button("✚ 添加 SD3 IP-Adapter")
            components[key('delete_sd3_ipadapter_button')] = gr.Button("➖ 移除 SD3 IP-Adapter", visible=False)
        components[key('sd3_ipadapter_count_state')] = gr.State(1)

        all_sd3_ipa_components_flat = images + weights + start_percents + end_percents
        components[key('all_sd3_ipadapter_components_flat')] = all_sd3_ipa_components_flat

    return components

def create_style_ui(prefix: str):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("FLUX.1 风格参考", open=False, visible=('style' in default_enabled_chains)) as accordion:
        components[key('style_accordion')] = accordion

        style_rows, images, strengths = [], [], []
        components.update({
            key('style_rows'): style_rows,
            key('style_images'): images,
            key('style_strengths'): strengths
        })

        for i in range(5):
            with gr.Row(visible=(i < 1)) as row:
                with gr.Column(scale=1):
                    images.append(gr.Image(label=f"Style Image {i+1}", type="pil", sources=["upload"], height=256))
                with gr.Column(scale=2):
                    strengths.append(gr.Slider(label="强度", minimum=0.0, maximum=2.0, step=0.05, value=1.0, interactive=True))
                style_rows.append(row)

        with gr.Row():
            components[key('add_style_button')] = gr.Button("✚ 添加风格参考图")
            components[key('delete_style_button')] = gr.Button("➖ 移除风格参考图", visible=False)
        components[key('style_count_state')] = gr.State(1)

        all_style_components_flat = images + strengths
        components[key('all_style_components_flat')] = all_style_components_flat

    return components

def create_embedding_ui(prefix: str):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("Embedding 设置", open=False, visible=('embedding' in default_enabled_chains)) as accordion:
        components[key('embedding_accordion')] = accordion
        gr.Markdown("💡 Civitai 请填写 Version ID；Hugging Face 请填写 `repo_id/文件路径`。下载后在正面或负面提示词中写入 `embedding:文件名` 才会生效。")

        embedding_rows, sources, ids, files, upload_buttons = [], [], [], [], []
        components.update({
            key('embedding_rows'): embedding_rows,
            key('embeddings_sources'): sources,
            key('embeddings_ids'): ids,
            key('embeddings_files'): files,
            key('embeddings_uploads'): upload_buttons
        })

        for i in range(MAX_EMBEDDINGS):
            with gr.Row(visible=(i < 1)) as row:
                sources.append(gr.Dropdown(label=f"Embedding 来源 {i+1}", choices=LORA_SOURCE_CHOICES, value="Civitai", scale=1, interactive=True))
                ids.append(gr.Textbox(label="Civitai Version ID / HF 文件 / 已上传文件", scale=3, interactive=True, type="text"))
                upload_btn = gr.UploadButton("上传", file_types=[".safetensors"], scale=1)
                files.append(gr.State(None))
                upload_buttons.append(upload_btn)
                embedding_rows.append(row)

        with gr.Row():
            components[key('add_embedding_button')] = gr.Button("✚ 添加 Embedding")
            components[key('delete_embedding_button')] = gr.Button("➖ 移除 Embedding", visible=False)
        components[key('embedding_count_state')] = gr.State(1)

        all_embedding_components_flat = []
        for i in range(MAX_EMBEDDINGS):
            all_embedding_components_flat.extend([sources[i], ids[i], files[i]])
        components[key('all_embedding_components_flat')] = all_embedding_components_flat

    return components

def create_conditioning_ui(prefix: str):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("区域提示设置", open=False, visible=('conditioning' in default_enabled_chains)) as accordion:
        components[key('conditioning_accordion')] = accordion
        gr.Markdown("💡 为矩形区域设置单独提示词；坐标 X、Y 从画面左上角开始计算。")

        cond_rows, prompts, widths, heights, xs, ys, strengths = [], [], [], [], [], [], []
        components.update({
            key('conditioning_rows'): cond_rows,
            key('conditioning_prompts'): prompts,
            key('conditioning_widths'): widths,
            key('conditioning_heights'): heights,
            key('conditioning_xs'): xs,
            key('conditioning_ys'): ys,
            key('conditioning_strengths'): strengths
        })

        for i in range(MAX_CONDITIONINGS):
            with gr.Column(visible=(i < 1)) as row_wrapper:
                prompts.append(gr.Textbox(label=f"区域提示词 {i+1}", lines=2, interactive=True))
                with gr.Row():
                    xs.append(gr.Number(label="X", value=0, interactive=True, step=8, scale=1))
                    ys.append(gr.Number(label="Y", value=0, interactive=True, step=8, scale=1))
                    widths.append(gr.Number(label="宽度", value=512, interactive=True, step=8, scale=1))
                    heights.append(gr.Number(label="高度", value=512, interactive=True, step=8, scale=1))
                    strengths.append(gr.Slider(label="强度", minimum=0.1, maximum=2.0, step=0.05, value=1.0, interactive=True, scale=2))
                cond_rows.append(row_wrapper)

        with gr.Row():
            components[key('add_conditioning_button')] = gr.Button("✚ 添加区域")
            components[key('delete_conditioning_button')] = gr.Button("➖ 移除区域", visible=False)
        components[key('conditioning_count_state')] = gr.State(1)

        all_cond_components_flat = prompts + widths + heights + xs + ys + strengths
        components[key('all_conditioning_components_flat')] = all_cond_components_flat

    return components

def on_vae_upload(file_obj):
    if not file_obj:
        return gr.update(), gr.update(), None

    hashed_filename = save_uploaded_file_with_hash(file_obj, VAE_DIR)
    return hashed_filename, "File", file_obj

def create_vae_override_ui(prefix: str):
    components = {}
    key = lambda name: f"{name}_{prefix}"
    source_choices = ["None"] + LORA_SOURCE_CHOICES

    with gr.Accordion("替换 VAE", open=False, visible=('vae' in default_enabled_chains)) as vae_accordion:
        components[key('vae_accordion')] = vae_accordion
        gr.Markdown("💡 Civitai 请填写 Version ID；Hugging Face 请填写 `repo_id/文件路径`。")
        with gr.Row():
            components[key('vae_source')] = gr.Dropdown(
                label="VAE 来源",
                choices=source_choices,
                value="None",
                scale=1,
                interactive=True
            )
            components[key('vae_id')] = gr.Textbox(
                label="Civitai Version ID / HF 文件 / 已上传文件",
                scale=3,
                interactive=True,
                type="text"
            )
            upload_btn = gr.UploadButton(
                "上传",
                file_types=[".safetensors"],
                scale=1
            )
            components[key('vae_upload_button')] = upload_btn
            components[key('vae_file')] = gr.State(None)

            upload_btn.upload(
                fn=on_vae_upload,
                inputs=[upload_btn],
                outputs=[components[key('vae_id')], components[key('vae_source')], components[key('vae_file')]]
            )

    return components

def create_reference_latent_ui(prefix: str, max_units=10):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("多图参考编辑", open=False, visible=('reference_latent' in default_enabled_chains)) as ref_accordion:
        components[key('reference_latent_accordion')] = ref_accordion
        gr.Markdown("💡 一张参考图用于图片编辑，多张参考图用于图片组合；仅兼容支持多模态输入的模型。")

        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"参考图 {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)

        components[key('reference_latent_rows')] = ref_image_groups
        components[key('reference_latent_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_reference_latent_button')] = gr.Button("✚ 添加参考图")
            components[key('delete_reference_latent_button')] = gr.Button("➖ 移除参考图", visible=False)
        components[key('reference_latent_count_state')] = gr.State(1)

        components[key('all_reference_latent_components_flat')] = ref_image_inputs

    return components

def create_hidream_o1_reference_ui(prefix: str, max_units=10):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("HiDream-O1 参考图编辑", open=False, visible=('hidream_o1_reference' in default_enabled_chains)) as ref_accordion:
        components[key('hidream_o1_reference_accordion')] = ref_accordion
        gr.Markdown("💡 建议使用 **HiDream-O1-Image-Dev** 并设置约 **4MP** 分辨率；一张图用于编辑，多张图用于组合。")

        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"参考图 {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)

        components[key('hidream_o1_reference_rows')] = ref_image_groups
        components[key('hidream_o1_reference_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_hidream_o1_reference_button')] = gr.Button("✚ 添加参考图")
            components[key('delete_hidream_o1_reference_button')] = gr.Button("➖ 移除参考图", visible=False)
        components[key('hidream_o1_reference_count_state')] = gr.State(1)

        components[key('all_hidream_o1_reference_components_flat')] = ref_image_inputs

    return components

def create_joyai_reference_ui(prefix: str, max_units=2):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("JoyAI 参考图编辑", open=False, visible=('joyai_image' in default_enabled_chains)) as ref_accordion:
        components[key('joyai_reference_accordion')] = ref_accordion
        gr.Markdown("💡 一张图建议使用 JoyAI-Image-Edit；多图组合建议使用 JoyAI-Image-Edit-Plus，复杂任务会自动申请更长 GPU 时长。")

        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"参考图 {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)

        components[key('joyai_reference_rows')] = ref_image_groups
        components[key('joyai_reference_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_joyai_reference_button')] = gr.Button("✚ 添加参考图")
            components[key('delete_joyai_reference_button')] = gr.Button("➖ 移除参考图", visible=False)
        components[key('joyai_reference_count_state')] = gr.State(1)

        components[key('all_joyai_reference_components_flat')] = ref_image_inputs

    return components

def create_reference_image_ui(prefix: str, max_units=10):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("Mage-Flow 参考图编辑", open=False, visible=('reference_image' in default_enabled_chains)) as ref_accordion:
        components[key('reference_image_accordion')] = ref_accordion
        gr.Markdown("💡 建议使用 Mage-Flow-Edit-Turbo 或 Mage-Flow-Edit；一张图用于编辑，多张图用于组合。")

        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"参考图 {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)

        components[key('reference_image_rows')] = ref_image_groups
        components[key('reference_image_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_reference_image_button')] = gr.Button("✚ 添加参考图")
            components[key('delete_reference_image_button')] = gr.Button("➖ 移除参考图", visible=False)
        components[key('reference_image_count_state')] = gr.State(1)

        components[key('all_reference_image_components_flat')] = ref_image_inputs

    return components

def create_pid_ui(prefix: str):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("PiD 4× 解码", open=False, visible=('pid' in default_enabled_chains)) as pid_accordion:
        components[key('pid_accordion')] = pid_accordion
        gr.Markdown("💡 使用 PiD（Pixel Diffusion Decoder）替代 VAE 解码器进行 4× 解码。")
        with gr.Row():
            components[key('pid_settings')] = gr.Dropdown(
                label="PiD 模式",
                choices=["OFF", "ON"],
                value="OFF",
                interactive=True
            )

    return components

def create_krea2_identity_edit_ui(prefix: str, max_units=2):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("Krea2 身份参考编辑", open=False, visible=('krea2_identity_edit' in default_enabled_chains)) as ref_accordion:
        components[key('krea2_identity_edit_accordion')] = ref_accordion
        gr.Markdown("💡 建议使用 Krea-2-Turbo；一张参考图用于身份编辑，多张图用于组合。Krea-2-Raw 的复杂任务会自动申请更长 GPU 时长。")

        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"参考图 {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)

        components[key('krea2_identity_edit_rows')] = ref_image_groups
        components[key('krea2_identity_edit_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_krea2_identity_edit_button')] = gr.Button("✚ 添加参考图")
            components[key('delete_krea2_identity_edit_button')] = gr.Button("➖ 移除参考图", visible=False)
        components[key('krea2_identity_edit_count_state')] = gr.State(1)

        components[key('all_krea2_identity_edit_components_flat')] = ref_image_inputs

    return components

def create_qwen_image_edit_ui(prefix: str, max_units=3):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("Qwen-Image 图片编辑", open=False, visible=('qwen_image_edit' in default_enabled_chains)) as ref_accordion:
        components[key('qwen_image_edit_accordion')] = ref_accordion
        gr.Markdown("💡 建议使用 Qwen-Image-Edit-2511-Lightning；一张参考图用于编辑，多张图用于组合。")

        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"参考图 {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)

        components[key('qwen_image_edit_rows')] = ref_image_groups
        components[key('qwen_image_edit_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_qwen_image_edit_button')] = gr.Button("✚ 添加参考图")
            components[key('delete_qwen_image_edit_button')] = gr.Button("➖ 移除参考图", visible=False)
        components[key('qwen_image_edit_count_state')] = gr.State(1)

        components[key('all_qwen_image_edit_components_flat')] = ref_image_inputs

    return components

def create_krea2_reference_edit_ui(prefix: str, max_units=3):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("Krea2 风格参考编辑", open=False, visible=('krea2_style_reference' in default_enabled_chains)) as ref_accordion:
        components[key('krea2_reference_edit_accordion')] = ref_accordion
        gr.Markdown("💡 建议使用 Krea-2-Turbo；添加参考图后可按其风格编辑画面。")

        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"参考图 {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)

        components[key('krea2_reference_edit_rows')] = ref_image_groups
        components[key('krea2_reference_edit_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_krea2_reference_edit_button')] = gr.Button("✚ 添加参考图")
            components[key('delete_krea2_reference_edit_button')] = gr.Button("➖ 移除参考图", visible=False)
        components[key('krea2_reference_edit_count_state')] = gr.State(1)

        components[key('all_krea2_reference_edit_components_flat')] = ref_image_inputs

    return components

def create_boogu_edit_ui(prefix: str, max_units=10):
    components = {}
    key = lambda name: f"{name}_{prefix}"

    with gr.Accordion("Boogu-Image 图片编辑", open=False, visible=('boogu_image_edit' in default_enabled_chains)) as ref_accordion:
        components[key('boogu_edit_accordion')] = ref_accordion
        gr.Markdown("💡 建议使用 Boogu-Image-Edit-Turbo 或 Boogu-Image-Edit；一张图用于编辑，多张图用于组合，复杂任务会自动申请更长 GPU 时长。")

        ref_image_groups = []
        ref_image_inputs = []
        with gr.Row():
            for i in range(max_units):
                with gr.Column(visible=(i < 1), min_width=160) as img_col:
                    img_comp = gr.Image(type="pil", label=f"参考图 {i+1}", sources=["upload"], height=150)
                    ref_image_groups.append(img_col)
                    ref_image_inputs.append(img_comp)

        components[key('boogu_edit_rows')] = ref_image_groups
        components[key('boogu_edit_images')] = ref_image_inputs

        with gr.Row():
            components[key('add_boogu_edit_button')] = gr.Button("✚ 添加参考图")
            components[key('delete_boogu_edit_button')] = gr.Button("➖ 移除参考图", visible=False)
        components[key('boogu_edit_count_state')] = gr.State(1)

        components[key('all_boogu_edit_components_flat')] = ref_image_inputs

    return components
