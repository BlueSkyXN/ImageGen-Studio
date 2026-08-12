"""One shared, task-switchable UI instead of five duplicated forms."""

from __future__ import annotations

import gradio as gr

from comfy_integration.nodes import SAMPLER_CHOICES, SCHEDULER_CHOICES
from core.execution_plan import RUN_MODE_CHOICES
from core.runtime_config import CONFIG
from core.settings import (
    ARCHITECTURES_CONFIG,
    MODEL_DEFAULTS_CONFIG,
    MODEL_MAP_CHECKPOINT,
    MODEL_TYPE_MAP,
    RESOLUTION_MAP,
)
from ui.guidance import QUICK_PRESETS, model_hint, task_help
from .ui_components import (
    create_anima_controlnet_lllite_ui,
    create_boogu_edit_ui,
    create_category_filter_ui,
    create_conditioning_ui,
    create_controlnet_ui,
    create_diffsynth_controlnet_ui,
    create_embedding_ui,
    create_flux1_ipadapter_ui,
    create_hidream_o1_reference_ui,
    create_ipadapter_ui,
    create_joyai_reference_ui,
    create_krea2_controlnet_ui,
    create_krea2_identity_edit_ui,
    create_krea2_reference_edit_ui,
    create_lora_settings_ui,
    create_model_architecture_filter_ui,
    create_pid_ui,
    create_qwen_image_edit_ui,
    create_reference_image_ui,
    create_reference_latent_ui,
    create_sd3_ipadapter_ui,
    create_style_ui,
    create_vae_override_ui,
)


TASK_CHOICES = [
    ("✨ 文生图", "txt2img"),
    ("🖼️ 图生图", "img2img"),
    ("🖌️ 局部重绘", "inpaint"),
    ("↔️ 扩图", "outpaint"),
    ("🔎 高清修复", "hires_fix"),
]

PROMPT_EXAMPLES = [
    ["一位银发少女站在雨夜车站，半身构图，电影感侧光，细腻插画，高质量"],
    ["现代东方客厅，浅木色与米白配色，午后自然光，广角室内摄影"],
    ["未来城市夜景，湿润街道倒影，霓虹灯，低机位，强烈纵深，电影概念设计"],
]


def create_ui() -> dict:
    prefix = "studio"
    components: dict = {"_task_prefixes": [(prefix, None)]}
    default_values = MODEL_DEFAULTS_CONFIG.get("Default", {})
    default_model = next(iter(MODEL_MAP_CHECKPOINT), None)
    default_architecture = MODEL_TYPE_MAP.get(default_model, "SDXL")
    default_architecture_info = ARCHITECTURES_CONFIG.get("architectures", {}).get(
        default_architecture, {}
    )
    default_resolution_key = default_architecture_info.get("model_type", "sdxl")
    default_resolution_map = RESOLUTION_MAP.get(
        default_resolution_key, RESOLUTION_MAP.get("sdxl", {})
    )
    default_aspect = next(iter(default_resolution_map), "1:1 (Square)")
    default_width, default_height = default_resolution_map.get(default_aspect, (1024, 1024))

    components[f"task_type_{prefix}"] = gr.Radio(
        choices=TASK_CHOICES,
        value="txt2img",
        label="1. 选择任务",
        elem_id="task-switcher",
    )
    components[f"task_help_{prefix}"] = gr.Markdown(
        task_help("txt2img"), elem_classes="task-help"
    )

    with gr.Row(equal_height=False, elem_id="workspace-row"):
        with gr.Column(scale=7, min_width=360):
            with gr.Accordion("2. 选择模型", open=True):
                components[f"quick_preset_{prefix}"] = gr.Dropdown(
                    choices=QUICK_PRESETS,
                    value="__manual__",
                    label="快速方案",
                    info="只是帮你定位模型；仍可继续调整所有参数。",
                    interactive=True,
                )
                components.update(create_model_architecture_filter_ui(prefix))
                with gr.Row():
                    components.update(create_category_filter_ui(prefix))
                    components[f"base_model_{prefix}"] = gr.Dropdown(
                        label="具体模型",
                        choices=list(MODEL_MAP_CHECKPOINT.keys()),
                        value=default_model,
                        filterable=True,
                        allow_custom_value=False,
                        scale=3,
                        interactive=True,
                    )
                components[f"model_hint_{prefix}"] = gr.Markdown(
                    model_hint(default_model) if default_model else "暂无可用模型。",
                    elem_classes="model-hint",
                )

            with gr.Accordion("批量、多图与模型 PK（可选）", open=False):
                components[f"run_mode_{prefix}"] = gr.Dropdown(
                    choices=RUN_MODE_CHOICES,
                    value="single",
                    label="运行方式",
                    info="所有模式都顺序执行，不会同时把多个大模型塞进显存。",
                )
                components[f"pk_models_{prefix}"] = gr.Dropdown(
                    choices=list(MODEL_MAP_CHECKPOINT.keys()),
                    value=[],
                    multiselect=True,
                    max_choices=max(1, CONFIG.max_pk_models - 1),
                    label="额外对比模型",
                    info=f"当前模型会自动加入；最多合计 {CONFIG.max_pk_models} 个。",
                    filterable=True,
                    visible=False,
                )
                components[f"batch_images_{prefix}"] = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    type="filepath",
                    allow_reordering=True,
                    label="批量输入 / 多图参考",
                    visible=False,
                )
                with gr.Row():
                    components[f"reference_role_{prefix}"] = gr.Dropdown(
                        choices=[
                            ("自动匹配模型能力", "auto"),
                            ("人物 / 身份参考", "identity"),
                            ("风格参考", "style"),
                        ],
                        value="auto",
                        label="多图融合用途",
                        visible=False,
                    )
                    components[f"pk_model_defaults_{prefix}"] = gr.Checkbox(
                        value=True,
                        label="PK 使用各模型推荐采样参数",
                        info="Prompt、尺寸与种子保持一致；步数和 CFG 按模型推荐值。",
                        visible=False,
                    )
                gr.Markdown(
                    "**怎么选：** 多图独立是 A→A′、B→B′；多图×多模型会执行图片和模型的组合；"
                    "多图融合是把整组图片同时交给 Qwen/Mage-Flow/JoyAI/Boogu/Krea/Flux 等兼容模型。"
                )

            with gr.Group(elem_classes="prompt-card"):
                components[f"prompt_{prefix}"] = gr.Textbox(
                    label="3. 描述你想要的画面",
                    placeholder="主体 + 动作 + 场景 + 构图 + 风格 + 光线",
                    lines=4,
                    max_lines=8,
                    value=default_values.get("positive_prompt", ""),
                )
                with gr.Accordion("负面提示词（可选）", open=False):
                    components[f"neg_prompt_{prefix}"] = gr.Textbox(
                        label="不希望出现的内容",
                        placeholder="例如：低质量、模糊、多余手指、水印",
                        lines=2,
                        value=default_values.get("negative_prompt", ""),
                    )
                gr.Examples(
                    examples=PROMPT_EXAMPLES,
                    inputs=[components[f"prompt_{prefix}"]],
                    label="点一下填入示例",
                )

            with gr.Group(visible=False) as source_panel:
                components[f"source_image_{prefix}"] = gr.Image(
                    type="pil",
                    label="源图片",
                    sources=["upload", "clipboard"],
                    height=300,
                )
            components[f"source_panel_{prefix}"] = source_panel

            with gr.Group(visible=False) as inpaint_panel:
                gr.Markdown("涂抹需要替换的区域；未涂抹部分会尽量保留。")
                components[f"inpaint_image_dict_{prefix}"] = gr.ImageEditor(
                    type="pil",
                    label="图片与蒙版",
                    sources=["upload", "clipboard"],
                    height=420,
                )
                with gr.Row():
                    components[f"inpaint_denoise_{prefix}"] = gr.Slider(
                        0.0, 1.0, value=1.0, step=0.05, label="局部重绘幅度"
                    )
                    components[f"grow_mask_by_{prefix}"] = gr.Slider(
                        0, 64, value=6, step=1, label="蒙版外扩像素"
                    )
            components[f"inpaint_panel_{prefix}"] = inpaint_panel

            with gr.Group(visible=False) as img2img_panel:
                components[f"img2img_denoise_{prefix}"] = gr.Slider(
                    0.0,
                    1.0,
                    value=0.7,
                    step=0.01,
                    label="重绘幅度",
                    info="0 更接近原图，1 改动最大。",
                )
            components[f"img2img_panel_{prefix}"] = img2img_panel

            with gr.Group(visible=False) as outpaint_panel:
                with gr.Row():
                    components[f"left_{prefix}"] = gr.Slider(0, 512, 64, step=64, label="向左扩展")
                    components[f"right_{prefix}"] = gr.Slider(0, 512, 64, step=64, label="向右扩展")
                with gr.Row():
                    components[f"top_{prefix}"] = gr.Slider(0, 512, 64, step=64, label="向上扩展")
                    components[f"bottom_{prefix}"] = gr.Slider(0, 512, 64, step=64, label="向下扩展")
                components[f"feathering_{prefix}"] = gr.Slider(
                    0, 100, 10, step=1, label="接缝羽化 / 蒙版外扩"
                )
            components[f"outpaint_panel_{prefix}"] = outpaint_panel

            with gr.Group(visible=False) as hires_panel:
                with gr.Row():
                    components[f"hires_upscaler_{prefix}"] = gr.Dropdown(
                        choices=["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],
                        value="nearest-exact",
                        label="放大算法",
                    )
                    components[f"hires_scale_by_{prefix}"] = gr.Slider(
                        1.0, 4.0, 1.5, step=0.1, label="放大倍数"
                    )
                components[f"hires_denoise_{prefix}"] = gr.Slider(
                    0.0, 1.0, 0.55, step=0.01, label="细节重绘幅度"
                )
            components[f"hires_panel_{prefix}"] = hires_panel

            with gr.Group() as size_panel:
                with gr.Row():
                    components[f"aspect_ratio_{prefix}"] = gr.Dropdown(
                        label="画幅",
                        choices=list(default_resolution_map) or [default_aspect],
                        value=default_aspect,
                        interactive=True,
                        scale=2,
                    )
                    components[f"width_{prefix}"] = gr.Number(
                        label="宽度", value=default_width, precision=0, interactive=True
                    )
                    components[f"height_{prefix}"] = gr.Number(
                        label="高度", value=default_height, precision=0, interactive=True
                    )
            components[f"size_panel_{prefix}"] = size_panel

            with gr.Accordion("高级采样与运行参数", open=False):
                with gr.Row():
                    components[f"sampler_{prefix}"] = gr.Dropdown(
                        choices=SAMPLER_CHOICES,
                        value=default_values.get("sampler_name", "euler"),
                        label="采样器",
                    )
                    components[f"scheduler_{prefix}"] = gr.Dropdown(
                        choices=SCHEDULER_CHOICES,
                        value=default_values.get("scheduler", "simple"),
                        label="调度器",
                    )
                with gr.Row():
                    components[f"steps_{prefix}"] = gr.Slider(
                        1, 100, default_values.get("steps", 8), step=1, label="采样步数"
                    )
                    components[f"cfg_{prefix}"] = gr.Slider(
                        1.0, 20.0, default_values.get("cfg", 1.0), step=0.1, label="CFG"
                    )
                with gr.Row():
                    components[f"seed_{prefix}"] = gr.Number(
                        label="随机种子（-1 为随机）", value=-1, precision=0
                    )
                    components[f"batch_size_{prefix}"] = gr.Slider(
                        1,
                        CONFIG.max_batch_size,
                        1,
                        step=1,
                        label="单次生成数量",
                        info="数量越大，显存与 ZeroGPU 配额消耗越高。",
                    )
                with gr.Row():
                    components[f"clip_skip_{prefix}"] = gr.Slider(
                        1, 2, 1, step=1, label="Clip Skip", visible=False
                    )
                    components[f"guidance_{prefix}"] = gr.Slider(
                        1.0, 10.0, 3.5, step=0.1, label="FLUX Guidance", visible=False
                    )
                    components[f"zero_gpu_{prefix}"] = gr.Dropdown(
                        choices=[("自动估算", 0), ("60 秒", 60), ("90 秒", 90), ("120 秒", 120)],
                        value=0,
                        label="ZeroGPU 运行时长",
                        info="通常保持自动；复杂高分辨率任务可手动提高。",
                    )
                with gr.Row():
                    components[f"auto_model_params_{prefix}"] = gr.Checkbox(
                        value=True,
                        label="切模型时自动应用推荐采样参数",
                        info="关闭后会保留你手调的步数、CFG、采样器和调度器。",
                        scale=3,
                    )
                    components[f"reset_model_params_{prefix}"] = gr.Button(
                        "恢复当前模型推荐值", size="sm", scale=1
                    )

        with gr.Column(scale=5, min_width=340, elem_id="result-column"):
            with gr.Row():
                components[f"run_{prefix}"] = gr.Button(
                    "开始生成", variant="primary", size="lg", scale=3
                )
                components[f"cancel_{prefix}"] = gr.Button(
                    "取消排队", variant="stop", size="lg", scale=1
                )
            components[f"result_{prefix}"] = gr.Gallery(
                label="生成结果",
                show_label=True,
                columns=2,
                object_fit="contain",
                height=720,
            )
            components[f"run_summary_{prefix}"] = gr.Markdown(
                "普通模式会在这里显示执行结果；PK/批量模式会逐项列出成功与失败。"
            )
            components[f"clear_result_{prefix}"] = gr.Button("清空结果", size="sm")
            gr.Markdown(
                "生成结果包含 Prompt、参数和完整 ComfyUI 工作流元数据，下载 PNG 后可继续复现。",
                elem_classes="result-note",
            )

    gr.Markdown("## 当前模型可用的扩展能力", elem_classes="advanced-title")
    gr.Markdown("下方只显示当前模型兼容的能力；切换模型后会自动更新。")
    components.update(create_lora_settings_ui(prefix))
    components.update(create_controlnet_ui(prefix))
    components.update(create_anima_controlnet_lllite_ui(prefix))
    components.update(create_diffsynth_controlnet_ui(prefix))
    components.update(create_krea2_controlnet_ui(prefix))
    components.update(create_ipadapter_ui(prefix))
    components.update(create_flux1_ipadapter_ui(prefix))
    components.update(create_sd3_ipadapter_ui(prefix))
    components.update(create_embedding_ui(prefix))
    components.update(create_style_ui(prefix))
    components.update(create_conditioning_ui(prefix))
    components.update(create_reference_latent_ui(prefix))
    components.update(create_hidream_o1_reference_ui(prefix))
    components.update(create_joyai_reference_ui(prefix))
    components.update(create_krea2_identity_edit_ui(prefix))
    components.update(create_krea2_reference_edit_ui(prefix))
    components.update(create_qwen_image_edit_ui(prefix))
    components.update(create_boogu_edit_ui(prefix))
    components.update(create_reference_image_ui(prefix))
    components.update(create_vae_override_ui(prefix))
    components.update(create_pid_ui(prefix))

    return components
