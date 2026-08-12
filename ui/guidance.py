"""Chinese-first task and model guidance generated from the existing registry."""

from __future__ import annotations

from core.settings import (
    ARCHITECTURES_CONFIG,
    FEATURES_CONFIG,
    MODEL_DEFAULTS_CONFIG,
    MODEL_MAP_CHECKPOINT,
    MODEL_TYPE_MAP,
)
from imagegen_utils.app_utils import get_model_generation_defaults


TASK_HELP = {
    "txt2img": "**文生图**：选择模型，写清主体、动作、构图、风格和光线，再点击生成。",
    "img2img": "**图生图**：上传原图；重绘幅度越低越接近原图，越高改动越大。",
    "inpaint": "**局部重绘**：上传图片并涂抹需要修改的区域；提示词只描述希望出现的新内容。",
    "outpaint": "**扩图**：上传原图，设置向四周扩展的像素；提示词描述画面外应延续的内容。",
    "hires_fix": "**高清修复**：上传图片，先用 1.5× 和 0.35–0.55 重绘幅度试跑，避免细节漂移。",
}


FEATURE_NAMES = {
    "lora": "LoRA",
    "controlnet": "ControlNet",
    "anima_controlnet_lllite": "Anima ControlNet",
    "diffsynth_controlnet": "DiffSynth ControlNet",
    "krea2_controlnet": "Krea2 ControlNet",
    "ipadapter": "IP-Adapter",
    "flux1_ipadapter": "FLUX IP-Adapter",
    "sd3_ipadapter": "SD3 IP-Adapter",
    "style": "风格参考",
    "embedding": "Embedding",
    "conditioning": "区域提示",
    "reference_latent": "多图编辑",
    "hidream_o1_reference": "HiDream 参考图",
    "joyai_image": "JoyAI 参考图",
    "krea2_identity_edit": "Krea2 身份参考",
    "krea2_style_reference": "Krea2 风格参考",
    "qwen_image_edit": "Qwen 图片编辑",
    "boogu_image_edit": "Boogu 图片编辑",
    "reference_image": "Mage-Flow 参考图",
    "vae": "VAE 替换",
    "pid": "PiD 解码",
}


QUICK_PRESETS = [
    ("手动选择", "__manual__"),
    ("快速通用 · Krea-2-Turbo", "Krea-2-Turbo"),
    ("中文与文字 · Qwen-Image Lightning", "lightx2v/Qwen-Image-2512-Lightning"),
    ("动漫角色 · Anima Turbo", "circlestone-labs/Anima-Turbo-v1.0"),
    ("快速编辑 · Qwen-Image-Edit Lightning", "lightx2v/Qwen-Image-Edit-2511-Lightning"),
    ("SDXL 动漫 · Animagine XL 4.0", "CagliostroLab/Animagine XL 4.0"),
]


def task_help(task_type: str) -> str:
    return TASK_HELP.get(task_type, TASK_HELP["txt2img"])


def model_hint(model_name: str) -> str:
    info = MODEL_MAP_CHECKPOINT.get(model_name)
    if not info:
        return "请选择配置中已有的模型。"

    architecture = MODEL_TYPE_MAP.get(model_name, info[2])
    architecture_config = ARCHITECTURES_CONFIG.get("architectures", {}).get(architecture, {})
    workflow_type = architecture_config.get(
        "model_type", architecture.lower().replace(" ", "").replace(".", "")
    )
    defaults = get_model_generation_defaults(
        model_name, workflow_type, MODEL_DEFAULTS_CONFIG
    )
    features = FEATURES_CONFIG.get(workflow_type, {}).get("enabled_chains", [])
    feature_text = "、".join(FEATURE_NAMES.get(item, item) for item in features[:6]) or "基础生成"
    if len(features) > 6:
        feature_text += f" 等 {len(features)} 项"

    tag_first_architectures = {"SDXL", "SD1.5", "Anima", "NewBie-Image"}
    if architecture in tag_first_architectures:
        language_tip = "更适合英文标签式 Prompt；中文意图可保留，但建议补充英文关键词。"
    elif architecture in {"Qwen-Image", "Z-Image", "HunyuanImage", "ERNIE-Image"}:
        language_tip = "可优先直接使用中文自然语言；涉及文字时把具体文字用引号写出。"
    else:
        language_tip = "可先用中文自然语言；若构图不稳定，再补充简短英文关键词。"

    edit_tip = ""
    if "edit" in model_name.lower():
        edit_tip = " 这是编辑向模型，配合下方自动出现的参考图能力使用。"

    return (
        f"**{architecture} · 推荐参数**：{defaults.get('steps', 20)} 步，"
        f"CFG {defaults.get('cfg', 1.0)}，{defaults.get('sampler_name', 'euler')} / "
        f"{defaults.get('scheduler', 'simple')}  \n"
        f"**Prompt 建议**：{language_tip}{edit_tip}  \n"
        f"**当前可用扩展**：{feature_text}"
    )


def recommended_params(model_name: str) -> tuple:
    """Return the registry defaults shown by the one-click reset control."""
    info = MODEL_MAP_CHECKPOINT.get(model_name)
    if not info:
        return 20, 1.0, "euler", "simple"
    architecture = MODEL_TYPE_MAP.get(model_name, info[2])
    architecture_config = ARCHITECTURES_CONFIG.get("architectures", {}).get(
        architecture, {}
    )
    workflow_type = architecture_config.get(
        "model_type", architecture.lower().replace(" ", "").replace(".", "")
    )
    defaults = get_model_generation_defaults(
        model_name, workflow_type, MODEL_DEFAULTS_CONFIG
    )
    return (
        defaults.get("steps", 20),
        defaults.get("cfg", 1.0),
        defaults.get("sampler_name", "euler"),
        defaults.get("scheduler", "simple"),
    )
