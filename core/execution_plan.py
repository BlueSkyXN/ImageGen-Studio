"""Build small sequential generation plans for PK and multi-image workflows.

The in-process ComfyUI runtime is intentionally single-model-at-a-time.  This
module therefore expands comparisons into bounded sequential runs instead of
trying to keep several checkpoints resident on one ZeroGPU allocation.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

from PIL import Image

from core.model_capabilities import supports_chain_for_model
from core.runtime_config import CONFIG
from core.settings import (
    ARCHITECTURES_CONFIG,
    FEATURES_CONFIG,
    MODEL_DEFAULTS_CONFIG,
    MODEL_MAP_CHECKPOINT,
    MODEL_TYPE_MAP,
)

MODE_SINGLE = "single"
MODE_MODEL_PK = "model_pk"
MODE_MULTI_INDEPENDENT = "multi_independent"
MODE_MULTI_MODEL_GRID = "multi_model_grid"
MODE_MULTI_REFERENCE = "multi_reference"

RUN_MODE_CHOICES = [
    ("普通生成", MODE_SINGLE),
    ("模型 PK：同一输入对比多个模型", MODE_MODEL_PK),
    ("多图独立：每张图分别处理", MODE_MULTI_INDEPENDENT),
    ("多图 × 多模型：组合对比", MODE_MULTI_MODEL_GRID),
    ("多图融合：多张参考图生成一组结果", MODE_MULTI_REFERENCE),
]

INDEPENDENT_IMAGE_TASK_KEYS = {
    "img2img": "img2img_image",
    "outpaint": "outpaint_image",
    "hires_fix": "hires_image",
}

COMPARISON_CHAIN_INPUT_KEYS = (
    "lora_data",
    "controlnet_data",
    "anima_controlnet_lllite_data",
    "diffsynth_controlnet_data",
    "krea2_controlnet_data",
    "ipadapter_data",
    "sd3_ipadapter_chain",
    "flux1_ipadapter_data",
    "style_data",
    "embedding_data",
    "conditioning_data",
    "reference_latent_data",
    "hidream_o1_reference_data",
    "joyai_reference_data",
    "krea2_identity_edit_data",
    "krea2_reference_edit_data",
    "qwen_image_edit_data",
    "boogu_edit_data",
    "reference_image_data",
)

# (chain name, pipeline input key, maximum images supported by its injector)
REFERENCE_CHAIN_SPECS = {
    "qwen_image_edit": ("qwen_image_edit_data", 3),
    "joyai_image": ("joyai_reference_data", 2),
    "boogu_image_edit": ("boogu_edit_data", 10),
    "reference_image": ("reference_image_data", 10),
    "reference_latent": ("reference_latent_data", 10),
    "hidream_o1_reference": ("hidream_o1_reference_data", 10),
    "krea2_identity_edit": ("krea2_identity_edit_data", 2),
    "krea2_style_reference": ("krea2_reference_edit_data", 3),
}


class ExecutionPlanError(ValueError):
    pass


@dataclass(frozen=True)
class PlannedGeneration:
    inputs: dict[str, Any]
    caption: str


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _caption(label: str, values: dict[str, Any]) -> str:
    seed = values.get("seed", "-")
    steps = values.get("num_inference_steps", "-")
    cfg = values.get("guidance_scale", "-")
    return f"{label} · Seed {seed} · {steps} 步 · CFG {cfg}"


def _workflow_type(model_name: str) -> str:
    architecture = MODEL_TYPE_MAP.get(model_name, "SDXL")
    architecture_info = ARCHITECTURES_CONFIG.get("architectures", {}).get(
        architecture, {}
    )
    return architecture_info.get(
        "model_type", architecture.lower().replace(" ", "").replace(".", "")
    )


def _model_defaults(model_name: str) -> dict[str, Any]:
    workflow_type = _workflow_type(model_name)
    defaults = {
        "steps": 25,
        "cfg": 7.0,
        "sampler_name": "euler",
        "scheduler": "simple",
    }
    defaults.update(MODEL_DEFAULTS_CONFIG.get("Default", {}))
    type_key = next(
        (
            key
            for key in MODEL_DEFAULTS_CONFIG
            if key.lower().replace(" ", "-").replace(".", "")
            == workflow_type.lower()
        ),
        None,
    )
    if type_key:
        section = MODEL_DEFAULTS_CONFIG.get(type_key, {})
        defaults.update(section.get("_defaults", {}))
        defaults.update(section.get(model_name, {}))
    return defaults


def _for_model(
    base_inputs: dict[str, Any], model_name: str, use_model_defaults: bool
) -> dict[str, Any]:
    # Pipeline processing replaces and occasionally mutates chain containers.
    # Copy those containers while sharing immutable/PIL payloads.
    values = {
        key: list(value)
        if isinstance(value, list)
        else dict(value)
        if isinstance(value, dict)
        else value
        for key, value in base_inputs.items()
    }
    values["model_display_name"] = model_name
    if use_model_defaults:
        defaults = _model_defaults(model_name)
        values.update(
            {
                "num_inference_steps": defaults.get("steps", 20),
                "guidance_scale": defaults.get("cfg", 1.0),
                "sampler": defaults.get("sampler_name", "euler"),
                "scheduler": defaults.get("scheduler", "simple"),
            }
        )
    return values


def _make_fair_comparison(values: dict[str, Any]) -> None:
    """Keep V1 comparisons to capabilities shared by every base checkpoint."""

    for key in COMPARISON_CHAIN_INPUT_KEYS:
        values[key] = []
    values["pid_settings"] = "OFF"
    values["vae_source"] = None
    values["vae_id"] = None
    values["vae_file"] = None


def load_uploaded_images(uploaded_files: Sequence[Any] | None) -> list[Image.Image]:
    """Materialize Gradio File values as detached PIL images."""

    images: list[Image.Image] = []
    total_megapixels = 0.0
    for item in uploaded_files or []:
        raw_path = getattr(item, "name", item)
        if not raw_path:
            continue
        path = Path(str(raw_path))
        if not path.is_file():
            raise ExecutionPlanError(f"找不到上传图片：{path.name}")
        try:
            with Image.open(path) as source:
                source.load()
                megapixels = (source.width * source.height) / 1_000_000
                if megapixels > CONFIG.max_input_megapixels:
                    raise ExecutionPlanError(
                        f"图片“{path.name}”为 {megapixels:.1f} MP，超过单图上限 "
                        f"{CONFIG.max_input_megapixels:g} MP。"
                    )
                total_megapixels += megapixels
                if total_megapixels > CONFIG.max_reference_megapixels:
                    raise ExecutionPlanError(
                        f"上传图片累计为 {total_megapixels:.1f} MP，超过上限 "
                        f"{CONFIG.max_reference_megapixels:g} MP；请缩小图片或减少数量。"
                    )
                images.append(source.convert("RGB").copy())
        except ExecutionPlanError:
            raise
        except Exception as exc:
            raise ExecutionPlanError(f"无法读取图片“{path.name}”：{exc}") from exc

    if len(images) > CONFIG.max_multi_images:
        raise ExecutionPlanError(
            f"一次最多上传 {CONFIG.max_multi_images} 张图片；当前为 {len(images)} 张。"
        )
    return images


def _pick_reference_chain(model_name: str, role: str) -> tuple[str, str, int]:
    workflow_type = _workflow_type(model_name)
    enabled = set(
        FEATURES_CONFIG.get(workflow_type, {}).get("enabled_chains", [])
    )

    if role == "identity":
        order = ["krea2_identity_edit"]
    elif role == "style":
        # The generic FLUX style injector has a different image/weight schema;
        # keep this high-level path limited to the validated Krea reference chain.
        order = ["krea2_style_reference"]
    else:
        order = [
            "qwen_image_edit",
            "joyai_image",
            "boogu_image_edit",
            "reference_image",
            "reference_latent",
            "hidream_o1_reference",
            "krea2_identity_edit",
            "krea2_style_reference",
        ]

    for chain_name in order:
        if (
            chain_name in enabled
            and chain_name in REFERENCE_CHAIN_SPECS
            and supports_chain_for_model(model_name, chain_name)
        ):
            input_key, maximum = REFERENCE_CHAIN_SPECS[chain_name]
            return chain_name, input_key, maximum

    if role in {"identity", "style"}:
        raise ExecutionPlanError(
            f"模型“{model_name}”不支持所选的{('身份' if role == 'identity' else '风格')}参考方式。"
        )
    raise ExecutionPlanError(
        f"模型“{model_name}”没有可自动使用的多图参考链；请换用编辑/多模态模型。"
    )


def build_execution_plan(
    base_inputs: dict[str, Any],
    mode: str = MODE_SINGLE,
    extra_models: Sequence[str] | None = None,
    images: Sequence[Image.Image] | None = None,
    reference_role: str = "auto",
    use_model_defaults: bool = True,
) -> list[PlannedGeneration]:
    """Expand one UI submission into a bounded list of sequential runs."""

    if mode not in {choice[1] for choice in RUN_MODE_CHOICES}:
        raise ExecutionPlanError(f"未知运行模式：{mode}")

    base_model = str(base_inputs.get("model_display_name") or "")
    if base_model not in MODEL_MAP_CHECKPOINT:
        raise ExecutionPlanError("请先选择有效模型。")

    comparison_mode = mode in {MODE_MODEL_PK, MODE_MULTI_MODEL_GRID}
    models = _unique(
        [base_model, *(extra_models or [])] if comparison_mode else [base_model]
    )
    unknown_models = [name for name in models if name not in MODEL_MAP_CHECKPOINT]
    if unknown_models:
        raise ExecutionPlanError(f"未知模型：{', '.join(unknown_models)}")
    if len(models) > CONFIG.max_pk_models:
        raise ExecutionPlanError(
            f"模型 PK 最多 {CONFIG.max_pk_models} 个模型；当前为 {len(models)} 个。"
        )

    if mode in {MODE_MODEL_PK, MODE_MULTI_MODEL_GRID} and len(models) < 2:
        raise ExecutionPlanError("模型 PK 至少需要再选择 1 个对比模型。")

    source_images = list(images or [])
    shared_seed = base_inputs.get("seed", -1)
    try:
        shared_seed = int(shared_seed)
    except (TypeError, ValueError):
        shared_seed = -1
    if shared_seed < 0 and mode != MODE_SINGLE:
        shared_seed = random.randint(0, 2**32 - 1)

    plan: list[PlannedGeneration] = []
    if mode == MODE_SINGLE:
        plan.append(PlannedGeneration(dict(base_inputs), base_model))

    elif mode == MODE_MODEL_PK:
        for model_name in models:
            values = _for_model(base_inputs, model_name, use_model_defaults)
            _make_fair_comparison(values)
            values["seed"] = shared_seed
            plan.append(
                PlannedGeneration(values, _caption(f"模型 PK · {model_name}", values))
            )

    elif mode in {MODE_MULTI_INDEPENDENT, MODE_MULTI_MODEL_GRID}:
        task_type = str(base_inputs.get("task_type") or "")
        task_input_key = INDEPENDENT_IMAGE_TASK_KEYS.get(task_type)
        if not task_input_key:
            raise ExecutionPlanError(
                "多图独立处理仅支持图生图、扩图和高清修复；局部重绘需要逐张绘制蒙版。"
            )
        if not source_images:
            raise ExecutionPlanError("请上传至少 1 张批量输入图片。")
        target_models = models if mode == MODE_MULTI_MODEL_GRID else [base_model]
        # Keep one checkpoint active for all its inputs before switching.  This
        # avoids needless reloads while preserving Gallery captions by source.
        for model_name in target_models:
            for image_index, image in enumerate(source_images, start=1):
                values = _for_model(
                    base_inputs,
                    model_name,
                    use_model_defaults if mode == MODE_MULTI_MODEL_GRID else False,
                )
                if mode == MODE_MULTI_MODEL_GRID:
                    _make_fair_comparison(values)
                values["seed"] = shared_seed
                values[task_input_key] = image
                caption = _caption(f"输入 {image_index} · {model_name}", values)
                plan.append(PlannedGeneration(values, caption))

    elif mode == MODE_MULTI_REFERENCE:
        if str(base_inputs.get("task_type")) != "txt2img":
            raise ExecutionPlanError("多图融合请把任务切换为“文生图”；参考图会直接进入编辑模型。")
        if not source_images:
            raise ExecutionPlanError("多图融合需要上传至少 1 张参考图。")
        chain_name, input_key, maximum = _pick_reference_chain(
            base_model, reference_role
        )
        if len(source_images) > maximum:
            raise ExecutionPlanError(
                f"当前模型的 {chain_name} 最多支持 {maximum} 张参考图。"
            )
        values = _for_model(base_inputs, base_model, False)
        existing = [value for value in values.get(input_key, []) if value is not None]
        values[input_key] = [*existing, *source_images][:maximum]
        values["seed"] = shared_seed
        plan.append(
            PlannedGeneration(
                values,
                _caption(
                    f"多图融合 · {base_model} · {len(source_images)} 张参考图",
                    values,
                ),
            )
        )

    if len(plan) > CONFIG.max_plan_jobs:
        raise ExecutionPlanError(
            f"本次会产生 {len(plan)} 个任务，超过上限 {CONFIG.max_plan_jobs}；请减少图片或模型。"
        )
    batch_size = max(1, int(base_inputs.get("batch_size") or 1))
    estimated_outputs = len(plan) * batch_size
    if estimated_outputs > CONFIG.max_plan_outputs:
        raise ExecutionPlanError(
            f"预计输出 {estimated_outputs} 张，超过上限 {CONFIG.max_plan_outputs}；"
            "请减少模型、输入图片或单次生成数量。"
        )
    # Release Comfy's global model state only at an actual model boundary.  The
    # final model remains warm for a likely follow-up generation.
    for current, following in pairwise(plan):
        if (
            current.inputs.get("model_display_name")
            != following.inputs.get("model_display_name")
        ):
            current.inputs["_release_models_after_run"] = True
    return plan


class _PlanProgress:
    def __init__(self, parent: Any, index: int, total: int, caption: str):
        self.parent = parent
        self.index = index
        self.total = total
        self.caption = caption

    def __call__(self, value: float = 0.0, desc: str | None = None):
        if not self.parent:
            return None
        try:
            fraction = max(0.0, min(1.0, float(value)))
        except (TypeError, ValueError):
            fraction = 0.0
        overall = (self.index + fraction) / self.total
        detail = f"[{self.index + 1}/{self.total}] {self.caption}"
        if desc:
            detail += f" · {desc}"
        return self.parent(overall, desc=detail)


def execute_generation_plan(
    plan: Sequence[PlannedGeneration],
    generate: Callable[[dict[str, Any], Any], Any],
    progress: Any = None,
    cancel_event: Any = None,
) -> tuple[list[Any], str]:
    """Run the plan sequentially and retain partial successes."""

    gallery: list[Any] = []
    summary: list[str] = []
    total = max(1, len(plan))
    for index, item in enumerate(plan):
        if cancel_event is not None and cancel_event.is_set():
            if gallery:
                summary.append("- ⏹️ 已取消：后续组合未执行，已保留成功结果。")
                break
            raise ExecutionPlanError("任务已取消，后续组合未执行。")
        try:
            result = generate(
                item.inputs,
                _PlanProgress(progress, index, total, item.caption),
            )
            paths = result if isinstance(result, list) else ([result] if result else [])
            for output_index, path in enumerate(paths, start=1):
                caption = item.caption
                if len(paths) > 1:
                    caption += f" · 结果 {output_index}"
                gallery.append((path, caption))
            summary.append(f"- ✅ {item.caption}：{len(paths)} 张")
        except Exception as exc:
            if cancel_event is not None and cancel_event.is_set():
                if gallery:
                    summary.append("- ⏹️ 已取消：后续组合未执行，已保留成功结果。")
                    break
                raise
            summary.append(f"- ❌ {item.caption}：{exc}")

    if not gallery:
        raise ExecutionPlanError("本次任务没有成功生成图片。\n" + "\n".join(summary))
    if progress:
        progress(1.0, desc="全部组合执行完成。")
    return gallery, "### 本次执行\n" + "\n".join(summary)
