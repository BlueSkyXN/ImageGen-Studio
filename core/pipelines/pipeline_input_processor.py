import os
import uuid
import numpy as np
import gradio as gr
from PIL import Image, ImageChops
from typing import Dict, Any, List

from core.settings import INPUT_DIR, MULTIPLIERS_MAP, LORA_DIR, EMBEDDING_DIR, VAE_DIR
from core.runtime_config import CONFIG
from imagegen_utils.app_utils import (
    sanitize_filename,
    get_lora_path,
    get_embedding_path,
    ensure_controlnet_model_downloaded,
    ensure_ipadapter_models_downloaded,
    _ensure_model_downloaded,
    ensure_sd3_ipadapter_models_downloaded,
    get_vae_path,
)


def _temp_png(stem: str) -> str:
    return os.path.join(INPUT_DIR, f"{stem}_{uuid.uuid4().hex}.png")


REFERENCE_IMAGE_LIMITS = {
    "controlnet_data": 5,
    "anima_controlnet_lllite_data": 5,
    "diffsynth_controlnet_data": 5,
    "krea2_controlnet_data": 5,
    "ipadapter_data": 5,
    "flux1_ipadapter_data": 5,
    "sd3_ipadapter_chain": 5,
    "style_data": 5,
    "reference_latent_data": 10,
    "hidream_o1_reference_data": 10,
    "joyai_reference_data": 2,
    "krea2_identity_edit_data": 2,
    "krea2_reference_edit_data": 3,
    "qwen_image_edit_data": 3,
    "boogu_edit_data": 10,
    "reference_image_data": 10,
}


def _pil_images(value: Any):
    if isinstance(value, Image.Image):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from _pil_images(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _pil_images(child)


def _validate_reference_image_budget(ui_inputs: Dict[str, Any]) -> None:
    """Bound decoded reference images before saving or entering a workflow."""

    all_images = []
    for input_key, chain_limit in REFERENCE_IMAGE_LIMITS.items():
        images = list(_pil_images(ui_inputs.get(input_key)))
        if len(images) > chain_limit:
            raise gr.Error(
                f"扩展“{input_key}”最多支持 {chain_limit} 张图片；当前为 {len(images)} 张。"
            )
        all_images.extend(images)

    if len(all_images) > CONFIG.max_reference_images:
        raise gr.Error(
            f"一次任务最多使用 {CONFIG.max_reference_images} 张参考/控制图；"
            f"当前合计 {len(all_images)} 张。"
        )

    total_megapixels = 0.0
    for index, image in enumerate(all_images, start=1):
        megapixels = (image.width * image.height) / 1_000_000
        if megapixels > CONFIG.max_input_megapixels:
            raise gr.Error(
                f"参考图 {index} 为 {megapixels:.1f}MP，超过单图上限 "
                f"{CONFIG.max_input_megapixels:g}MP。"
            )
        total_megapixels += megapixels
    if total_megapixels > CONFIG.max_reference_megapixels:
        raise gr.Error(
            f"参考/控制图累计为 {total_megapixels:.1f}MP，超过上限 "
            f"{CONFIG.max_reference_megapixels:g}MP；请缩小图片或减少数量。"
        )

def process_pipeline_inputs(ui_inputs: Dict[str, Any], progress: gr.Progress, workflow_model_type: str) -> Dict[str, Any]:
    task_type = ui_inputs['task_type']
    temp_files_to_clean = []
    _validate_reference_image_budget(ui_inputs)

    multiplier = MULTIPLIERS_MAP.get(workflow_model_type, 8)
    img_w, img_h = 0, 0
    if task_type == 'txt2img':
        img_w = int(ui_inputs.get('width', 0))
        img_h = int(ui_inputs.get('height', 0))
    elif task_type == 'img2img':
        input_image_pil = ui_inputs.get('img2img_image')
        if input_image_pil:
            img_w, img_h = input_image_pil.width, input_image_pil.height
    elif task_type == 'inpaint':
        inpaint_img = ui_inputs.get('inpaint_image')
        inpaint_dict = ui_inputs.get('inpaint_image_dict')
        if inpaint_img:
            img_w, img_h = inpaint_img.width, inpaint_img.height
        elif inpaint_dict and inpaint_dict.get('background'):
            img_w, img_h = inpaint_dict['background'].width, inpaint_dict['background'].height
    elif task_type == 'outpaint':
        input_image_pil = ui_inputs.get('outpaint_image')
        if input_image_pil:
            img_w, img_h = input_image_pil.width, input_image_pil.height
    elif task_type == 'hires_fix':
        input_image_pil = ui_inputs.get('hires_image')
        if input_image_pil:
            img_w, img_h = input_image_pil.width, input_image_pil.height

    if task_type == "txt2img" and (img_w <= 0 or img_h <= 0):
        raise gr.Error("文生图的宽度和高度必须为正整数。")

    if img_w > 0 and img_h > 0:
        input_megapixels = (img_w * img_h) / 1_000_000
        if input_megapixels > CONFIG.max_input_megapixels:
            scale = (CONFIG.max_input_megapixels / input_megapixels) ** 0.5
            suggested_w = max(multiplier, int(img_w * scale) // multiplier * multiplier)
            suggested_h = max(multiplier, int(img_h * scale) // multiplier * multiplier)
            raise gr.Error(
                f"输入图片为 {input_megapixels:.1f}MP，超过当前上限 "
                f"{CONFIG.max_input_megapixels:g}MP；建议缩小到约 "
                f"{suggested_w}×{suggested_h}。"
            )

        projected_w, projected_h = img_w, img_h
        if task_type == "hires_fix":
            upscale = float(ui_inputs.get("hires_scale_by") or 1.5)
            projected_w, projected_h = int(img_w * upscale), int(img_h * upscale)
        elif task_type == "outpaint":
            projected_w = img_w + int(ui_inputs.get("left") or 0) + int(ui_inputs.get("right") or 0)
            projected_h = img_h + int(ui_inputs.get("top") or 0) + int(ui_inputs.get("bottom") or 0)
        projected_megapixels = (projected_w * projected_h) / 1_000_000
        if projected_megapixels > CONFIG.max_output_megapixels:
            raise gr.Error(
                f"预计输出为 {projected_w}×{projected_h}（{projected_megapixels:.1f}MP），"
                f"超过当前上限 {CONFIG.max_output_megapixels:g}MP；请降低放大倍数或扩边尺寸。"
            )

        if (img_w % multiplier != 0) or (img_h % multiplier != 0):
            suggested_w = max(multiplier, round(img_w / multiplier) * multiplier)
            suggested_h = max(multiplier, round(img_h / multiplier) * multiplier)
            warning_msg = (
                f"当前模型要求宽高均为 {multiplier} 的倍数；"
                f"收到 {img_w}×{img_h}，可调整为约 {suggested_w}×{suggested_h}。"
            )
            raise gr.Error(warning_msg)

    lora_data = ui_inputs.get('lora_data', [])
    active_loras_for_gpu, active_loras_for_meta = [], []
    if lora_data:
        sources, ids, scales, files = lora_data[0::4], lora_data[1::4], lora_data[2::4], lora_data[3::4]
        for i, (source, lora_id, scale, _) in enumerate(zip(sources, ids, scales, files)):
            if scale > 0 and lora_id and lora_id.strip():
                lora_filename = None
                if source == "File":
                    lora_filename = sanitize_filename(lora_id)
                    local_path = os.path.join(LORA_DIR, lora_filename)
                    if not os.path.exists(local_path):
                        raise gr.Error(f"已上传的 LoRA“{lora_id}”已不存在，请重新上传。")
                elif source in ("Civitai", "Hugging Face"):
                    local_path, status = get_lora_path(source, lora_id, os.environ.get("CIVITAI_API_KEY", ""), progress)
                    if local_path: lora_filename = os.path.basename(local_path)
                    else: raise gr.Error(f"LoRA“{lora_id}”准备失败：{status}")

                if lora_filename:
                    active_loras_for_gpu.append({"lora_name": lora_filename, "strength_model": scale, "strength_clip": scale})
                    active_loras_for_meta.append(f"{source} {lora_id}:{scale}")

    ui_inputs['denoise'] = 1.0
    if task_type == 'img2img': ui_inputs['denoise'] = ui_inputs.get('img2img_denoise', 0.7)
    elif task_type == 'hires_fix': ui_inputs['denoise'] = ui_inputs.get('hires_denoise', 0.55)
    elif task_type == 'inpaint': ui_inputs['denoise'] = ui_inputs.get('inpaint_denoise', 1.0)

    if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)

    if task_type == 'img2img':
        input_image_pil = ui_inputs.get('img2img_image')
        if not input_image_pil:
            raise gr.Error("图生图需要先上传源图片。")
        temp_file_path = _temp_png("temp_input")
        input_image_pil.save(temp_file_path, "PNG")
        ui_inputs['input_image'] = os.path.basename(temp_file_path)
        temp_files_to_clean.append(temp_file_path)
        ui_inputs['width'] = input_image_pil.width
        ui_inputs['height'] = input_image_pil.height

    elif task_type == 'inpaint':
        inpaint_img = ui_inputs.get('inpaint_image')
        inpaint_dict = ui_inputs.get('inpaint_image_dict')

        if inpaint_img:
            temp_file_path = _temp_png("temp_inpaint")
            inpaint_img.save(temp_file_path, "PNG")
            ui_inputs['input_image'] = os.path.basename(temp_file_path)
            temp_files_to_clean.append(temp_file_path)
            ui_inputs['width'] = inpaint_img.width
            ui_inputs['height'] = inpaint_img.height
        elif inpaint_dict and inpaint_dict.get('background') and inpaint_dict.get('layers'):
            background_img = inpaint_dict['background'].convert("RGBA")
            composite_mask_pil = Image.new('L', background_img.size, 0)
            for layer in inpaint_dict['layers']:
                if layer:
                    layer_alpha = layer.split()[-1]
                    composite_mask_pil = ImageChops.lighter(composite_mask_pil, layer_alpha)

            inverted_mask_alpha = Image.fromarray(255 - np.array(composite_mask_pil), mode='L')
            r, g, b, _ = background_img.split()
            composite_image_with_mask = Image.merge('RGBA', [r, g, b, inverted_mask_alpha])

            temp_file_path = _temp_png("temp_inpaint_composite")
            composite_image_with_mask.save(temp_file_path, "PNG")

            ui_inputs['input_image'] = os.path.basename(temp_file_path)
            temp_files_to_clean.append(temp_file_path)
            ui_inputs.pop('inpaint_mask', None)
            ui_inputs['width'] = background_img.width
            ui_inputs['height'] = background_img.height
        else:
            raise gr.Error("局部重绘需要输入图片和有效蒙版。")

    elif task_type == 'outpaint':
        input_image_pil = ui_inputs.get('outpaint_image')
        if not input_image_pil:
            raise gr.Error("扩图需要先上传源图片。")
        temp_file_path = _temp_png("temp_input")
        input_image_pil.save(temp_file_path, "PNG")
        ui_inputs['input_image'] = os.path.basename(temp_file_path)
        temp_files_to_clean.append(temp_file_path)

        ui_inputs['megapixels'] = 0.25
        ui_inputs['grow_mask_by'] = ui_inputs.get('feathering', 10)
        ui_inputs['width'] = input_image_pil.width + int(ui_inputs.get('left') or 0) + int(ui_inputs.get('right') or 0)
        ui_inputs['height'] = input_image_pil.height + int(ui_inputs.get('top') or 0) + int(ui_inputs.get('bottom') or 0)

    elif task_type == 'hires_fix':
        input_image_pil = ui_inputs.get('hires_image')
        if not input_image_pil:
            raise gr.Error("高清修复需要先上传源图片。")
        temp_file_path = _temp_png("temp_input")
        input_image_pil.save(temp_file_path, "PNG")
        ui_inputs['input_image'] = os.path.basename(temp_file_path)
        temp_files_to_clean.append(temp_file_path)
        hires_scale = float(ui_inputs.get('hires_scale_by') or 1.5)
        ui_inputs['width'] = int(input_image_pil.width * hires_scale)
        ui_inputs['height'] = int(input_image_pil.height * hires_scale)

    embedding_data = ui_inputs.get('embedding_data', [])
    embedding_filenames = []
    if embedding_data:
        emb_sources, emb_ids, emb_files = embedding_data[0::3], embedding_data[1::3], embedding_data[2::3]
        for i, (source, emb_id, _) in enumerate(zip(emb_sources, emb_ids, emb_files)):
            if emb_id and emb_id.strip():
                emb_filename = None
                if source == "File":
                    emb_filename = sanitize_filename(emb_id)
                    local_path = os.path.join(EMBEDDING_DIR, emb_filename)
                    if not os.path.exists(local_path):
                        raise gr.Error(f"已上传的 Embedding“{emb_id}”已不存在，请重新上传。")
                elif source in ("Civitai", "Hugging Face"):
                    local_path, status = get_embedding_path(source, emb_id, os.environ.get("CIVITAI_API_KEY", ""), progress)
                    if local_path: emb_filename = os.path.basename(local_path)
                    else: raise gr.Error(f"Embedding“{emb_id}”准备失败：{status}")

                if emb_filename:
                    embedding_filenames.append(emb_filename)

    controlnet_data = ui_inputs.get('controlnet_data', [])
    active_controlnets = []
    if controlnet_data:
        (cn_images, _, _, cn_strengths, cn_filepaths) = [controlnet_data[i::5] for i in range(5)]
        for i in range(len(cn_images)):
            if cn_images[i] and cn_strengths[i] > 0 and cn_filepaths[i] and cn_filepaths[i] != "None":
                ensure_controlnet_model_downloaded(cn_filepaths[i], progress)
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                cn_temp_path = _temp_png(f"temp_cn_{i}")
                cn_images[i].save(cn_temp_path, "PNG")
                temp_files_to_clean.append(cn_temp_path)
                active_controlnets.append({
                    "image": os.path.basename(cn_temp_path), "strength": cn_strengths[i],
                    "start_percent": 0.0, "end_percent": 1.0, "control_net_name": cn_filepaths[i]
                })

    anima_controlnet_lllite_data = ui_inputs.get('anima_controlnet_lllite_data', [])
    active_anima_controlnets = []
    if anima_controlnet_lllite_data:
        (cn_images, _, _, cn_strengths, cn_filepaths, cn_starts, cn_ends) = [anima_controlnet_lllite_data[i::7] for i in range(7)]
        for i in range(len(cn_images)):
            if cn_images[i] and cn_strengths[i] > 0 and cn_filepaths[i] and cn_filepaths[i] != "None":
                _ensure_model_downloaded(cn_filepaths[i], progress)
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                cn_temp_path = _temp_png(f"temp_anima_cn_{i}")
                cn_images[i].save(cn_temp_path, "PNG")
                temp_files_to_clean.append(cn_temp_path)
                active_anima_controlnets.append({
                    "image": os.path.basename(cn_temp_path), "strength": cn_strengths[i],
                    "start_percent": cn_starts[i], "end_percent": cn_ends[i], "control_net_name": cn_filepaths[i]
                })

    diffsynth_controlnet_data = ui_inputs.get('diffsynth_controlnet_data', [])
    active_diffsynth_controlnets = []
    if diffsynth_controlnet_data:
        (cn_images, _, _, cn_strengths, cn_filepaths) = [diffsynth_controlnet_data[i::5] for i in range(5)]
        for i in range(len(cn_images)):
            if cn_images[i] and cn_strengths[i] > 0 and cn_filepaths[i] and cn_filepaths[i] != "None":
                ensure_controlnet_model_downloaded(cn_filepaths[i], progress)
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                cn_temp_path = _temp_png(f"temp_diffsynth_cn_{i}")
                cn_images[i].save(cn_temp_path, "PNG")
                temp_files_to_clean.append(cn_temp_path)
                active_diffsynth_controlnets.append({
                    "image": os.path.basename(cn_temp_path), "strength": cn_strengths[i],
                    "control_net_name": cn_filepaths[i]
                })

    krea2_controlnet_data = ui_inputs.get('krea2_controlnet_data', [])
    active_krea2_controlnets = []
    if krea2_controlnet_data:
        (cn_images, _, _, cn_strengths, cn_filepaths) = [krea2_controlnet_data[i::5] for i in range(5)]
        for i in range(len(cn_images)):
            if cn_images[i] and cn_strengths[i] > 0 and cn_filepaths[i] and cn_filepaths[i] != "None":
                ensure_controlnet_model_downloaded(cn_filepaths[i], progress)
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                cn_temp_path = _temp_png(f"temp_krea2_cn_{i}")
                cn_images[i].save(cn_temp_path, "PNG")
                temp_files_to_clean.append(cn_temp_path)
                active_krea2_controlnets.append({
                    "image": os.path.basename(cn_temp_path), "strength": cn_strengths[i],
                    "control_net_name": cn_filepaths[i]
                })

    ipadapter_data = ui_inputs.get('ipadapter_data', [])
    active_ipadapters = []
    if ipadapter_data:
        num_ipa_units = (len(ipadapter_data) - 5) // 3
        final_preset, final_weight, final_lora_strength, final_embeds_scaling, final_combine_method = ipadapter_data[-5:]
        ipa_images, ipa_weights, ipa_lora_strengths = [ipadapter_data[i*num_ipa_units:(i+1)*num_ipa_units] for i in range(3)]
        all_presets_to_download = set()
        for i in range(num_ipa_units):
            if ipa_images[i] and ipa_weights[i] > 0 and final_preset:
                all_presets_to_download.add(final_preset)
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                ipa_temp_path = _temp_png(f"temp_ipa_{i}")
                ipa_images[i].save(ipa_temp_path, "PNG")
                temp_files_to_clean.append(ipa_temp_path)
                active_ipadapters.append({
                    "image": os.path.basename(ipa_temp_path), "preset": final_preset,
                    "weight": ipa_weights[i], "lora_strength": ipa_lora_strengths[i]
                })
        if active_ipadapters and final_preset:
            all_presets_to_download.add(final_preset)
        for preset in all_presets_to_download:
            ensure_ipadapter_models_downloaded(preset, progress)

        model_type_key = 'sd15' if workflow_model_type == 'sd15' else 'sdxl'
        if active_ipadapters:
            active_ipadapters.append({
                'is_final_settings': True, 'model_type': model_type_key, 'final_preset': final_preset,
                'final_weight': final_weight, 'final_lora_strength': final_lora_strength,
                'final_embeds_scaling': final_embeds_scaling, 'final_combine_method': final_combine_method
            })

    flux1_ipadapter_data = ui_inputs.get('flux1_ipadapter_data', [])
    active_flux1_ipadapters = []
    if flux1_ipadapter_data:
        num_units = len(flux1_ipadapter_data) // 4
        f_images = flux1_ipadapter_data[0*num_units : 1*num_units]
        f_weights = flux1_ipadapter_data[1*num_units : 2*num_units]
        f_starts = flux1_ipadapter_data[2*num_units : 3*num_units]
        f_ends = flux1_ipadapter_data[3*num_units : 4*num_units]
        for i in range(len(f_images)):
            if f_images[i] and f_weights[i] > 0:
                for filename in ["ip-adapter.bin"]:
                    _ensure_model_downloaded(filename, progress)

                from huggingface_hub import snapshot_download
                progress(0.5, desc="Caching HF SigLIP model...")
                snapshot_download(
                    repo_id="google/siglip-so400m-patch14-384",
                    allow_patterns=["*.json", "*.safetensors", "*.txt"],
                    ignore_patterns=["*.msgpack", "*.h5", "*.bin"]
                )

                temp_path = _temp_png(f"temp_fipa_{i}")
                f_images[i].save(temp_path, "PNG")
                temp_files_to_clean.append(temp_path)
                active_flux1_ipadapters.append({
                    "image": os.path.basename(temp_path),
                    "weight": f_weights[i], "start_percent": f_starts[i], "end_percent": f_ends[i]
                })

    sd3_ipadapter_data = ui_inputs.get('sd3_ipadapter_chain', [])
    active_sd3_ipadapters = []
    if sd3_ipadapter_data:
        num_units = len(sd3_ipadapter_data) // 4
        s_images = sd3_ipadapter_data[0*num_units : 1*num_units]
        s_weights = sd3_ipadapter_data[1*num_units : 2*num_units]
        s_starts = sd3_ipadapter_data[2*num_units : 3*num_units]
        s_ends = sd3_ipadapter_data[3*num_units : 4*num_units]
        sd3_ipa_downloaded = False
        for i in range(len(s_images)):
            if s_images[i] and s_weights[i] > 0:
                if not sd3_ipa_downloaded:
                    ensure_sd3_ipadapter_models_downloaded(progress)
                    sd3_ipa_downloaded = True
                temp_path = _temp_png(f"temp_s3ipa_{i}")
                s_images[i].save(temp_path, "PNG")
                temp_files_to_clean.append(temp_path)
                active_sd3_ipadapters.append({
                    "image": os.path.basename(temp_path),
                    "weight": s_weights[i], "start_percent": s_starts[i], "end_percent": s_ends[i]
                })

    style_data = ui_inputs.get('style_data', [])
    active_styles = []
    if style_data:
        num_units = len(style_data) // 2
        st_images = style_data[0*num_units : 1*num_units]
        st_strengths = style_data[1*num_units : 2*num_units]
        style_models_downloaded = False
        for i in range(len(st_images)):
            if st_images[i] and st_strengths[i] > 0:
                if not style_models_downloaded:
                    _ensure_model_downloaded("sigclip_vision_patch14_384.safetensors", progress)
                    _ensure_model_downloaded("flux1-redux-dev.safetensors", progress)
                    style_models_downloaded = True
                temp_path = _temp_png(f"temp_style_{i}")
                st_images[i].save(temp_path, "PNG")
                temp_files_to_clean.append(temp_path)
                active_styles.append({
                    "image": os.path.basename(temp_path), "strength": st_strengths[i]
                })

    reference_latent_data = ui_inputs.get('reference_latent_data', [])
    active_reference_latents = []
    if reference_latent_data:
        for img in reference_latent_data:
            if img:
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                temp_path = _temp_png("temp_ref")
                img.save(temp_path, "PNG")
                temp_files_to_clean.append(temp_path)
                active_reference_latents.append(os.path.basename(temp_path))

    hidream_o1_reference_data = ui_inputs.get('hidream_o1_reference_data', [])
    active_hidream_o1_reference = []
    if hidream_o1_reference_data:
        for img in hidream_o1_reference_data:
            if img:
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                temp_path = _temp_png("temp_ho1_ref")
                img.save(temp_path, "PNG")
                temp_files_to_clean.append(temp_path)
                active_hidream_o1_reference.append(os.path.basename(temp_path))

    joyai_reference_data = ui_inputs.get('joyai_reference_data', [])
    active_joyai_reference = []
    if joyai_reference_data:
        for img in joyai_reference_data:
            if img:
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                temp_path = _temp_png("temp_joyai_ref")
                img.save(temp_path, "PNG")
                temp_files_to_clean.append(temp_path)
                active_joyai_reference.append(os.path.basename(temp_path))

    krea2_identity_edit_data = ui_inputs.get('krea2_identity_edit_data', [])
    active_krea2_identity_edit = []
    if krea2_identity_edit_data:
        for img in krea2_identity_edit_data:
            if img:
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                temp_path = _temp_png("temp_krea2_id_ref")
                img.save(temp_path, "PNG")
                temp_files_to_clean.append(temp_path)
                active_krea2_identity_edit.append(os.path.basename(temp_path))

    krea2_reference_edit_data = ui_inputs.get('krea2_reference_edit_data', [])
    active_krea2_reference_edit = []
    if krea2_reference_edit_data:
        for img in krea2_reference_edit_data:
            if img:
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                temp_path = _temp_png("temp_krea2_reference_ref")
                img.save(temp_path, "PNG")
                temp_files_to_clean.append(temp_path)
                active_krea2_reference_edit.append(os.path.basename(temp_path))

    qwen_image_edit_data = ui_inputs.get('qwen_image_edit_data', [])
    active_qwen_image_edit = []
    if qwen_image_edit_data:
        for img in qwen_image_edit_data:
            if img:
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                temp_path = _temp_png("temp_qwen_edit_ref")
                img.save(temp_path, "PNG")
                temp_files_to_clean.append(temp_path)
                active_qwen_image_edit.append(os.path.basename(temp_path))

    boogu_edit_data = ui_inputs.get('boogu_edit_data', [])
    active_boogu_edit = []
    if boogu_edit_data:
        for img in boogu_edit_data:
            if img:
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                temp_path = _temp_png("temp_boogu_edit_ref")
                img.save(temp_path, "PNG")
                temp_files_to_clean.append(temp_path)
                active_boogu_edit.append(os.path.basename(temp_path))

    reference_image_data = ui_inputs.get('reference_image_data', [])
    active_reference_images = []
    if reference_image_data:
        for img in reference_image_data:
            if img:
                if not os.path.exists(INPUT_DIR): os.makedirs(INPUT_DIR)
                temp_path = _temp_png("temp_ref_img")
                img.save(temp_path, "PNG")
                temp_files_to_clean.append(temp_path)
                active_reference_images.append(os.path.basename(temp_path))

    vae_source = ui_inputs.get('vae_source')
    vae_id = ui_inputs.get('vae_id')
    vae_name_override = None
    if vae_source and vae_source != "None":
        if vae_source == "File":
            vae_name_override = sanitize_filename(vae_id)
            local_path = os.path.join(VAE_DIR, vae_name_override)
            if not os.path.exists(local_path):
                raise gr.Error(f"已上传的 VAE“{vae_id}”已不存在，请重新上传。")
        elif vae_source in ("Civitai", "Hugging Face") and vae_id and vae_id.strip():
            local_path, status = get_vae_path(vae_source, vae_id, os.environ.get("CIVITAI_API_KEY", ""), progress)
            if local_path: vae_name_override = os.path.basename(local_path)
            else: raise gr.Error(f"VAE“{vae_id}”准备失败：{status}")
    if vae_name_override:
        ui_inputs['vae_name'] = vae_name_override

    conditioning_data = ui_inputs.get('conditioning_data', [])
    active_conditioning = []
    if conditioning_data:
        num_units = len(conditioning_data) // 6
        prompts, widths, heights, xs, ys, strengths = [conditioning_data[i*num_units : (i+1)*num_units] for i in range(6)]
        for i in range(num_units):
            if prompts[i] and prompts[i].strip():
                active_conditioning.append({
                    "prompt": prompts[i], "width": int(widths[i]), "height": int(heights[i]),
                    "x": int(xs[i]), "y": int(ys[i]), "strength": float(strengths[i])
                })

    return {
        "active_loras_for_gpu": active_loras_for_gpu,
        "active_loras_for_meta": active_loras_for_meta,
        "active_controlnets": active_controlnets,
        "active_anima_controlnets": active_anima_controlnets,
        "active_diffsynth_controlnets": active_diffsynth_controlnets,
        "active_krea2_controlnets": active_krea2_controlnets,
        "active_ipadapters": active_ipadapters,
        "active_flux1_ipadapters": active_flux1_ipadapters,
        "active_sd3_ipadapters": active_sd3_ipadapters,
        "active_styles": active_styles,
        "active_reference_latents": active_reference_latents,
        "active_hidream_o1_reference": active_hidream_o1_reference,
        "active_joyai_reference": active_joyai_reference,
        "active_krea2_identity_edit": active_krea2_identity_edit,
        "active_krea2_reference_edit": active_krea2_reference_edit,
        "active_qwen_image_edit": active_qwen_image_edit,
        "active_boogu_edit": active_boogu_edit,
        "active_reference_images": active_reference_images,
        "active_conditioning": active_conditioning,
        "temp_files_to_clean": temp_files_to_clean
    }
