"""
MCP Common Utilities & Data Structures
Contains YAML loading utilities, config file paths, task definitions, and async task database.
"""

import os
import time
import urllib.parse
import urllib.request
import urllib.error
import ipaddress
import socket
import base64
import io
import copy
import threading
import yaml
from typing import Dict, Any
from PIL import Image
from core.runtime_config import CONFIG

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_YAML_DIR = os.path.join(_PROJECT_ROOT, "yaml")

_MODEL_ARCHITECTURES_PATH = os.path.join(_YAML_DIR, "model_architectures.yaml")
_MODEL_LIST_PATH = os.path.join(_YAML_DIR, "model_list.yaml")
_MODEL_DEFAULTS_PATH = os.path.join(_YAML_DIR, "model_defaults.yaml")
_IMAGE_GEN_FEATURES_PATH = os.path.join(_YAML_DIR, "image_gen_features.yaml")
_CHAIN_FEATURES_PATH = os.path.join(_YAML_DIR, "chain_features.yaml")
_CONSTANTS_PATH = os.path.join(_YAML_DIR, "constants.yaml")


_MAX_IMAGE_DOWNLOAD_BYTES = 50 * 1024 * 1024  # 50 MB
_IMAGE_DOWNLOAD_TIMEOUT = 30  # seconds
_ALLOWED_IMAGE_CONTENT_TYPES = frozenset([
    "image/png", "image/jpeg", "image/jpg", "image/gif",
    "image/webp", "image/bmp", "image/tiff",
])


def _validate_public_image_url(url: str) -> None:
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Image URL must use HTTP or HTTPS and include a host.")
    if parsed.username or parsed.password:
        raise ValueError("Image URLs containing credentials are not accepted.")
    try:
        addresses = {
            item[4][0]
            for item in socket.getaddrinfo(
                parsed.hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
            )
        }
    except socket.gaierror as exc:
        raise ValueError(f"Could not resolve image URL host: {exc}") from exc
    for address in addresses:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
        if not ip.is_global:
            raise ValueError("Image URL must resolve to a public network address.")


class _PublicOnlyRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _validate_public_image_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _get_ipadapter_presets_by_arch() -> Dict[str, list]:
    """Load IPAdapter presets from yaml/ipadapter.yaml for SD1.5 and SDXL."""
    ipadapter_yaml_path = os.path.join(_YAML_DIR, "ipadapter.yaml")
    data = _load_yaml(ipadapter_yaml_path)
    res = {}
    for arch in ("SD1.5", "SDXL"):
        std = data.get("IPAdapter_presets", {}).get(arch, [])
        face = data.get("IPAdapter_FaceID_presets", {}).get(arch, [])
        res[arch] = list(std) + list(face)
    return res


def _parse_image_param(image_param: Any) -> Any:
    """Parse a Base64 Data URI, HTTP/HTTPS URL, or PIL.Image into a PIL Image object."""
    if isinstance(image_param, Image.Image):
        return _validate_decoded_image(image_param)

    if not isinstance(image_param, str) or not image_param.strip():
        return None

    image_param = image_param.strip()

    # HTTP / HTTPS URL — download the image
    if image_param.startswith("http://") or image_param.startswith("https://"):
        return _validate_decoded_image(_download_image_from_url(image_param))

    # Base64 Data URI (e.g. data:image/png;base64,...)
    if image_param.startswith("data:image/"):
        _, encoded = image_param.split(",", 1) if "," in image_param else ("", image_param)
        if len(encoded) > (_MAX_IMAGE_DOWNLOAD_BYTES * 4 // 3) + 8:
            raise ValueError("Base64 image exceeds the 50 MB decoded-size limit.")
        data = base64.b64decode(encoded)
        if len(data) > _MAX_IMAGE_DOWNLOAD_BYTES:
            raise ValueError("Decoded image exceeds the 50 MB size limit.")
        return _validate_decoded_image(Image.open(io.BytesIO(data)))

    # Base64 string without header
    if len(image_param) > 100:
        try:
            if len(image_param) > (_MAX_IMAGE_DOWNLOAD_BYTES * 4 // 3) + 8:
                raise ValueError("Base64 image exceeds the 50 MB decoded-size limit.")
            data = base64.b64decode(image_param)
            if len(data) > _MAX_IMAGE_DOWNLOAD_BYTES:
                raise ValueError("Decoded image exceeds the 50 MB size limit.")
            return _validate_decoded_image(Image.open(io.BytesIO(data)))
        except ValueError:
            raise
        except Exception:
            pass

    raise ValueError(
        "Invalid image parameter format. Expected a Base64 Data URI (e.g., 'data:image/png;base64,...') "
        "or an HTTP/HTTPS URL."
    )


def _validate_decoded_image(image: Image.Image) -> Image.Image:
    image.load()
    megapixels = (image.width * image.height) / 1_000_000
    if megapixels > CONFIG.max_input_megapixels:
        raise ValueError(
            f"Decoded image is {megapixels:.1f} MP; maximum is "
            f"{CONFIG.max_input_megapixels:g} MP."
        )
    return image.copy()


def _download_image_from_url(url: str) -> Image.Image:
    """Download an image from an HTTP/HTTPS URL and return it as a PIL Image.

    Security measures:
    - Timeout to prevent hanging on slow/malicious servers.
    - Response size cap to prevent memory exhaustion.
    - Content-Type validation to reject non-image responses.
    """
    _validate_public_image_url(url)
    req = urllib.request.Request(url, headers={"User-Agent": "ImageGen-MCP/1.0"})
    opener = urllib.request.build_opener(_PublicOnlyRedirectHandler())
    try:
        with opener.open(req, timeout=_IMAGE_DOWNLOAD_TIMEOUT) as resp:
            # Validate content type
            content_type = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if content_type and content_type not in _ALLOWED_IMAGE_CONTENT_TYPES:
                raise ValueError(
                    f"URL returned non-image Content-Type '{content_type}'. "
                    f"Expected one of: {', '.join(sorted(_ALLOWED_IMAGE_CONTENT_TYPES))}."
                )

            # Enforce size limit
            content_length = resp.headers.get("Content-Length")
            if content_length and int(content_length) > _MAX_IMAGE_DOWNLOAD_BYTES:
                raise ValueError(
                    f"Image at URL is too large ({int(content_length)} bytes). "
                    f"Maximum allowed size is {_MAX_IMAGE_DOWNLOAD_BYTES} bytes."
                )

            # Stream-read with size cap
            chunks = []
            total = 0
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                total += len(chunk)
                if total > _MAX_IMAGE_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Image download exceeded maximum allowed size of "
                        f"{_MAX_IMAGE_DOWNLOAD_BYTES} bytes."
                    )
                chunks.append(chunk)

            data = b"".join(chunks)

    except urllib.error.HTTPError as e:
        raise ValueError(f"HTTP error {e.code} when downloading image from URL: {e.reason}") from e
    except urllib.error.URLError as e:
        raise ValueError(f"Failed to download image from URL: {e}") from e

    if not data:
        raise ValueError("Downloaded image data is empty.")

    return Image.open(io.BytesIO(data))


def _load_yaml(filepath: str) -> dict:
    """Safely load a YAML file, returning an empty dict if the file does not exist."""
    if not os.path.exists(filepath):
        print(f"Warning: YAML file not found: {filepath}")
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


_COMMON_OPTIONAL_INPUTS = [
    "steps", "cfg", "sampler", "scheduler", "seed",
    "negative_prompt", "batch_size", "chain", "async_execution",
]

_TASK_DEFINITIONS = [
    {
        "task_type": "txt2img",
        "display_name": "Text-to-Image",
        "description": "Generate images from text prompts. Canvas width and height must be specified.",
        "required_inputs": ["prompt", "width", "height"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
    },
    {
        "task_type": "img2img",
        "display_name": "Image-to-Image",
        "description": "Perform global repaint and style transfer based on a source image. Denoise strength must be specified.",
        "required_inputs": ["prompt", "image", "denoise"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
    },
    {
        "task_type": "inpaint",
        "display_name": "Inpaint",
        "description": "Repaint specified masked regions of the input image (with alpha mask/channel).",
        "required_inputs": ["prompt", "image"],
        "optional_inputs": ["denoise"] + _COMMON_OPTIONAL_INPUTS,
    },
    {
        "task_type": "outpaint",
        "display_name": "Outpaint",
        "description": "Extend the canvas outward from the source image. Padding pixel values for top, bottom, left, and right must be specified.",
        "required_inputs": ["prompt", "image", "pad_left", "pad_right", "pad_top", "pad_bottom"],
        "optional_inputs": _COMMON_OPTIONAL_INPUTS,
    },
    {
        "task_type": "hires_fix",
        "display_name": "Hi-Res Fix / Upscale",
        "description": "Enhance details and upscale an existing low-resolution image.",
        "required_inputs": ["prompt", "image", "upscale_by"],
        "optional_inputs": ["upscaler", "denoise"] + _COMMON_OPTIONAL_INPUTS,
    },
]

_TASKS_DB: Dict[str, Dict[str, Any]] = {}
_TASKS_LOCK = threading.RLock()


def _update_task(task_id: str, **changes) -> None:
    with _TASKS_LOCK:
        if task_id in _TASKS_DB:
            _TASKS_DB[task_id].update(changes)


def _get_task_snapshot(task_id: str) -> Dict[str, Any] | None:
    with _TASKS_LOCK:
        task = _TASKS_DB.get(task_id)
        return copy.deepcopy(task) if task is not None else None


class DummyProgress:
    def __call__(self, progress=0.0, desc=None):
        pass


def _get_public_base_url() -> str:
    """Auto-resolve the publicly accessible base URL (including protocol and port)."""
    # 1. Explicit environment variable override
    public_url = os.getenv("PUBLIC_URL") or os.getenv("BASE_URL")
    if public_url:
        return public_url.rstrip("/")

    # 2. Hugging Face Space environment variable
    space_host = os.getenv("SPACE_HOST")
    if space_host:
        if not space_host.startswith("http://") and not space_host.startswith("https://"):
            return f"https://{space_host}"
        return space_host.rstrip("/")

    # 3. Local Gradio config fallback
    try:
        from core.settings import GRADIO_SERVER_NAME, SERVER_PORT
    except ImportError:
        GRADIO_SERVER_NAME = "127.0.0.1"
        SERVER_PORT = 7860

    server_name = os.getenv("GRADIO_SERVER_NAME", GRADIO_SERVER_NAME)
    if server_name == "0.0.0.0":
        server_name = "127.0.0.1"
    port = os.getenv("GRADIO_SERVER_PORT", str(SERVER_PORT))

    return f"http://{server_name}:{port}"


def _execute_imagegen_pipeline(task_id: str, params: dict):
    """Execute the image generation pipeline in the background and update _TASKS_DB."""
    start_time = time.time()
    try:
        _update_task(
            task_id,
            status="processing",
            progress=10,
            updated_at=int(start_time),
        )

        from core.generation_logic import sd_image_pipeline

        task_type = params["task_type"]
        model = params["model"]
        prompt = params["prompt"]

        model_defaults = _load_yaml(_MODEL_DEFAULTS_PATH)
        model_list = _load_yaml(_MODEL_LIST_PATH)
        checkpoints = model_list.get("Checkpoint", {})
        found_arch = None
        for arch_name, arch_data in checkpoints.items():
            if isinstance(arch_data, dict):
                for m in arch_data.get("models", []):
                    if m.get("display_name") == model:
                        found_arch = arch_name
                        break
            if found_arch:
                break

        arch_defaults_section = model_defaults.get(found_arch, {}) if found_arch else {}
        arch_level_defaults = arch_defaults_section.get("_defaults", {})
        model_specific_defaults = arch_defaults_section.get(model, {})
        global_defaults = model_defaults.get("Default", {})
        merged_defaults = {**global_defaults, **arch_level_defaults, **model_specific_defaults}

        steps = params.get("steps") if params.get("steps") is not None else merged_defaults.get("steps", 20)
        cfg = params.get("cfg") if params.get("cfg") is not None else merged_defaults.get("cfg", 1.0)
        sampler = params.get("sampler") or merged_defaults.get("sampler_name", "euler")
        scheduler = params.get("scheduler") or merged_defaults.get("scheduler", "simple")

        ui_inputs = {
            "task_type": task_type,
            "model_display_name": model,
            "base_model_" + task_type: model,
            "positive_prompt": prompt,
            "negative_prompt": params.get("negative_prompt", merged_defaults.get("negative_prompt", "")),
            "width": params.get("width", 1024),
            "height": params.get("height", 1024),
            "num_inference_steps": steps,
            "guidance_scale": cfg,
            "sampler": sampler,
            "scheduler": scheduler,
            "seed": params.get("seed", -1),
            "batch_size": params.get("batch_size", 1),
            "zero_gpu_duration": params.get("zero_gpu_duration"),
            "denoise": params.get("denoise", 1.0),
        }

        if "image" in params and params["image"]:
            pil_img = _parse_image_param(params["image"])
            if pil_img:
                if task_type == "img2img":
                    ui_inputs["img2img_image"] = pil_img
                    ui_inputs["img2img_denoise"] = params.get("denoise", 0.7)
                elif task_type == "inpaint":
                    ui_inputs["inpaint_image"] = pil_img
                    ui_inputs["inpaint_denoise"] = params.get("denoise", 1.0)
                elif task_type == "outpaint":
                    ui_inputs["outpaint_image"] = pil_img
                    ui_inputs["left"] = params.get("pad_left", 0)
                    ui_inputs["right"] = params.get("pad_right", 0)
                    ui_inputs["top"] = params.get("pad_top", 0)
                    ui_inputs["bottom"] = params.get("pad_bottom", 0)
                    ui_inputs["feathering"] = params.get("feathering", 10)
                elif task_type == "hires_fix":
                    ui_inputs["hires_image"] = pil_img
                    upscaler = params.get("upscaler", "nearest-exact")
                    if upscaler == "latent" or upscaler not in ["nearest-exact", "bilinear", "area", "bicubic", "bislerp"]:
                        upscaler = "nearest-exact"
                    ui_inputs["hires_upscaler"] = upscaler
                    ui_inputs["hires_scale_by"] = params.get("upscale_by", 2.0)
                    ui_inputs["hires_denoise"] = params.get("denoise", 0.55)

        chain = params.get("chain", [])
        if chain:
            lora_data = []
            embedding_data = []
            controlnet_data = []
            diffsynth_controlnet_data = []
            ipadapter_data = []
            ipadapter_images = []
            ipadapter_weights = []
            ipadapter_lora_strengths = []
            ipadapter_global_preset = params.get("ipadapter_preset") or params.get("preset")
            ipadapter_global_embeds_scaling = params.get("ipadapter_embeds_scaling") or params.get("embeds_scaling")
            ipadapter_global_combine_method = params.get("ipadapter_combine_method") or params.get("combine_method")
            ipadapter_global_final_weight = params.get("ipadapter_final_weight") or params.get("final_weight")
            flux1_ipadapter_images = []
            flux1_ipadapter_weights = []
            flux1_ipadapter_starts = []
            flux1_ipadapter_ends = []
            sd3_ipadapter_images = []
            sd3_ipadapter_weights = []
            sd3_ipadapter_starts = []
            sd3_ipadapter_ends = []
            style_images = []
            style_strengths = []
            krea2_identity_edit_data = []
            krea2_reference_edit_data = []
            krea2_controlnet_data = []
            anima_controlnet_lllite_data = []
            reference_latent_data = []
            reference_image_data = []
            joyai_reference_data = []
            boogu_edit_data = []
            qwen_image_edit_data = []
            hidream_o1_reference_data = []
            cond_prompts = []
            cond_widths = []
            cond_heights = []
            cond_xs = []
            cond_ys = []
            cond_strengths = []

            for item in chain:
                itype = item.get("injector_type")
                if itype == "lora":
                    lora_data.extend([
                        item.get("source", item.get("lora_source", "Civitai")),
                        item.get("lora_value", ""),
                        item.get("scale", 1.0),
                        None
                    ])
                elif itype == "embedding":
                    e_source = item.get("source", item.get("embedding_source", "Civitai"))
                    e_val = item.get("embedding_value", item.get("value", item.get("embedding_id", "")))
                    if e_source and e_val:
                        embedding_data.extend([
                            e_source,
                            str(e_val),
                            None
                        ])
                elif itype == "conditioning":
                    p = item.get("prompt", "")
                    if p:
                        cond_prompts.append(p)
                        cond_widths.append(int(item.get("width", 512)))
                        cond_heights.append(int(item.get("height", 512)))
                        cond_xs.append(int(item.get("x", 0)))
                        cond_ys.append(int(item.get("y", 0)))
                        cond_strengths.append(float(item.get("strength", 1.0)))
                elif itype == "controlnet":
                    cn_type = item.get("type", item.get("Type", ""))
                    cn_series = item.get("series", item.get("Series", ""))
                    cn_strength = float(item.get("strength", 1.0))
                    cn_img = _parse_image_param(item.get("image"))

                    cn_filepath = item.get("control_net_name", "None")
                    cn_raw = _load_yaml(os.path.join(_YAML_DIR, "controlnet_models.yaml")).get("ControlNet", {})

                    cn_arch_key = None
                    if found_arch:
                        arch_cfg = _load_yaml(os.path.join(_YAML_DIR, "model_architectures.yaml")).get("architectures", {})
                        cn_arch_key = arch_cfg.get(found_arch, {}).get("controlnet_key", found_arch)

                    arch_entries = []
                    if cn_arch_key and cn_arch_key in cn_raw:
                        arch_entries = cn_raw[cn_arch_key]
                    elif found_arch and found_arch in cn_raw:
                        arch_entries = cn_raw[found_arch]
                    else:
                        for val in cn_raw.values():
                            if isinstance(val, list):
                                arch_entries.extend(val)
                            elif isinstance(val, dict):
                                arch_entries.append(val)

                    if arch_entries:
                        for entry in arch_entries:
                            entry_types = entry.get("Type", [])
                            if isinstance(entry_types, str):
                                entry_types = [entry_types]
                            if not cn_type or cn_type in entry_types:
                                if not cn_series or entry.get("Series") == cn_series:
                                    cn_filepath = entry.get("Filepath", cn_filepath)
                                    if not cn_series:
                                        cn_series = entry.get("Series", "")
                                    if not cn_type and entry_types:
                                        cn_type = entry_types[0]
                                    break

                    controlnet_data.extend([
                        cn_img,
                        cn_type,
                        cn_series,
                        cn_strength,
                        cn_filepath
                    ])
                elif itype == "anima_controlnet_lllite":
                    cn_type = item.get("type", item.get("Type", ""))
                    cn_series = item.get("series", item.get("Series", ""))
                    cn_strength = float(item.get("strength", 1.0))
                    cn_start = float(item.get("start_percent", 0.0))
                    cn_end = float(item.get("end_percent", 1.0))
                    cn_img = _parse_image_param(item.get("image"))

                    cn_filepath = item.get("control_net_name", "None")
                    anima_cfg = _load_yaml(os.path.join(_YAML_DIR, "anima_controlnet_lllite_models.yaml")).get("Anima_ControlNet_Lllite", [])
                    if anima_cfg:
                        for entry in anima_cfg:
                            entry_types = entry.get("Type", [])
                            if isinstance(entry_types, str):
                                entry_types = [entry_types]
                            if not cn_type or cn_type in entry_types:
                                if not cn_series or entry.get("Series") == cn_series:
                                    cn_filepath = entry.get("Filepath", cn_filepath)
                                    if not cn_series:
                                        cn_series = entry.get("Series", "")
                                    if not cn_type and entry_types:
                                        cn_type = entry_types[0]
                                    break

                    anima_controlnet_lllite_data.extend([
                        cn_img,
                        cn_type,
                        cn_series,
                        cn_strength,
                        cn_filepath,
                        cn_start,
                        cn_end
                    ])
                elif itype == "diffsynth_controlnet":
                    cn_type = item.get("type", "")
                    cn_series = item.get("series", "")
                    cn_strength = float(item.get("strength", 1.0))
                    cn_img = _parse_image_param(item.get("image"))

                    cn_filepath = "None"
                    diffsynth_raw = _load_yaml(os.path.join(_YAML_DIR, "diffsynth_controlnet_models.yaml")).get("DiffSynth_ControlNet", {})
                    diffsynth_entries = []
                    if isinstance(diffsynth_raw, dict):
                        for val in diffsynth_raw.values():
                            if isinstance(val, list):
                                diffsynth_entries.extend(val)
                            elif isinstance(val, dict):
                                diffsynth_entries.append(val)
                    elif isinstance(diffsynth_raw, list):
                        diffsynth_entries = diffsynth_raw

                    if diffsynth_entries:
                        for entry in diffsynth_entries:
                            if not cn_type or cn_type in entry.get("Type", []):
                                if not cn_series or entry.get("Series") == cn_series:
                                    cn_filepath = entry.get("Filepath", cn_filepath)
                                    if not cn_series:
                                        cn_series = entry.get("Series", cn_series)
                                    if not cn_type and entry.get("Type"):
                                        cn_type = entry.get("Type")[0]
                                    break

                    diffsynth_controlnet_data.extend([
                        cn_img,
                        cn_type,
                        cn_series,
                        cn_strength,
                        cn_filepath
                    ])
                elif itype == "krea2_controlnet":
                    cn_type = item.get("type", "Depth")
                    cn_series = item.get("series", "Patil")
                    cn_strength = float(item.get("strength", 1.0))
                    cn_img = _parse_image_param(item.get("image"))

                    cn_filepath = "depth-control-lora.safetensors"
                    krea2_cfg = _load_yaml(os.path.join(_YAML_DIR, "krea2_controlnet_models.yaml")).get("Krea2_ControlNet", [])
                    if krea2_cfg:
                        for entry in krea2_cfg:
                            if cn_type in entry.get("Type", []):
                                if not cn_series or entry.get("Series") == cn_series:
                                    cn_filepath = entry.get("Filepath", cn_filepath)
                                    cn_series = entry.get("Series", cn_series)
                                    break

                    krea2_controlnet_data.extend([
                        cn_img,
                        cn_type,
                        cn_series,
                        cn_strength,
                        cn_filepath
                    ])
                elif itype == "flux1_ipadapter":
                    if len(flux1_ipadapter_images) < 5:
                        img = _parse_image_param(item.get("image"))
                        weight = float(item.get("weight", 1.0))
                        start_at = float(item.get("start_at", item.get("start_percent", item.get("start", 0.0))))
                        end_at = float(item.get("end_at", item.get("end_percent", item.get("end", 1.0))))
                        flux1_ipadapter_images.append(img)
                        flux1_ipadapter_weights.append(weight)
                        flux1_ipadapter_starts.append(start_at)
                        flux1_ipadapter_ends.append(end_at)
                elif itype == "sd3_ipadapter":
                    if len(sd3_ipadapter_images) < 5:
                        img = _parse_image_param(item.get("image"))
                        weight = float(item.get("weight", 1.0))
                        start_at = float(item.get("start_at", item.get("start_percent", item.get("start", 0.0))))
                        end_at = float(item.get("end_at", item.get("end_percent", item.get("end", 1.0))))
                        sd3_ipadapter_images.append(img)
                        sd3_ipadapter_weights.append(weight)
                        sd3_ipadapter_starts.append(start_at)
                        sd3_ipadapter_ends.append(end_at)
                elif itype == "ipadapter":
                    if len(ipadapter_images) < 5:
                        img = _parse_image_param(item.get("image"))
                        weight = float(item.get("weight", 1.0))
                        lora_str = float(item.get("lora_strength", 0.6))
                        ipadapter_images.append(img)
                        ipadapter_weights.append(weight)
                        ipadapter_lora_strengths.append(lora_str)

                        if "preset" in item and not ipadapter_global_preset:
                            ipadapter_global_preset = item["preset"]
                        if "embeds_scaling" in item and not ipadapter_global_embeds_scaling:
                            ipadapter_global_embeds_scaling = item["embeds_scaling"]
                        if "combine_method" in item and not ipadapter_global_combine_method:
                            ipadapter_global_combine_method = item["combine_method"]
                        if "final_weight" in item and ipadapter_global_final_weight is None:
                            ipadapter_global_final_weight = float(item["final_weight"])
                elif itype in ("style", "flux1_style"):
                    img = _parse_image_param(item.get("image"))
                    if img:
                        style_images.append(img)
                        style_strengths.append(float(item.get("strength", item.get("weight", 1.0))))
                elif itype == "pid":
                    is_enabled = item.get("enabled", True)
                    if isinstance(is_enabled, str):
                        is_enabled = is_enabled.upper() in ("ON", "TRUE", "1")
                    ui_inputs["pid_settings"] = "ON" if is_enabled else "OFF"
                elif itype == "krea2_identity_edit":
                    img = _parse_image_param(item.get("image"))
                    if img:
                        krea2_identity_edit_data.append(img)
                elif itype == "krea2_style_reference":
                    img = _parse_image_param(item.get("image"))
                    if img:
                        krea2_reference_edit_data.append(img)
                elif itype in ("reference_latent", "reference_edit"):
                    img = _parse_image_param(item.get("image"))
                    if img:
                        reference_latent_data.append(img)
                elif itype in ("reference_image", "mage_flow_reference_edit"):
                    img = _parse_image_param(item.get("image"))
                    if img:
                        reference_image_data.append(img)
                elif itype in ("joyai_image", "joyai_reference_edit"):
                    img = _parse_image_param(item.get("image"))
                    if img:
                        joyai_reference_data.append(img)
                elif itype in ("boogu_image_edit", "boogu_edit"):
                    img = _parse_image_param(item.get("image"))
                    if img:
                        boogu_edit_data.append(img)
                elif itype == "qwen_image_edit":
                    img = _parse_image_param(item.get("image"))
                    if img:
                        qwen_image_edit_data.append(img)
                elif itype == "hidream_o1_reference":
                    img = _parse_image_param(item.get("image"))
                    if img:
                        hidream_o1_reference_data.append(img)
                elif itype == "vae":
                    v_source = item.get("source", item.get("vae_source", "Civitai"))
                    v_val = item.get("vae_value", item.get("value", item.get("vae_id", item.get("vae_name", ""))))
                    if v_source and v_val:
                        ui_inputs["vae_source"] = v_source
                        ui_inputs["vae_id"] = str(v_val)

            if lora_data: ui_inputs["lora_data"] = lora_data
            if embedding_data: ui_inputs["embedding_data"] = embedding_data
            if controlnet_data: ui_inputs["controlnet_data"] = controlnet_data
            if anima_controlnet_lllite_data: ui_inputs["anima_controlnet_lllite_data"] = anima_controlnet_lllite_data
            if diffsynth_controlnet_data: ui_inputs["diffsynth_controlnet_data"] = diffsynth_controlnet_data
            if krea2_controlnet_data: ui_inputs["krea2_controlnet_data"] = krea2_controlnet_data
            if ipadapter_images:
                preset = ipadapter_global_preset or "STANDARD (medium strength)"
                embeds_scaling = ipadapter_global_embeds_scaling or "V only"
                combine_method = ipadapter_global_combine_method or "concat"
                final_weight = float(ipadapter_global_final_weight) if ipadapter_global_final_weight is not None else 1.0
                final_lora_strength = 0.6

                presets_by_arch = _get_ipadapter_presets_by_arch()
                target_arch = "SD1.5" if found_arch in ("sd15", "SD1.5") else "SDXL"
                allowed_presets = presets_by_arch.get(target_arch, [])

                if preset not in allowed_presets:
                    raise ValueError(
                        f"Invalid IPAdapter preset '{preset}' for model architecture '{target_arch}'. "
                        f"Preset must match the target model architecture. Allowed presets for {target_arch}: {allowed_presets}"
                    )

                ui_inputs["ipadapter_data"] = (
                    ipadapter_images + ipadapter_weights + ipadapter_lora_strengths +
                    [preset, final_weight, final_lora_strength, embeds_scaling, combine_method]
                )
            elif ipadapter_data:
                ui_inputs["ipadapter_data"] = ipadapter_data
            if flux1_ipadapter_images:
                ui_inputs["flux1_ipadapter_data"] = (
                    flux1_ipadapter_images + flux1_ipadapter_weights + flux1_ipadapter_starts + flux1_ipadapter_ends
                )
            if sd3_ipadapter_images:
                ui_inputs["sd3_ipadapter_chain"] = (
                    sd3_ipadapter_images + sd3_ipadapter_weights + sd3_ipadapter_starts + sd3_ipadapter_ends
                )
            if style_images: ui_inputs["style_data"] = style_images + style_strengths
            if krea2_identity_edit_data: ui_inputs["krea2_identity_edit_data"] = krea2_identity_edit_data
            if krea2_reference_edit_data: ui_inputs["krea2_reference_edit_data"] = krea2_reference_edit_data
            if reference_latent_data: ui_inputs["reference_latent_data"] = reference_latent_data
            if reference_image_data: ui_inputs["reference_image_data"] = reference_image_data
            if joyai_reference_data: ui_inputs["joyai_reference_data"] = joyai_reference_data
            if boogu_edit_data: ui_inputs["boogu_edit_data"] = boogu_edit_data
            if qwen_image_edit_data: ui_inputs["qwen_image_edit_data"] = qwen_image_edit_data
            if hidream_o1_reference_data: ui_inputs["hidream_o1_reference_data"] = hidream_o1_reference_data
            if cond_prompts:
                ui_inputs["conditioning_data"] = (
                    cond_prompts + cond_widths + cond_heights + cond_xs + cond_ys + cond_strengths
                )

        if "vae_source" in params and "vae_id" in params:
            ui_inputs["vae_source"] = params["vae_source"]
            ui_inputs["vae_id"] = str(params["vae_id"])

        pid_val = params.get("pid") if params.get("pid") is not None else params.get("pid_settings")
        if pid_val is not None:
            if isinstance(pid_val, bool):
                ui_inputs["pid_settings"] = "ON" if pid_val else "OFF"
            elif str(pid_val).upper() in ("ON", "TRUE", "1"):
                ui_inputs["pid_settings"] = "ON"
            else:
                ui_inputs["pid_settings"] = "OFF"

        _update_task(task_id, progress=50)

        # Execute Pipeline
        output = sd_image_pipeline.run(ui_inputs=ui_inputs, progress=DummyProgress())

        try:
            from core.settings import OUTPUT_DIR
        except ImportError:
            OUTPUT_DIR = os.path.join(_PROJECT_ROOT, "output")

        os.makedirs(OUTPUT_DIR, exist_ok=True)

        import tempfile
        import gradio.processing_utils as pu

        gradio_cache_dir = os.path.join(tempfile.gettempdir(), "gradio")
        os.makedirs(gradio_cache_dir, exist_ok=True)

        base_url = _get_public_base_url()
        images = []
        raw_list = output if isinstance(output, list) else ([output] if output else [])
        for idx, item in enumerate(raw_list):
            target_path = None
            if hasattr(item, "save"):  # PIL Image
                filename = f"mcp_{task_id}_{idx}.png"
                filepath = os.path.join(OUTPUT_DIR, filename)
                item.save(filepath)
                target_path = filepath
            elif isinstance(item, str) and os.path.exists(item):
                target_path = item

            if target_path:
                try:
                    cached_path = pu.save_file_to_cache(target_path, cache_dir=gradio_cache_dir)
                    abs_path = os.path.abspath(cached_path).replace("\\", "/")
                except Exception as e:
                    print(f"Warning: Failed to cache image file to Gradio temp dir: {e}")
                    abs_path = os.path.abspath(target_path).replace("\\", "/")

                url = f"{base_url}/gradio_api/file={urllib.parse.quote(abs_path)}"
                images.append(url)
            elif item:
                images.append(str(item))

        execution_time = round(time.time() - start_time, 2)
        _update_task(
            task_id,
            status="completed",
            progress=100,
            completed_at=int(time.time()),
            result={
                "images": images,
                "seed": ui_inputs.get("seed", params.get("seed", -1)),
                "width": ui_inputs.get("width", params.get("width", 1024)),
                "height": ui_inputs.get("height", params.get("height", 1024)),
                "execution_time_seconds": execution_time,
            },
        )

    except Exception as e:
        _update_task(
            task_id,
            status="failed",
            progress=0,
            failed_at=int(time.time()),
            error={"code": "EXECUTION_ERROR", "message": str(e)},
        )
