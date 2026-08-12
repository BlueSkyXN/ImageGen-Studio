import os
import requests
import hashlib
import re
import threading
from functools import wraps
from typing import Sequence, Mapping, Any, Union, Set
from pathlib import Path
import shutil

import gradio as gr
from huggingface_hub import hf_hub_download, constants as hf_constants
import torch
import numpy as np
from PIL import Image, ImageChops
import yaml

from core.settings import *
from core.runtime_config import CONFIG

IPADAPTER_PRESETS = None
_DOWNLOAD_LOCK = threading.RLock()


def _download_serialized(function):
    """Protect shared model directories and symlink creation in one process."""

    @wraps(function)
    def wrapped(*args, **kwargs):
        with _DOWNLOAD_LOCK:
            return function(*args, **kwargs)

    return wrapped

class UniqueKeyLoader(yaml.SafeLoader):
    """
    A custom YAML loader that handles duplicate keys by grouping their values into a list.
    """
    def construct_mapping(self, node, deep=False):
        mapping = []
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            value = self.construct_object(value_node, deep=deep)
            mapping.append((key, value))

        result = {}
        for k, v in mapping:
            if k in result:
                if isinstance(result[k], list):
                    result[k].append(v)
                else:
                    result[k] = [result[k], v]
            else:
                result[k] = v
        return result

UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, UniqueKeyLoader.construct_mapping)

@_download_serialized
def save_uploaded_file_with_hash(file_obj: gr.File, target_dir: str) -> str:
    if not file_obj:
        return ""

    temp_path = file_obj.name

    sha256 = hashlib.sha256()
    with open(temp_path, 'rb') as f:
        for block in iter(lambda: f.read(65536), b''):
            sha256.update(block)

    file_hash = sha256.hexdigest()
    _, extension = os.path.splitext(temp_path)
    hashed_filename = f"{file_hash}{extension.lower()}"

    dest_path = os.path.join(target_dir, hashed_filename)

    os.makedirs(target_dir, exist_ok=True)
    if not os.path.exists(dest_path):
        shutil.copy(temp_path, dest_path)
        print(f"✅ Saved uploaded file as: {dest_path}")
    else:
        print(f"ℹ️ File already exists (deduplicated): {dest_path}")

    return hashed_filename

def bytes_to_gb(byte_size: int) -> float:
    if byte_size is None or byte_size == 0:
        return 0.0
    return round(byte_size / (1024 ** 3), 2)

def get_directory_size(path: str) -> int:
    total_size = 0
    if not os.path.exists(path):
        return 0
    try:
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if os.path.isfile(fp) and not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
    except OSError as e:
        print(f"Warning: Could not access {path} to calculate size: {e}")
    return total_size


def _existing_disk_path(path: str) -> str:
    """Find an existing ancestor suitable for ``shutil.disk_usage``."""

    candidate = os.path.abspath(os.path.expanduser(path))
    while not os.path.exists(candidate):
        parent = os.path.dirname(candidate)
        if parent == candidate:
            return os.getcwd()
        candidate = parent
    return candidate


def _assert_download_space(
    expected_bytes: int | None, filename: str, target_path: str
) -> None:
    """Reject a download before HF Space exhausts its ephemeral filesystem."""

    free_bytes = shutil.disk_usage(_existing_disk_path(target_path)).free
    reserve_bytes = int(CONFIG.min_free_disk_gb * 1024**3)
    needed_bytes = max(0, int(expected_bytes or 0)) + reserve_bytes
    if free_bytes < needed_bytes:
        expected_text = (
            f"，该文件约 {bytes_to_gb(expected_bytes):g} GB"
            if expected_bytes
            else ""
        )
        raise gr.Error(
            f"磁盘空间不足，无法下载“{filename}”{expected_text}。"
            f"当前剩余 {bytes_to_gb(free_bytes):g} GB，系统需保留 "
            f"{CONFIG.min_free_disk_gb:g} GB。请删除缓存、挂载持久存储或换用较小模型。"
        )


def _hf_remote_file_size(repo_id: str, filename: str) -> int | None:
    """Read the Hub file size without downloading its body."""

    try:
        from huggingface_hub import get_hf_file_metadata, hf_hub_url

        metadata = get_hf_file_metadata(
            hf_hub_url(repo_id=repo_id, filename=filename),
            token=os.environ.get("HF_TOKEN"),
        )
        return int(metadata.size) if metadata.size is not None else None
    except Exception as exc:
        # The actual download will surface access/network errors with context.
        print(f"Warning: Could not preflight Hub file size for {repo_id}/{filename}: {exc}")
        return None

def get_value_at_index(obj: Union[Sequence, Mapping], index: int) -> Any:
    try:
        return obj[index]
    except (KeyError, IndexError):
        try:
            return obj["result"][index]
        except (KeyError, IndexError):
            return None

def sanitize_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        return ""
    return "".join(char for char in prompt if char.isprintable() or char in ('\n', '\t'))

def sanitize_id(input_id: str) -> str:
    if not isinstance(input_id, str):
        return ""
    input_id = input_id.strip()
    if "civitai" in input_id.lower():
        version_match = re.search(r'modelVersionId=(\d+)', input_id)
        if version_match:
            return version_match.group(1)
        model_match = re.search(r'/models/(\d+)', input_id)
        if model_match:
            return model_match.group(1)
    return re.sub(r'[^0-9]', '', input_id)

def sanitize_url(url: str) -> str:
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")
    url = url.strip()
    if not re.match(r'^https?://[^\s/$.?#].[^\s]*$', url):
        raise ValueError("Invalid URL format or scheme. Only HTTP and HTTPS are allowed.")
    return url

def sanitize_filename(filename: str) -> str:
    if not isinstance(filename, str):
        return ""
    sanitized = filename.replace('..', '')
    sanitized = re.sub(r'[^\w\.\-]', '_', sanitized)
    return sanitized.lstrip('/\\')

def get_civitai_file_info(version_id: str) -> dict | None:
    api_url = f"https://civitai.com/api/v1/model-versions/{version_id}"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()

        model_type = data.get('model', {}).get('type')

        result_file = None
        for file_data in data.get('files', []):
            if file_data.get('type') == 'Model' and file_data['name'].endswith(('.safetensors', '.pt', '.bin')):
                result_file = file_data.copy()
                break

        if not result_file and data.get('files'):
            result_file = data['files'][0].copy()

        if result_file:
            result_file['model_type'] = model_type
            return result_file
    except Exception:
        return None

def download_file(url: str, save_path: str, api_key: str = None, progress=None, desc: str = "") -> str:
    if os.path.exists(save_path):
        return f"File already exists: {os.path.basename(save_path)}"

    headers = {'Authorization': f'Bearer {api_key}'} if api_key and api_key.strip() else {}
    try:
        if progress:
            progress(0, desc=desc)

        response = requests.get(url, stream=True, headers=headers, timeout=15)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))

        with open(save_path, "wb") as f:
            downloaded = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                if progress and total_size > 0:
                    downloaded += len(chunk)
                    progress(downloaded / total_size, desc=desc)
        return f"Successfully downloaded: {os.path.basename(save_path)}"
    except Exception as e:
        if os.path.exists(save_path):
            os.remove(save_path)
        return f"Download failed for {os.path.basename(save_path)}: {e}"

@_download_serialized
def get_lora_path(source: str, id_or_url: str, civitai_key: str, progress) -> tuple[str | None, str]:
    if not id_or_url or not id_or_url.strip():
        return None, "No ID/URL provided."

    try:
        if source == "Civitai":
            version_id = sanitize_id(id_or_url)
            if not version_id:
                return None, "Invalid Civitai ID provided. Must be numeric."

            file_info = get_civitai_file_info(version_id)
            if file_info:
                model_type = file_info.get('model_type')
                if model_type and model_type.lower() == 'checkpoint':
                    return None, f"Invalid Civitai model type '{model_type}' for LoRA. Checkpoint models are not allowed."

            filename = sanitize_filename(f"civitai_{version_id}.safetensors")
            local_path = os.path.join(LORA_DIR, filename)
            api_key_to_use = civitai_key
            source_name = f"Civitai ID {version_id}"
        elif source == "Hugging Face":
            parts = id_or_url.strip().split('/')
            if len(parts) < 3:
                return None, "Invalid Hugging Face path. Format: repo_owner/repo_name/filename"
            repo_id = f"{parts[0]}/{parts[1]}"
            repo_file_path = "/".join(parts[2:])
            unique_name = id_or_url.strip().replace('/', '_')
            filename = sanitize_filename(unique_name)
            local_path = os.path.join(LORA_DIR, filename)
            source_name = f"HF {repo_file_path}"
        else:
            return None, "Invalid source."

    except ValueError as e:
        return None, f"Input validation failed: {e}"

    if os.path.lexists(local_path):
        if not os.path.exists(local_path):
            os.remove(local_path)
        else:
            return local_path, "File already exists."

    if source == "Civitai":
        if not file_info or not file_info.get('downloadUrl'):
            return None, f"Could not get download link for {source_name}."

        status = download_file(file_info['downloadUrl'], local_path, api_key_to_use, progress=progress, desc=f"Downloading {source_name}")
        return (local_path, status) if "Successfully" in status else (None, status)
    elif source == "Hugging Face":
        try:
            if progress: progress(0, desc=f"Downloading {source_name}")
            cached_path = hf_hub_download(repo_id=repo_id, filename=repo_file_path, token=os.environ.get("HF_TOKEN"))
            os.makedirs(LORA_DIR, exist_ok=True)
            os.symlink(cached_path, local_path)
            if progress: progress(1.0, desc=f"Downloaded {source_name}")
            return local_path, f"Successfully downloaded: {filename}"
        except Exception as e:
            return None, f"Hugging Face download failed: {e}"

@_download_serialized
def get_embedding_path(source: str, id_or_url: str, civitai_key: str, progress) -> tuple[str | None, str]:
    if not id_or_url or not id_or_url.strip():
        return None, "No ID/URL provided."

    try:
        if source == "Civitai":
            version_id = sanitize_id(id_or_url)
            if not version_id:
                return None, "Invalid Civitai ID. Must be numeric."

            file_info = get_civitai_file_info(version_id)
            if file_info:
                model_type = file_info.get('model_type')
                if model_type and model_type.lower() == 'checkpoint':
                    return None, f"Invalid Civitai model type '{model_type}' for Embedding. Checkpoint models are not allowed."

            file_ext = ".safetensors"
            if file_info and file_info.get('name') and file_info['name'].lower().endswith(('.pt', '.bin')):
                file_ext = os.path.splitext(file_info['name'])[1]

            filename = sanitize_filename(f"civitai_{version_id}{file_ext}")
            local_path = os.path.join(EMBEDDING_DIR, filename)
            api_key_to_use = civitai_key
            source_name = f"Embedding Civitai ID {version_id}"
        elif source == "Hugging Face":
            parts = id_or_url.strip().split('/')
            if len(parts) < 3:
                return None, "Invalid Hugging Face path. Format: repo_owner/repo_name/filename"
            repo_id = f"{parts[0]}/{parts[1]}"
            repo_file_path = "/".join(parts[2:])
            filename = sanitize_filename(parts[-1])
            local_path = os.path.join(EMBEDDING_DIR, filename)
            source_name = f"Embedding HF {repo_file_path}"
        else:
            return None, "Invalid source."

    except ValueError as e:
        return None, f"Input validation failed: {e}"

    if os.path.lexists(local_path):
        if not os.path.exists(local_path):
            os.remove(local_path)
        else:
            return local_path, "File already exists."

    if source == "Civitai":
        if not file_info or not file_info.get('downloadUrl'):
            return None, f"Could not get download link for {source_name}."

        status = download_file(file_info['downloadUrl'], local_path, api_key_to_use, progress=progress, desc=f"Downloading {source_name}")
        return (local_path, status) if "Successfully" in status else (None, status)
    elif source == "Hugging Face":
        try:
            if progress: progress(0, desc=f"Downloading {source_name}")
            cached_path = hf_hub_download(repo_id=repo_id, filename=repo_file_path, token=os.environ.get("HF_TOKEN"))
            os.makedirs(EMBEDDING_DIR, exist_ok=True)
            os.symlink(cached_path, local_path)
            if progress: progress(1.0, desc=f"Downloaded {source_name}")
            return local_path, f"Successfully downloaded: {filename}"
        except Exception as e:
            return None, f"Hugging Face download failed: {e}"

@_download_serialized
def get_vae_path(source: str, id_or_url: str, civitai_key: str, progress) -> tuple[str | None, str]:
    if not id_or_url or not id_or_url.strip():
        return None, "No ID/URL provided."

    try:
        if source == "Civitai":
            version_id = sanitize_id(id_or_url)
            if not version_id:
                return None, "Invalid Civitai ID. Must be numeric."

            file_info = get_civitai_file_info(version_id)
            if file_info:
                model_type = file_info.get('model_type')
                if model_type and model_type.lower() == 'checkpoint':
                    return None, f"Invalid Civitai model type '{model_type}' for VAE. Checkpoint models are not allowed."

            file_ext = ".safetensors"
            if file_info and file_info.get('name') and file_info['name'].lower().endswith(('.pt', '.bin')):
                file_ext = os.path.splitext(file_info['name'])[1]

            filename = sanitize_filename(f"civitai_{version_id}{file_ext}")
            local_path = os.path.join(VAE_DIR, filename)
            api_key_to_use = civitai_key
            source_name = f"VAE Civitai ID {version_id}"
        elif source == "Hugging Face":
            parts = id_or_url.strip().split('/')
            if len(parts) < 3:
                return None, "Invalid Hugging Face path. Format: repo_owner/repo_name/filename"
            repo_id = f"{parts[0]}/{parts[1]}"
            repo_file_path = "/".join(parts[2:])
            unique_name = id_or_url.strip().replace('/', '_')
            filename = sanitize_filename(unique_name)
            local_path = os.path.join(VAE_DIR, filename)
            source_name = f"VAE HF {repo_file_path}"
        else:
            return None, "Invalid source."

    except ValueError as e:
        return None, f"Input validation failed: {e}"

    if os.path.lexists(local_path):
        if not os.path.exists(local_path):
            os.remove(local_path)
        else:
            return local_path, "File already exists."

    if source == "Civitai":
        if not file_info or not file_info.get('downloadUrl'):
            return None, f"Could not get download link for {source_name}."

        status = download_file(file_info['downloadUrl'], local_path, api_key_to_use, progress=progress, desc=f"Downloading {source_name}")
        return (local_path, status) if "Successfully" in status else (None, status)
    elif source == "Hugging Face":
        try:
            if progress: progress(0, desc=f"Downloading {source_name}")
            cached_path = hf_hub_download(repo_id=repo_id, filename=repo_file_path, token=os.environ.get("HF_TOKEN"))
            os.makedirs(VAE_DIR, exist_ok=True)
            os.symlink(cached_path, local_path)
            if progress: progress(1.0, desc=f"Downloaded {source_name}")
            return local_path, f"Successfully downloaded: {filename}"
        except Exception as e:
            return None, f"Hugging Face download failed: {e}"

@_download_serialized
def _ensure_model_downloaded(display_name: str, progress=gr.Progress()):
    if display_name not in ALL_MODEL_MAP:
        raise ValueError(f"Model '{display_name}' not found in configuration.")

    model_info = ALL_MODEL_MAP[display_name]
    repo_filename = model_info[1]
    base_filename = os.path.basename(repo_filename)

    download_info = ALL_FILE_DOWNLOAD_MAP.get(base_filename)
    if not download_info:
        raise gr.Error(f"配置中找不到模型文件“{base_filename}”，无法下载。")

    category = download_info.get("category")
    dest_dir = CATEGORY_TO_DIR_MAP.get(category)

    if not dest_dir:
        raise ValueError(f"Unknown YAML category '{category}' for '{base_filename}'.")

    dest_path = os.path.join(dest_dir, base_filename)

    if os.path.lexists(dest_path):
        if not os.path.exists(dest_path):
            print(f"⚠️ Found and removed broken symlink: {dest_path}")
            os.remove(dest_path)
        else:
            return base_filename

    source = download_info.get("source")
    try:
        progress(0, desc=f"Downloading: {base_filename}")

        if source == "hf":
            repo_id = download_info.get("repo_id")
            hf_filename = download_info.get("repository_file_path", base_filename)
            if not repo_id:
                raise ValueError(f"repo_id is missing for HF model '{base_filename}'")

            # Re-link an existing Hub cache entry without requiring extra free
            # space.  Only preflight when bytes really need to be downloaded.
            try:
                cached_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_filename,
                    token=os.environ.get("HF_TOKEN"),
                    local_files_only=True,
                )
            except Exception:
                cached_path = None
            if cached_path is None:
                expected_bytes = _hf_remote_file_size(repo_id, hf_filename)
                _assert_download_space(
                    expected_bytes,
                    base_filename,
                    hf_constants.HF_HUB_CACHE,
                )
                cached_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=hf_filename,
                    token=os.environ.get("HF_TOKEN"),
                )
            os.makedirs(dest_dir, exist_ok=True)
            os.symlink(cached_path, dest_path)
            print(f"✅ Symlinked '{cached_path}' to '{dest_path}'")

        elif source == "civitai":
            model_version_id = download_info.get("model_version_id")
            if not model_version_id:
                raise ValueError(f"model_version_id is missing for Civitai model '{base_filename}'")

            file_info = get_civitai_file_info(model_version_id)
            if not file_info or not file_info.get('downloadUrl'):
                raise ConnectionError(f"Could not get download URL for Civitai model version ID {model_version_id}")

            expected_bytes = None
            if file_info.get("sizeKB") is not None:
                expected_bytes = int(float(file_info["sizeKB"]) * 1024)
            _assert_download_space(expected_bytes, base_filename, dest_dir)
            status = download_file(
                file_info['downloadUrl'], dest_path, api_key=os.environ.get("CIVITAI_API_KEY", ""), progress=progress, desc=f"Downloading: {base_filename}"
            )
            if "Failed" in status:
                raise ConnectionError(status)
        else:
            raise NotImplementedError(f"Download source '{source}' is not implemented for '{base_filename}'")

        progress(1.0, desc=f"Downloaded: {base_filename}")

    except Exception as e:
        if os.path.lexists(dest_path):
            try:
                os.remove(dest_path)
            except OSError: pass
        raise gr.Error(f"模型“{display_name}”下载或链接失败：{e}")

    return base_filename

@_download_serialized
def ensure_controlnet_model_downloaded(filename: str, progress):
    if not filename or filename == "None":
        return

    download_info = ALL_FILE_DOWNLOAD_MAP.get(filename)
    if not download_info:
        raise gr.Error(f"配置中找不到 ControlNet 模型“{filename}”，无法下载。")

    category = download_info.get("category", "controlnet")
    dest_dir = CATEGORY_TO_DIR_MAP.get(category, CONTROLNET_DIR)
    dest_path = os.path.join(dest_dir, filename)

    if os.path.lexists(dest_path):
        if not os.path.exists(dest_path):
            print(f"⚠️ Found and removed broken symlink: {dest_path}")
            os.remove(dest_path)
        else:
            return

    source = download_info.get("source")

    try:
        if source == "hf":
            repo_id = download_info.get("repo_id")
            repo_filename = download_info.get("repository_file_path", filename)
            if not repo_id:
                raise ValueError("repo_id is missing for Hugging Face download.")

            progress(0, desc=f"Downloading CN: {filename}")
            cached_path = hf_hub_download(repo_id=repo_id, filename=repo_filename, token=os.environ.get("HF_TOKEN"))
            os.makedirs(dest_dir, exist_ok=True)
            os.symlink(cached_path, dest_path)
            print(f"✅ Symlinked ControlNet '{cached_path}' to '{dest_path}'")
            progress(1.0, desc=f"Downloaded CN: {filename}")

        elif source == "civitai":
            model_version_id = download_info.get("model_version_id")
            if not model_version_id:
                raise ValueError("model_version_id is missing for Civitai download.")

            file_info = get_civitai_file_info(model_version_id)
            if not file_info or not file_info.get('downloadUrl'):
                raise ConnectionError(f"Could not get download URL for Civitai model version ID {model_version_id}")

            status = download_file(
                file_info['downloadUrl'],
                dest_path,
                api_key=os.environ.get("CIVITAI_API_KEY", ""),
                progress=progress,
                desc=f"Downloading CN: {filename}"
            )
            if "Failed" in status:
                raise ConnectionError(status)
        else:
            raise NotImplementedError(f"Download source '{source}' is not implemented for ControlNets.")

    except Exception as e:
        if os.path.lexists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        raise gr.Error(f"ControlNet 模型“{filename}”下载失败：{e}")

@_download_serialized
def ensure_file_downloaded(filename: str, progress=None):
    if not filename or filename == "None":
        return

    download_info = ALL_FILE_DOWNLOAD_MAP.get(filename)
    if not download_info:
        print(f"⚠️ Warning: File '{filename}' not found in configuration (file_list.yaml). Cannot download.")
        return

    category = download_info.get("category", "loras")
    dest_dir = CATEGORY_TO_DIR_MAP.get(category, LORA_DIR)
    dest_path = os.path.join(dest_dir, filename)

    if os.path.lexists(dest_path):
        if not os.path.exists(dest_path):
            print(f"⚠️ Found and removed broken symlink: {dest_path}")
            os.remove(dest_path)
        else:
            return

    source = download_info.get("source")
    try:
        if source == "hf":
            repo_id = download_info.get("repo_id")
            repo_filename = download_info.get("repository_file_path", filename)
            if not repo_id:
                raise ValueError("repo_id is missing for Hugging Face download.")

            if progress and callable(progress):
                progress(0, desc=f"Downloading: {filename}")
            cached_path = hf_hub_download(repo_id=repo_id, filename=repo_filename, token=os.environ.get("HF_TOKEN"))
            os.makedirs(dest_dir, exist_ok=True)
            os.symlink(cached_path, dest_path)
            print(f"✅ Symlinked '{cached_path}' to '{dest_path}'")
            if progress and callable(progress):
                progress(1.0, desc=f"Downloaded: {filename}")

        elif source == "civitai":
            model_version_id = download_info.get("model_version_id")
            if not model_version_id:
                raise ValueError("model_version_id is missing for Civitai download.")

            file_info = get_civitai_file_info(model_version_id)
            if not file_info or not file_info.get('downloadUrl'):
                raise ConnectionError(f"Could not get download URL for Civitai model version ID {model_version_id}")

            status = download_file(
                file_info['downloadUrl'],
                dest_path,
                api_key=os.environ.get("CIVITAI_API_KEY", ""),
                progress=progress,
                desc=f"Downloading: {filename}"
            )
            if "Failed" in status:
                raise ConnectionError(status)
        else:
            raise NotImplementedError(f"Download source '{source}' is not implemented for '{filename}'.")

    except Exception as e:
        if os.path.lexists(dest_path):
            try:
                os.remove(dest_path)
            except OSError:
                pass
        raise gr.Error(f"文件“{filename}”下载失败：{e}")

def load_ipadapter_presets():
    global IPADAPTER_PRESETS
    if IPADAPTER_PRESETS is not None:
        return

    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _IPADAPTER_MODELS_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'ipadapter_models.yaml')

    try:
        with open(_IPADAPTER_MODELS_PATH, 'r', encoding='utf-8') as f:
            presets_list = yaml.load(f, Loader=UniqueKeyLoader)

        IPADAPTER_PRESETS = {item['preset_name']: item for item in presets_list}
        print("✅ IPAdapter presets loaded successfully.")
    except Exception as e:
        print(f"❌ FATAL: Could not load or parse ipadapter_models.yaml. IPAdapter will not work. Error: {e}")
        IPADAPTER_PRESETS = {}

@_download_serialized
def ensure_ipadapter_models_downloaded(preset_name: str, progress):
    if not preset_name:
        return

    if IPADAPTER_PRESETS is None:
        raise RuntimeError("IPAdapter presets have not been loaded. `load_ipadapter_presets` must be called on startup.")

    preset_info = IPADAPTER_PRESETS.get(preset_name)
    if not preset_info:
        print(f"⚠️ Warning: IPAdapter preset '{preset_name}' not found in configuration. Skipping download.")
        return

    model_files_to_check = []

    def add_files(value, type_name):
        if not value: return
        if isinstance(value, list):
            for v in value:
                model_files_to_check.append((v, type_name))
        else:
            model_files_to_check.append((value, type_name))

    add_files(preset_info.get('clip_vision'), 'CLIP_VISION')
    add_files(preset_info.get('ipadapter'), 'IPADAPTER')
    add_files(preset_info.get('loras'), 'LORA')

    for filename, model_type in model_files_to_check:
        if not filename:
            continue

        temp_display_name = f"ipadapter_asset_{filename}"

        if temp_display_name not in ALL_MODEL_MAP:
             ALL_MODEL_MAP[temp_display_name] = (None, filename, model_type, None, None)

        try:
            _ensure_model_downloaded(temp_display_name, progress)
        except Exception as e:
            print(f"❌ Error ensuring download for IPAdapter asset '{filename}': {e}")

@_download_serialized
def ensure_sd3_ipadapter_models_downloaded(progress):
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    yaml_path = os.path.join(_PROJECT_ROOT, 'yaml', 'ipadapter_sd3_models.yaml')
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            sd3_models = yaml.safe_load(f)
        if sd3_models:
            if 'ipadapter' in sd3_models:
                _ensure_model_downloaded(sd3_models['ipadapter'], progress)
            if 'clip_vision' in sd3_models:
                _ensure_model_downloaded(sd3_models['clip_vision'], progress)
    except Exception as e:
        print(f"Warning: Failed to load or download sd3 ipadapter models: {e}")

def get_model_generation_defaults(model_display_name: str, model_type: str, defaults_config: dict):
    final_defaults = {
        'steps': 25, 'cfg': 7.0, 'sampler_name': 'euler', 'scheduler': 'simple',
        'positive_prompt': '', 'negative_prompt': ''
    }

    if 'Default' in defaults_config:
        final_defaults.update(defaults_config['Default'])

    model_type_key = next((key for key in defaults_config if key.lower().replace(" ", "-").replace(".", "") == model_type.lower()), None)
    if model_type_key:
        model_type_config = defaults_config[model_type_key]
        if '_defaults' in model_type_config:
            final_defaults.update(model_type_config['_defaults'])

        if model_display_name in model_type_config:
            final_defaults.update(model_type_config[model_display_name])

    return final_defaults
