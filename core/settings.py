import yaml
import os
from collections import OrderedDict

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _project_path(*parts):
    return os.path.join(_PROJECT_ROOT, *parts)


CHECKPOINT_DIR = _project_path("models", "checkpoints")
LORA_DIR = _project_path("models", "loras")
EMBEDDING_DIR = _project_path("models", "embeddings")
CONTROLNET_DIR = _project_path("models", "controlnet")
MODEL_PATCHES_DIR = _project_path("models", "model_patches")
DIFFUSION_MODELS_DIR = _project_path("models", "diffusion_models")
VAE_DIR = _project_path("models", "vae")
TEXT_ENCODERS_DIR = _project_path("models", "text_encoders")
STYLE_MODELS_DIR = _project_path("models", "style_models")
CLIP_VISION_DIR = _project_path("models", "clip_vision")
IPADAPTER_DIR = _project_path("models", "ipadapter")
IPADAPTER_FLUX_DIR = _project_path("models", "ipadapter-flux")
INPUT_DIR = _project_path("input")
OUTPUT_DIR = _project_path("output")

CATEGORY_TO_DIR_MAP = {
    "diffusion_models": DIFFUSION_MODELS_DIR,
    "text_encoders": TEXT_ENCODERS_DIR,
    "vae": VAE_DIR,
    "checkpoints": CHECKPOINT_DIR,
    "loras": LORA_DIR,
    "controlnet": CONTROLNET_DIR,
    "model_patches": MODEL_PATCHES_DIR,
    "embeddings": EMBEDDING_DIR,
    "style_models": STYLE_MODELS_DIR,
    "clip_vision": CLIP_VISION_DIR,
    "ipadapter": IPADAPTER_DIR,
    "ipadapter-flux": IPADAPTER_FLUX_DIR
}

_MODEL_LIST_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'model_list.yaml')
_FILE_LIST_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'file_list.yaml')
_IPADAPTER_LIST_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'ipadapter.yaml')
_CONSTANTS_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'constants.yaml')
_MODEL_ARCHITECTURES_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'model_architectures.yaml')
_IMAGE_GEN_FEATURES_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'image_gen_features.yaml')
_MODEL_DEFAULTS_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'model_defaults.yaml')

def load_constants_from_yaml(filepath=_CONSTANTS_PATH):
    if not os.path.exists(filepath):
        print(f"Warning: Constants file not found at {filepath}. Using fallback values.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_architectures_config(filepath=_MODEL_ARCHITECTURES_PATH):
    if not os.path.exists(filepath):
        print(f"Warning: Architectures file not found at {filepath}.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_features_config(filepath=_IMAGE_GEN_FEATURES_PATH):
    if not os.path.exists(filepath):
        print(f"Warning: Features file not found at {filepath}.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_model_defaults(filepath=_MODEL_DEFAULTS_PATH):
    if not os.path.exists(filepath):
        print(f"Warning: Model defaults file not found at {filepath}.")
        return {}
    with open(filepath, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def load_file_download_map(filepath=_FILE_LIST_PATH):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"The file list (for downloads) was not found at: {filepath}")

    with open(filepath, 'r', encoding='utf-8') as f:
        file_list_data = yaml.safe_load(f)

    download_info_map = {}
    for category, files in file_list_data.get('file', {}).items():
        if isinstance(files, list):
            for file_info in files:
                if 'filename' in file_info:
                    file_info['category'] = category
                    download_info_map[file_info['filename']] = file_info
    return download_info_map


def load_models_from_yaml(model_list_filepath=_MODEL_LIST_PATH, download_map=None):
    if not os.path.exists(model_list_filepath):
        raise FileNotFoundError(f"The model list file was not found at: {model_list_filepath}")
    if download_map is None:
        raise ValueError("download_map must be provided to load_models_from_yaml")

    with open(model_list_filepath, 'r', encoding='utf-8') as f:
        model_data = yaml.safe_load(f)

    model_maps = {
        "MODEL_MAP_CHECKPOINT": OrderedDict(),
        "ALL_MODEL_MAP": OrderedDict(),
    }
    category_map_names = {
        "Checkpoint": "MODEL_MAP_CHECKPOINT",
        "Checkpoints": "MODEL_MAP_CHECKPOINT"
    }

    for category, architectures in model_data.items():
        if category in category_map_names:
            map_name = category_map_names[category]
            if not isinstance(architectures, dict): continue

            for arch, arch_data in architectures.items():
                if not isinstance(arch_data, dict): continue

                latent_type = arch_data.get('latent_type', 'latent')
                models = arch_data.get('models', [])
                if not isinstance(models, list): continue

                for model in models:
                    display_name = model['display_name']
                    path_or_components = model.get('path') or model.get('components')
                    mod_category = model.get('category', None)

                    repo_id = ''
                    if isinstance(path_or_components, str):
                        download_info = download_map.get(path_or_components, {})
                        repo_id = download_info.get('repo_id', '')

                    model_tuple = (
                        repo_id,
                        path_or_components,
                        arch,
                        latent_type,
                        mod_category
                    )
                    model_maps[map_name][display_name] = model_tuple
                    model_maps["ALL_MODEL_MAP"][display_name] = model_tuple

    return model_maps

try:
    ALL_FILE_DOWNLOAD_MAP = load_file_download_map()
    loaded_maps = load_models_from_yaml(download_map=ALL_FILE_DOWNLOAD_MAP)
    MODEL_MAP_CHECKPOINT = loaded_maps["MODEL_MAP_CHECKPOINT"]
    ALL_MODEL_MAP = loaded_maps["ALL_MODEL_MAP"]

    category_to_model_type = {
        "diffusion_models": "UNET",
        "text_encoders": "TEXT_ENCODER",
        "vae": "VAE",
        "checkpoints": "SDXL",
        "loras": "LORA",
        "controlnet": "CONTROLNET",
        "model_patches": "MODEL_PATCH",
        "style_models": "STYLE",
        "clip_vision": "CLIP_VISION",
        "ipadapter": "IPADAPTER",
        "ipadapter-flux": "IPADAPTER_FLUX"
    }
    for filename, file_info in ALL_FILE_DOWNLOAD_MAP.items():
        if filename not in ALL_MODEL_MAP:
            category = file_info.get('category')
            model_type = category_to_model_type.get(category, 'UNKNOWN')
            repo_id = file_info.get('repo_id', '')
            ALL_MODEL_MAP[filename] = (repo_id, filename, model_type, None, None)

    MODEL_TYPE_MAP = {k: v[2] for k, v in ALL_MODEL_MAP.items()}

    ARCH_CATEGORIES_MAP = {}
    for display_name, info in MODEL_MAP_CHECKPOINT.items():
        arch = info[2]
        cat = info[4] if len(info) > 4 else None
        if arch not in ARCH_CATEGORIES_MAP:
            ARCH_CATEGORIES_MAP[arch] = []
        if cat and cat not in ARCH_CATEGORIES_MAP[arch]:
            ARCH_CATEGORIES_MAP[arch].append(cat)

except Exception as e:
    print(f"FATAL: Could not load model configuration from YAML. Error: {e}")
    ALL_FILE_DOWNLOAD_MAP = {}
    MODEL_MAP_CHECKPOINT, ALL_MODEL_MAP = {}, {}
    MODEL_TYPE_MAP = {}
    ARCH_CATEGORIES_MAP = {}


try:
    _constants = load_constants_from_yaml()
    MAX_LORAS = _constants.get('MAX_LORAS', 5)
    MAX_EMBEDDINGS = _constants.get('MAX_EMBEDDINGS', 5)
    MAX_CONDITIONINGS = _constants.get('MAX_CONDITIONINGS', 10)
    MAX_CONTROLNETS = _constants.get('MAX_CONTROLNETS', 5)
    MAX_IPADAPTERS = _constants.get('MAX_IPADAPTERS', 5)
    LORA_SOURCE_CHOICES = _constants.get('LORA_SOURCE_CHOICES', ["Civitai", "File"])
    RESOLUTION_MAP = _constants.get('RESOLUTION_MAP', {})
    MULTIPLIERS_MAP = _constants.get('MULTIPLIERS_MAP', {})
    ARCHITECTURES_CONFIG = load_architectures_config()
    FEATURES_CONFIG = load_features_config()
    MODEL_DEFAULTS_CONFIG = load_model_defaults()
except Exception as e:
    print(f"FATAL: Could not load constants from YAML. Error: {e}")
    MAX_LORAS, MAX_EMBEDDINGS, MAX_CONDITIONINGS, MAX_CONTROLNETS, MAX_IPADAPTERS = 5, 5, 10, 5, 5
    LORA_SOURCE_CHOICES = ["Civitai", "File"]
    RESOLUTION_MAP = {}
    MULTIPLIERS_MAP = {}
    ARCHITECTURES_CONFIG = {}
    FEATURES_CONFIG = {}
    MODEL_DEFAULTS_CONFIG = {}
