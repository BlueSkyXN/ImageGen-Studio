import os
import yaml
from functools import lru_cache

@lru_cache(maxsize=1)
def load_controlnet_config():
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _CN_MODEL_LIST_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'controlnet_models.yaml')
    try:
        print("--- Loading controlnet_models.yaml ---")
        with open(_CN_MODEL_LIST_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("--- ✅ controlnet_models.yaml loaded successfully ---")
        return config.get("ControlNet", {})
    except Exception as e:
        print(f"Error loading controlnet_models.yaml: {e}")
        return {}


def get_cn_defaults(arch_val):
    cn_full_config = load_controlnet_config()
    cn_config = cn_full_config.get(arch_val, [])

    if not cn_config:
        return [], None, [], None, "None"

    all_types = sorted(list(set(t for model in cn_config for t in model.get("Type", []))))
    default_type = all_types[0] if all_types else None

    series_choices = []
    if default_type:
        series_choices = sorted(list(set(model.get("Series", "Default") for model in cn_config if default_type in model.get("Type", []))))
    default_series = series_choices[0] if series_choices else None

    filepath = "None"
    if default_series and default_type:
        for model in cn_config:
            if model.get("Series") == default_series and default_type in model.get("Type", []):
                filepath = model.get("Filepath")
                break

    return all_types, default_type, series_choices, default_series, filepath


@lru_cache(maxsize=1)
def load_anima_controlnet_lllite_config():
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _CN_MODEL_LIST_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'anima_controlnet_lllite_models.yaml')
    try:
        print("--- Loading anima_controlnet_lllite_models.yaml ---")
        with open(_CN_MODEL_LIST_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("--- ✅ anima_controlnet_lllite_models.yaml loaded successfully ---")
        return config.get("Anima_ControlNet_Lllite", [])
    except Exception as e:
        print(f"Error loading anima_controlnet_lllite_models.yaml: {e}")
        return []


def get_anima_cn_defaults():
    cn_config = load_anima_controlnet_lllite_config()
    if not cn_config:
        return [], None, [], None, "None"
    all_types = sorted(list(set(t for model in cn_config for t in model.get("Type", []))))
    default_type = all_types[0] if all_types else None
    series_choices = []
    if default_type:
        series_choices = sorted(list(set(model.get("Series", "Default") for model in cn_config if default_type in model.get("Type", []))))
    default_series = series_choices[0] if series_choices else None
    filepath = "None"
    if default_series and default_type:
        for model in cn_config:
            if model.get("Series") == default_series and default_type in model.get("Type", []):
                filepath = model.get("Filepath")
                break
    return all_types, default_type, series_choices, default_series, filepath


@lru_cache(maxsize=1)
def load_diffsynth_controlnet_config():
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _CN_MODEL_LIST_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'diffsynth_controlnet_models.yaml')
    try:
        print("--- Loading diffsynth_controlnet_models.yaml ---")
        with open(_CN_MODEL_LIST_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("--- ✅ diffsynth_controlnet_models.yaml loaded successfully ---")
        return config.get("DiffSynth_ControlNet", {})
    except Exception as e:
        print(f"Error loading diffsynth_controlnet_models.yaml: {e}")
        return {}


def get_diffsynth_cn_defaults(arch_val):
    cn_full_config = load_diffsynth_controlnet_config()
    cn_config = cn_full_config.get(arch_val, [])

    if not cn_config:
        return [], None, [], None, "None"

    all_types = sorted(list(set(t for model in cn_config for t in model.get("Type", []))))
    default_type = all_types[0] if all_types else None

    series_choices = []
    if default_type:
        series_choices = sorted(list(set(model.get("Series", "Default") for model in cn_config if default_type in model.get("Type", []))))
    default_series = series_choices[0] if series_choices else None

    filepath = "None"
    if default_series and default_type:
        for model in cn_config:
            if model.get("Series") == default_series and default_type in model.get("Type", []):
                filepath = model.get("Filepath")
                break

    return all_types, default_type, series_choices, default_series, filepath


@lru_cache(maxsize=1)
def load_krea2_controlnet_config():
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _CN_MODEL_LIST_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'krea2_controlnet_models.yaml')
    try:
        print("--- Loading krea2_controlnet_models.yaml ---")
        with open(_CN_MODEL_LIST_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("--- ✅ krea2_controlnet_models.yaml loaded successfully ---")
        return config.get("Krea2_ControlNet", [])
    except Exception as e:
        print(f"Error loading krea2_controlnet_models.yaml: {e}")
        return []

def get_krea2_cn_defaults():
    cn_config = load_krea2_controlnet_config()
    if not cn_config:
        return [], None, [], None, "None"

    all_types = sorted(list(set(t for model in cn_config for t in model.get("Type", []))))
    default_type = all_types[0] if all_types else None

    series_choices = []
    if default_type:
        series_choices = sorted(list(set(model.get("Series", "Default") for model in cn_config if default_type in model.get("Type", []))))
    default_series = series_choices[0] if series_choices else None

    filepath = "None"
    if default_series and default_type:
        for model in cn_config:
            if model.get("Series") == default_series and default_type in model.get("Type", []):
                filepath = model.get("Filepath")
                break

    return all_types, default_type, series_choices, default_series, filepath


@lru_cache(maxsize=1)
def load_ipadapter_config():
    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    _IPA_MODEL_LIST_PATH = os.path.join(_PROJECT_ROOT, 'yaml', 'ipadapter.yaml')
    try:
        print("--- Loading ipadapter.yaml ---")
        with open(_IPA_MODEL_LIST_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print("--- ✅ ipadapter.yaml loaded successfully ---")
        return config
    except Exception as e:
        print(f"Error loading ipadapter.yaml: {e}")
        return {}
