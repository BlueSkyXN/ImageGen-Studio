import os
from copy import deepcopy
from .common import _load_yaml, _CHAIN_FEATURES_PATH, _YAML_DIR
from .error_schema import make_not_found_error


def _build_feature_entry(chain_name: str, chain_data: dict, include_schema: bool = False) -> dict:
    entry = {
        "feature_name": chain_name,
        "chains": chain_data.get("chains", chain_name),
        "display_name": chain_data.get("display_name", chain_name),
        "description": chain_data.get("description", ""),
        "supported_tasks": chain_data.get("supported_tasks", []),
        "max_count": chain_data.get("max_count", 1),
        "usage_guideline": chain_data.get("usage_guideline", ""),
    }
    if include_schema:
        schema = deepcopy(chain_data.get("parameters_schema", {}))
        if chain_name in ("krea2_controlnet", "diffsynth_controlnet", "controlnet", "anima_controlnet_lllite"):
            config_key = (
                "Krea2_ControlNet" if chain_name == "krea2_controlnet"
                else "DiffSynth_ControlNet" if chain_name == "diffsynth_controlnet"
                else "Anima_ControlNet_Lllite" if chain_name == "anima_controlnet_lllite"
                else "ControlNet"
            )
            yaml_filename = f"{chain_name}_models.yaml"
            model_path = os.path.join(_YAML_DIR, yaml_filename)
            raw_models = _load_yaml(model_path).get(config_key, [])
            models_list = []
            if isinstance(raw_models, dict):
                for val in raw_models.values():
                    if isinstance(val, list):
                        models_list.extend(val)
                    elif isinstance(val, dict):
                        models_list.append(val)
            elif isinstance(raw_models, list):
                models_list = raw_models

            types_set = set()
            for m in models_list:
                t_val = m.get("Type", [])
                if isinstance(t_val, list):
                    types_set.update(t_val)
                elif isinstance(t_val, str):
                    types_set.add(t_val)
            types = sorted(list(types_set))
            series = sorted(list(set(m.get("Series") for m in models_list if m.get("Series"))))
            if "properties" in schema:
                if "type" in schema["properties"] and types:
                    schema["properties"]["type"]["enum"] = types
                if "series" in schema["properties"] and series:
                    schema["properties"]["series"]["enum"] = series
            if chain_name == "controlnet" and isinstance(raw_models, dict):
                schema["architectures"] = raw_models
        elif chain_name == "ipadapter":
            from .common import _get_ipadapter_presets_by_arch
            presets_by_arch = _get_ipadapter_presets_by_arch()
            schema["presets_by_architecture"] = presets_by_arch
            all_presets = sorted(list(set(presets_by_arch.get("SD1.5", []) + presets_by_arch.get("SDXL", []))))
            if "properties" in schema and "preset" in schema["properties"]:
                schema["properties"]["preset"]["enum"] = all_presets
        entry["parameters_schema"] = schema
    return entry


def handle_get_feature_list(
    feature_name: str | list[str] = "",
    include_schema_on_empty: bool = False,
) -> list | dict:
    """
    Dynamically load supported advanced features from chain_features.yaml.

    - If feature_name is empty: returns summaries by default; callers can set
      include_schema_on_empty for the legacy Fluxus full-schema contract.
    - If feature_name is specified (single feature name, comma-separated string, or list of strings):
      returns complete feature details INCLUDING parameters_schema for the requested feature(s).
    """
    chain_features = _load_yaml(_CHAIN_FEATURES_PATH)

    targets = []
    is_single_string_query = False

    if isinstance(feature_name, list):
        targets = [str(x).strip() for x in feature_name if str(x).strip()]
    elif isinstance(feature_name, str) and feature_name.strip():
        raw_str = feature_name.strip()
        parts = [x.strip() for x in raw_str.split(",") if x.strip()]
        targets = parts
        if len(parts) == 1 and "," not in raw_str:
            is_single_string_query = True

    # Empty discovery can serve both contracts: compact ImageGen summaries or
    # Fluxus-compatible full schemas.
    if not targets:
        return [
            _build_feature_entry(
                name, data, include_schema=include_schema_on_empty
            )
            for name, data in chain_features.items()
        ]

    # Helper function to resolve feature target by key or chains alias
    def _resolve_target(target_name: str) -> str | None:
        if target_name in chain_features:
            return target_name
        for feat_key, feat_data in chain_features.items():
            feat_chains = feat_data.get("chains")
            if isinstance(feat_chains, str) and feat_chains == target_name:
                return feat_key
            elif isinstance(feat_chains, list) and target_name in feat_chains:
                return feat_key
        return None

    resolved_targets = []
    # Case 2: Specific feature(s) requested -> validate existence
    for target in targets:
        resolved = _resolve_target(target)
        if not resolved:
            return make_not_found_error("feature_name", target)
        resolved_targets.append(resolved)

    # Case 3: Return full info including parameters_schema
    results = [
        _build_feature_entry(target, chain_features[target], include_schema=True)
        for target in resolved_targets
    ]

    if is_single_string_query and len(results) == 1:
        return results[0]

    return results
