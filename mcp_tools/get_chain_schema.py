"""Fluxus-compatible feature schema endpoint."""

from .common import _CHAIN_FEATURES_PATH, _load_yaml
from .error_schema import make_not_found_error, make_validation_error


def handle_get_chain_schema(chain_type: str) -> dict:
    if not chain_type:
        return make_validation_error(
            "Parameter 'chain_type' is required.",
            missing_fields=["chain_type"],
        )

    chain_features = _load_yaml(_CHAIN_FEATURES_PATH)
    resolved = chain_type if chain_type in chain_features else None
    if resolved is None:
        for feature_name, feature_data in chain_features.items():
            aliases = feature_data.get("chains")
            if aliases == chain_type or (
                isinstance(aliases, list) and chain_type in aliases
            ):
                resolved = feature_name
                break

    if resolved is None:
        return make_not_found_error("chain_type", chain_type)

    chain_data = chain_features[resolved]
    return {
        "feature_name": chain_type,
        "canonical_feature_name": resolved,
        "chains": chain_data.get("chains", resolved),
        "display_name": chain_data.get("display_name", resolved),
        "description": chain_data.get("description", ""),
        "supported_tasks": chain_data.get("supported_tasks", []),
        "max_count": chain_data.get("max_count", 1),
        "usage_guideline": chain_data.get("usage_guideline", ""),
        "parameters_schema": chain_data.get("parameters_schema", {}),
    }
