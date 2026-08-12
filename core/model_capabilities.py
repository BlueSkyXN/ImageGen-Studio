"""Small checkpoint-level capability guards layered over architecture YAML."""

from __future__ import annotations

_EDIT_ONLY_CHAINS = {
    "qwen_image_edit",
    "boogu_image_edit",
    "reference_image",  # Mage-Flow edit checkpoints
}


def supports_chain_for_model(model_name: str, chain_name: str) -> bool:
    """Return whether a concrete checkpoint is known to support a chain.

    Most capabilities apply to every checkpoint of an architecture.  Qwen,
    Boogu, and Mage-Flow register generation and edit checkpoints together, so
    their reference injectors require the concrete Edit variant.
    """

    if chain_name in _EDIT_ONLY_CHAINS:
        return "edit" in str(model_name).casefold()
    return True
