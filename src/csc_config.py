from typing import Sequence

from transformers import PreTrainedTokenizer

from src.configuration_qwen_3_5 import Qwen3_5Config
from src.configuration_qwen3 import Qwen3Config


def normalize_layer_indices(indices: Sequence[int], num_layers: int) -> list[int]:
    normalized: list[int] = []
    for idx in indices:
        if idx == -2:
            break
        resolved_idx = idx if idx >= 0 else num_layers + idx
        if resolved_idx < 0 or resolved_idx >= num_layers:
            raise ValueError(f"Layer index out of range: {idx} (resolved={resolved_idx}, total={num_layers})")
        if resolved_idx not in normalized:
            normalized.append(resolved_idx)
    return normalized


def load_csc_config(
    model_name_or_path: str,
    adapter_layers: Sequence[int],
    use_cache: bool,
    tokenizer: PreTrainedTokenizer | None = None,
) -> tuple[Qwen3Config, list[int]]:
    config = Qwen3Config.from_pretrained(model_name_or_path)
    # text_config = getattr(config, "text_config", None)
    # if text_config is None:
    #     raise ValueError("Qwen3.5 CSC training expects a nested config with text_config.")

    num_layers = getattr(config, "num_hidden_layers", None)
    if num_layers is None:
        raise ValueError("Cannot determine num_hidden_layers from Qwen3.5 text_config.")

    resolved_layers = normalize_layer_indices(adapter_layers, num_layers)
    config.use_cache = use_cache
    config.use_csc_adapter = True
    config.csc_adapter_layers = resolved_layers

    if tokenizer is not None:
        if tokenizer.pad_token_id is not None:
            config.pad_token_id = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.bos_token_id is not None:
            config.bos_token_id = tokenizer.bos_token_id

    config.use_cache = use_cache
    return config, resolved_layers
