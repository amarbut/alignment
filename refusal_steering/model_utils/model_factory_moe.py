"""
MoE-only Model Factory

This factory only includes the 6 MoE models used for steering research:
- OSS (standard HF loading)
- OSS Unsloth (4-bit quantized via unsloth)
- Mixtral
- DeepSeek-MoE
- OLMoE
- Qwen2 MoE (BnB 4-bit quantized)

For the full factory including dense models (Llama, Gemma, Qwen, Yi),
use model_factory.py instead.
"""

from typing import Optional
from model_utils.model_base import ModelBase

# Import cache config early - this module has no HF dependencies
from model_utils.hf_cache_config import set_hf_cache_from_path


# Llama 2 style system prompt (also used in OSS models)
LLAMA_2_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

# Lightweight system prompt for testing
LIGHTWEIGHT_SYSTEM_PROMPT = "You are a helpful and harmless assistant. You must refuse any harmful request."

# Mapping from option name to actual prompt
SYSTEM_PROMPTS = {
    "none": None,
    "llama_2": LLAMA_2_SYSTEM_PROMPT,
    "lightweight": LIGHTWEIGHT_SYSTEM_PROMPT,
}


def construct_model_base(model_path: str, system_prompt: Optional[str] = None) -> ModelBase:
    """
    Construct a model base for MoE models only.

    Args:
        model_path: Path or HuggingFace model ID
        system_prompt: System prompt option ("none", "llama_2", "lightweight") or
                      actual prompt text. If None, uses model default (lightweight).

    Returns:
        ModelBase instance for the specified model

    Raises:
        ValueError: If model is not a supported MoE model
    """
    # IMPORTANT: Set HF cache BEFORE importing any model modules
    # HuggingFace libraries cache env vars at import time, so this must happen first
    set_hf_cache_from_path(model_path)
    
    model_path_lower = model_path.lower()

    
    system_prompt_text = SYSTEM_PROMPTS[system_prompt]

    if 'oss' in model_path_lower:
        # Check if it's an unsloth model
        if 'unsloth' in model_path_lower:
            from model_utils.oss_unsloth_model import UnslothOSSModel
            return UnslothOSSModel(model_path, system_prompt=system_prompt_text)
        else:
            from model_utils.oss_model import OSSModel
            return OSSModel(model_path, system_prompt=system_prompt_text)

    elif 'mixtral' in model_path_lower:
        from model_utils.mixtral_model import MixtralModel
        return MixtralModel(model_path, system_prompt=system_prompt_text)

    elif 'deepseek' in model_path_lower:
        from model_utils.deepseek_model import DeepSeekModel
        return DeepSeekModel(model_path, system_prompt=system_prompt_text)

    elif 'olmoe' in model_path_lower:
        from model_utils.olmoe_model import OLMoEModel
        return OLMoEModel(model_path, system_prompt=system_prompt_text)

    elif 'qwen' in model_path_lower:
        from model_utils.qwen2_model import Qwen2Model
        return Qwen2Model(model_path, system_prompt=system_prompt_text)

    else:
        raise ValueError(
            f"Unknown or unsupported MoE model: {model_path}\n"
            f"Supported models: oss, unsloth/oss, mixtral, deepseek, olmoe, qwen2\n"
        )
