"""
MoE-only Model Factory

This factory only includes the 5 MoE models used for steering research:
- OSS (standard HF loading)
- OSS Unsloth (4-bit quantized via unsloth)
- Mixtral
- DeepSeek-MoE
- OLMoE

For the full factory including dense models (Llama, Gemma, Qwen, Yi),
use model_factory.py instead.
"""

from typing import Optional
from pipeline.model_utils.model_base import ModelBase
from pipeline.config import SYSTEM_PROMPTS


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
    # Resolve system prompt option to actual text
    if system_prompt in SYSTEM_PROMPTS:
        system_prompt_text = SYSTEM_PROMPTS[system_prompt]
    else:
        # Assume it's already the actual prompt text (or None)
        system_prompt_text = system_prompt

    model_path_lower = model_path.lower()

    if 'oss' in model_path_lower:
        # Check if it's an unsloth model
        if 'unsloth' in model_path_lower:
            from pipeline.model_utils.oss_unsloth_model import UnslothOSSModel
            return UnslothOSSModel(model_path, system_prompt=system_prompt_text)
        else:
            from pipeline.model_utils.oss_model import OSSModel
            return OSSModel(model_path, system_prompt=system_prompt_text)

    elif 'mixtral' in model_path_lower:
        from pipeline.model_utils.mixtral_model import MixtralModel
        return MixtralModel(model_path, system_prompt=system_prompt_text)

    elif 'deepseek' in model_path_lower:
        from pipeline.model_utils.deepseek_model import DeepSeekModel
        return DeepSeekModel(model_path, system_prompt=system_prompt_text)

    elif 'olmoe' in model_path_lower:
        from pipeline.model_utils.olmoe_model import OLMoEModel
        return OLMoEModel(model_path, system_prompt=system_prompt_text)

    else:
        raise ValueError(
            f"Unknown or unsupported MoE model: {model_path}\n"
            f"Supported models: oss, unsloth/oss, mixtral, deepseek, olmoe\n"
            f"For dense models (llama, gemma, qwen, yi), use model_factory.py instead."
        )
