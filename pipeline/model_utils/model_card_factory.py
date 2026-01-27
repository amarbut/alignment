"""
Model Card Factory.

Creates appropriate model card based on the model type/path.
"""

from pipeline.model_utils.model_card import ModelCard


def create_model_card(model_base) -> ModelCard:
    """
    Create appropriate model card for a model.

    Detection order (most specific to least):
    1. Check ModelBase class name
    2. Check model_name_or_path
    3. Check model.config.architectures

    Args:
        model_base: ModelBase instance

    Returns:
        ModelCard instance appropriate for the model

    Raises:
        ValueError: If no model card matches the model type
    """
    model_type = model_base.__class__.__name__
    model_path = model_base.model_name_or_path.lower()

    # Get architecture from config
    if hasattr(model_base.model, 'config'):
        arch = getattr(model_base.model.config, 'architectures', [])
        arch_str = ''.join(arch).lower() if arch else ''
    else:
        arch_str = ''

    # === OSS Detection (Two Variants) ===

    # Unsloth OSS (check first - more specific)
    if 'UnslothOSSModel' in model_type or 'unsloth' in model_path:
        from pipeline.model_utils.unsloth_oss_model_card import UnslothOSSModelCard
        return UnslothOSSModelCard(model_base.model, model_base)

    # Standard OSS
    elif 'OSSModel' in model_type or 'oss' in model_path or 'oss' in arch_str:
        from pipeline.model_utils.oss_model_card import OSSModelCard
        return OSSModelCard(model_base.model, model_base)

    # === Mixtral Detection ===
    elif 'MixtralModel' in model_type or 'mixtral' in model_path or 'mixtral' in arch_str:
        from pipeline.model_utils.mixtral_model_card import MixtralModelCard
        return MixtralModelCard(model_base.model, model_base)

    # === DeepSeek Detection ===
    elif 'DeepSeekModel' in model_type or 'deepseek' in model_path or 'deepseek' in arch_str:
        from pipeline.model_utils.deepseek_model_card import DeepSeekModelCard
        return DeepSeekModelCard(model_base.model, model_base)

    # === OLMoE Detection ===
    elif 'OLMoEModel' in model_type or 'olmoe' in model_path or 'olmoe' in arch_str:
        from pipeline.model_utils.olmoe_model_card import OLMoEModelCard
        return OLMoEModelCard(model_base.model, model_base)

    else:
        raise ValueError(
            f"No model card available for:\n"
            f"  Model type: {model_type}\n"
            f"  Model path: {model_path}\n"
            f"  Architectures: {arch_str}\n"
            f"To add support, create a new model card file in pipeline/model_utils/"
        )
