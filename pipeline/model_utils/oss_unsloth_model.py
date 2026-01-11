"""
OSS Model class for Unsloth models.

Uses FastLanguageModel.from_pretrained() instead of AutoModelForCausalLM
to properly load 4-bit quantized unsloth models.
"""

import torch
import functools

from typing import List
from torch import Tensor
from jaxtyping import Float

from pipeline.model_utils.oss_model import (
    OSSModel,
    tokenize_instructions_oss_chat,
    OSS_REFUSAL_TOKS,
    OSS_CHAT_TEMPLATE
)

try:
    from unsloth import FastLanguageModel
except ImportError:
    raise ImportError(
        "unsloth is required for UnslothOSSModel. "
        "Install it with: pip install unsloth"
    )


class UnslothOSSModel(OSSModel):
    """
    OSS model loader for unsloth 4-bit quantized models.

    Inherits all the chat template, tokenization, and utility functions
    from OSSModel, but uses unsloth's FastLanguageModel for loading.
    """

    def _load_model(self, model_path, dtype=None, max_seq_length=2048):
        """
        Load model using unsloth's FastLanguageModel.

        Args:
            model_path: Path to unsloth model
            dtype: Data type (None for auto-detect)
            max_seq_length: Maximum sequence length
        """
        print(f"Loading unsloth model from {model_path}...")

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=max_seq_length,
            dtype=dtype,  # Auto-detect
            load_in_4bit=True,
        )

        # Store tokenizer for later use in _load_tokenizer
        self._tokenizer_from_unsloth = tokenizer

        # Set to eval mode and disable gradients
        model.eval()
        model.requires_grad_(False)
        torch.set_grad_enabled(False)

        # Disable cache
        model.config.use_cache = False

        print(f"✓ Loaded unsloth model: {type(model).__name__}")

        return model

    def _load_tokenizer(self, model_path):
        """
        Return the tokenizer from unsloth loading.

        The tokenizer is already loaded in _load_model, so we just return it.
        """
        if hasattr(self, '_tokenizer_from_unsloth'):
            tokenizer = self._tokenizer_from_unsloth
        else:
            # Fallback: load separately
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_model_block_modules(self):
        """
        Get block modules, unwrapping PEFT/unsloth wrapping if needed.

        Unsloth models may be wrapped in PEFT layers, so we need to
        navigate to the actual base model layers.
        """
        # Navigate through PEFT/unsloth wrapping
        base_model = self.model
        while hasattr(base_model, 'model') and not hasattr(base_model, 'layers'):
            base_model = base_model.model

        if not hasattr(base_model, 'layers'):
            raise ValueError(
                f"Could not find model.layers in {type(self.model).__name__}. "
                f"Model structure: {[attr for attr in dir(base_model) if not attr.startswith('_')]}"
            )

        return base_model.layers

    def _get_attn_modules(self):
        """Get attention modules from unwrapped model."""
        return torch.nn.ModuleList([
            block_module.self_attn
            for block_module in self.model_block_modules
        ])

    def _get_mlp_modules(self):
        """Get MLP modules from unwrapped model."""
        return torch.nn.ModuleList([
            block_module.mlp
            for block_module in self.model_block_modules
        ])

    # All other methods inherited from OSSModel:
    # - _get_tokenize_instructions_fn() - uses tokenize_instructions_oss_chat
    # - _get_eoi_toks() - extracts from OSS_CHAT_TEMPLATE
    # - _get_refusal_toks() - returns OSS_REFUSAL_TOKS
    # - _get_refusal_score_suffix_toks() - returns None
    # - _get_refusal_phrases() - returns list of refusal phrases
    # - _get_orthogonalization_mod_fn() - if needed
    # - _get_act_add_mod_fn() - if needed
