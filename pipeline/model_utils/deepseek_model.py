"""
DeepSeek-MoE-16B-Chat model implementation.

Special features:
- 64 routed + 2 shared experts per MoE layer
- First layer is dense (not MoE)
- Top-6 routing
- Standard chat template format
"""

import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

from pipeline.model_utils.model_base import ModelBase
from pipeline.model_utils.hf_cache_config import set_hf_cache


# DeepSeek refusal tokens (common refusal starters)
DEEPSEEK_REFUSAL_TOKS = [40, 8016, 1699]  # 'I', 'Sorry', 'As' - verify with tokenizer


def format_instruction_deepseek_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    """Format instruction for DeepSeek chat model."""
    # DeepSeek uses a simple User/Assistant format
    formatted = f"User: {instruction}\n\nAssistant:"

    if include_trailing_whitespace:
        formatted += " "

    if output is not None:
        formatted += output

    return formatted


def tokenize_instructions_deepseek_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    """Tokenize instructions for DeepSeek chat model."""
    if outputs is not None:
        prompts = [
            format_instruction_deepseek_chat(
                instruction=instruction,
                output=output,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_deepseek_chat(
                instruction=instruction,
                system=system,
                include_trailing_whitespace=include_trailing_whitespace
            )
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


class DeepSeekModel(ModelBase):
    """ModelBase implementation for DeepSeek-MoE-16B-Chat."""

    def _load_model(self, model_path, dtype=torch.bfloat16):
        # Set HF cache location for DeepSeek models
        set_hf_cache('deepseek')

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()

        model.config.use_cache = False
        model.requires_grad_(False)
        torch.set_grad_enabled(False)

        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(
            tokenize_instructions_deepseek_chat,
            tokenizer=self.tokenizer,
            system=None,
            include_trailing_whitespace=True
        )

    def _get_eoi_toks(self):
        # End of instruction tokens
        return self.tokenizer.encode("\n\n", add_special_tokens=False)

    def _get_refusal_toks(self):
        return DEEPSEEK_REFUSAL_TOKS

    def _get_refusal_phrases(self):
        phrases = [" I'm sorry", " I cannot", " I can't", " As an AI", "I am unable", "I don't"]
        return [self.tokenizer.encode(p, add_special_tokens=False) for p in phrases]

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([
            block_module.self_attn for block_module in self.model_block_modules
        ])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([
            block_module.mlp for block_module in self.model_block_modules
        ])
