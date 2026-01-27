"""
OLMoE-1B-7B-Instruct model implementation.

Special features:
- 64 experts per MoE layer
- Top-8 routing (8 of 64 experts per token)
- 16 layers total
- All layers are MoE (no dense layers)
"""

import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List

from pipeline.model_utils.model_base import ModelBase
from pipeline.model_utils.hf_cache_config import set_hf_cache


# OLMoE refusal tokens - to be verified with tokenizer
OLMOE_REFUSAL_TOKS = []  # Will be set after tokenizer loads


def format_instruction_olmoe_chat(
    instruction: str,
    output: str = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    """Format instruction for OLMoE chat model using chat template."""
    # OLMoE uses a standard chat format
    # We'll use the tokenizer's chat template in tokenize function
    return instruction


def tokenize_instructions_olmoe_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    """Tokenize instructions for OLMoE chat model."""
    prompts = []
    for i, instruction in enumerate(instructions):
        messages = [{"role": "user", "content": instruction}]

        # Use chat template
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        if outputs is not None:
            formatted += outputs[i]

        prompts.append(formatted)

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result


class OLMoEModel(ModelBase):
    """ModelBase implementation for OLMoE-1B-7B-Instruct."""

    def _load_model(self, model_path, dtype=torch.bfloat16):
        # Set HF cache location
        set_hf_cache('oss')  # Use same cache as OSS models

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
            tokenize_instructions_olmoe_chat,
            tokenizer=self.tokenizer,
            system=None,
            include_trailing_whitespace=True
        )

    def _get_eoi_toks(self):
        # End of instruction tokens - the assistant prompt marker
        # OLMoE uses <|assistant|> token
        return self.tokenizer.encode("<|assistant|>", add_special_tokens=False)

    def _get_refusal_toks(self):
        # Common refusal starters for OLMoE
        # Verified with tokenizer - include both with and without leading space
        # 'I' -> 42, ' I' -> 309, 'As' -> 1909, ' As' -> 1284,
        # 'Sorry' -> 15953, ' Sorry' -> 26070
        refusal_starters = ['I', ' I', 'As', ' As', 'Sorry', ' Sorry']
        toks = []
        for starter in refusal_starters:
            encoded = self.tokenizer.encode(starter, add_special_tokens=False)
            if len(encoded) > 0:
                toks.append(encoded[0])
        return list(set(toks))  # Remove duplicates

    def _get_refusal_phrases(self):
        phrases = [" I'm sorry", " I cannot", " I can't", " As an AI", " I am unable", " I don't"]
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
