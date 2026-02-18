"""
Qwen2-57B-A14B-Instruct MoE model implementation.

Special features:
- 64 routed experts + 1 shared expert per MoE layer
- Top-4 routing (4 of 64 experts per token)
- 28 layers total (all MoE when mlp_only_layers is empty)
- BnB 4-bit quantization for H100-80GB
- ChatML format: <|im_start|>role\ncontent<|im_end|>
"""

import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List

from model_utils.model_base import ModelBase
from model_utils.hf_cache_config import set_hf_cache


def tokenize_instructions_qwen2_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str] = None,
    system: str = None,
    include_trailing_whitespace: bool = True
):
    """Tokenize instructions for Qwen2 chat model using ChatML template."""
    prompts = []
    for i, instruction in enumerate(instructions):
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": instruction})

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


class Qwen2Model(ModelBase):
    """ModelBase implementation for Qwen2-57B-A14B-Instruct (BnB 4-bit quantized)."""

    def _load_model(self, model_path, dtype=torch.bfloat16):
        set_hf_cache('qwen2')

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
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
            tokenize_instructions_qwen2_chat,
            tokenizer=self.tokenizer,
            system=self._system_prompt,
            include_trailing_whitespace=True
        )

    def _get_eoi_toks(self):
        # Qwen2 ChatML format ends user turn with: <|im_end|>\n<|im_start|>assistant\n
        return self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def _get_refusal_toks(self):
        refusal_starters = ['I', ' I', 'As', ' As', 'Sorry', ' Sorry']
        toks = []
        for starter in refusal_starters:
            encoded = self.tokenizer.encode(starter, add_special_tokens=False)
            if len(encoded) > 0:
                toks.append(encoded[0])
        return list(set(toks))

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
