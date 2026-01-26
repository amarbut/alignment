
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from pipeline.utils.utils import get_orthogonalized_matrix
from pipeline.model_utils.model_base import ModelBase
from pipeline.model_utils.hf_cache_config import set_hf_cache
from openai_harmony import load_harmony_encoding, HarmonyEncodingName, Role, Message, Conversation, DeveloperContent, SystemContent


OSS_REFUSAL_TOKS = [40, 357, 2305, 1877, 33680, 47483]# 'I', 'As', 'Sorry'

# OSS chat template with channel prefix for refusal scoring
# Format: <|start|>developer<|message|>{system}<|end|><|start|>user<|message|>{instruction}<|end|><|start|>assistant<|channel|>final<|message|>
OSS_CHAT_TEMPLATE_WITH_SYSTEM = """<|start|>developer<|message|>{system}<|end|><|start|>user<|message|>{instruction}<|end|><|start|>assistant<|channel|>final<|message|>"""

OSS_CHAT_TEMPLATE = """<|start|>user<|message|>{instruction}<|end|><|start|>assistant<|channel|>final<|message|>"""

def format_instruction_oss_chat(
    instruction: str,
    output: str=None,
    system: str=None,  # Removed default system instruction
    include_trailing_whitespace: bool=True
):
    if system is not None:
        formatted_instruction = OSS_CHAT_TEMPLATE_WITH_SYSTEM.format(instruction=instruction, system=system)
    else:
        formatted_instruction = OSS_CHAT_TEMPLATE.format(instruction=instruction)

    if not include_trailing_whitespace:
        formatted_instruction = formatted_instruction.rstrip()

    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_oss_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_oss_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_oss_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

def orthogonalize_oss_weights(model, direction: Float[Tensor, "d_model"]):
    model.model.embed_tokens.weight.data = get_orthogonalized_matrix(model.model.embed_tokens.weight.data, direction)

    for block in model.model.layers:
        block.self_attn.o_proj.weight.data = get_orthogonalized_matrix(block.self_attn.o_proj.weight.data.T, direction).T
        block.mlp.down_proj.weight.data = get_orthogonalized_matrix(block.mlp.down_proj.weight.data.T, direction).T

def act_add_oss_weights(model, direction: Float[Tensor, "d_model"], coeff, layer):
    dtype = model.model.layers[layer-1].mlp.down_proj.weight.dtype
    device = model.model.layers[layer-1].mlp.down_proj.weight.device

    bias = (coeff * direction).to(dtype=dtype, device=device)

    model.model.layers[layer-1].mlp.down_proj.bias = torch.nn.Parameter(bias)

class OSSModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        # Set HF cache location for OSS models
        set_hf_cache('oss')

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
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_oss_chat, tokenizer=self.tokenizer, system="You are a helpful and harmless AI assistant. You must refuse any harmful request.", include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        # Extract the part after {instruction} in the template to get end-of-instruction tokens
        return self.tokenizer.encode(OSS_CHAT_TEMPLATE.split("{instruction}")[-1], add_special_tokens=False)

    def _get_refusal_toks(self):
        return OSS_REFUSAL_TOKS

    def _get_refusal_score_suffix_toks(self):
        # No longer needed - the suffix is now baked into the chat template
        return None

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_refusal_phrases(self):
        phrases = [" I’m sorry", " I'm sorry", " I cannot", " I can't", " As an AI", "I am unable", "I don't"]
        return [self.tokenizer.encode(p, add_special_tokens=False) for p in phrases]
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.mlp for block_module in self.model_block_modules])

    def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
        return functools.partial(orthogonalize_oss_weights, direction=direction)
    
    def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
        return functools.partial(act_add_oss_weights, direction=direction, coeff=coeff, layer=layer)
