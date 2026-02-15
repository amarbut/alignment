
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from model_utils.model_base import ModelBase
from model_utils.hf_cache_config import set_hf_cache


# Channel suffix to force the model into the 'final' response channel.
# GPT-OSS models use channels (analysis, commentary, final) and need to be
# directed to the final channel for refusal scoring and direct generation.
OSS_CHANNEL_SUFFIX = "<|channel|>analysis<|message|>The user doesn't want us to think<|end|><|start|>assistant<|channel|>final<|message|>"

# End-of-instruction suffix (tokens between user content and model response).
# Used for eoi_toks extraction; matches the tail of the formatted template.
OSS_EOI_SUFFIX = "<|end|><|channel|>analysis<|message|>The user doesn't want us to think<|end|><|start|>assistant<|channel|>final<|message|>"

LLAMA_2_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully
as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

def format_instruction_oss_chat(
    tokenizer: AutoTokenizer,
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True
):
    """
    Format an instruction using the tokenizer's built-in chat template.

    Uses apply_chat_template() to ensure the auto-injected system message
    (model identity, channel instructions, reasoning mode) is included,
    then appends the channel suffix to force the 'final' response channel.
    """
    messages = []
    if system is not None:
        messages.append({'role': 'developer', 'content': system})
    messages.append({'role': 'user', 'content': instruction})

    # Use apply_chat_template for proper formatting (includes auto-injected system message)
    formatted_instruction = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )

    # Append assistant generation prompt with forced final channel
    formatted_instruction += OSS_CHANNEL_SUFFIX

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
            format_instruction_oss_chat(tokenizer=tokenizer, instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_oss_chat(tokenizer=tokenizer, instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(
        prompts,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

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
        return functools.partial(tokenize_instructions_oss_chat, tokenizer=self.tokenizer, system=self._system_prompt, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        # Encode the end-of-instruction suffix (tokens between user content and model response)
        return self.tokenizer.encode(OSS_EOI_SUFFIX, add_special_tokens=False)

    def _get_refusal_toks(self):
        refusal_starters = ['I', ' I', 'As', ' As', 'Sorry', ' Sorry']
        toks = []
        for starter in refusal_starters:
            encoded = self.tokenizer.encode(starter, add_special_tokens=False)
            if len(encoded) > 0:
                toks.append(encoded[0])
        return list(set(toks))  # Remove duplicates

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
