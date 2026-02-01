
import torch
import functools

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List
from torch import Tensor
from jaxtyping import Int, Float

from model_utils.model_base import ModelBase
from model_utils.hf_cache_config import set_hf_cache



MIXTRAL_REFUSAL_TOKS = [315, 19385, 1136] # 'I', 'Sorry''As'

def format_instruction_mixtral_chat(
    instruction: str,
    output: str=None,
    system: str=None,
    include_trailing_whitespace: bool=True
):
    
    formatted_instruction = [{"role": "user", "content": instruction}]
    
    if system is not None:
        formatted_instruction = [{"role": "system", "content": system}] + formatted_instruction


    if output is not None:
        formatted_instruction += output

    return formatted_instruction

def tokenize_instructions_mixtral_chat(
    tokenizer: AutoTokenizer,
    instructions: List[str],
    outputs: List[str]=None,
    system: str=None,
    include_trailing_whitespace=True
):
    if outputs is not None:
        prompts = [
            format_instruction_mixtral_chat(instruction=instruction, output=output, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction, output in zip(instructions, outputs)
        ]
    else:
        prompts = [
            format_instruction_mixtral_chat(instruction=instruction, system=system, include_trailing_whitespace=include_trailing_whitespace)
            for instruction in instructions
        ]

    result = tokenizer(tokenizer.apply_chat_template(prompts,
                                                     tokenize=False,
                                                     add_generation_prompt=True),
        
        padding=True,
        truncation=False,
        return_tensors="pt",
    )

    return result

class MixtralModel(ModelBase):

    def _load_model(self, model_path, dtype=torch.bfloat16):
        # Set HF cache location for Mixtral models
        set_hf_cache('mixtral')

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
        dtype = torch.bfloat16
        bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path, quantization_config=bnb_config)

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_mixtral_chat, tokenizer=self.tokenizer, system=self._system_prompt, include_trailing_whitespace=True)

    def _get_eoi_toks(self):
        return self.tokenizer.encode("</s>", add_special_tokens=False)

    def _get_refusal_toks(self):
        return MIXTRAL_REFUSAL_TOKS
    
    def _get_refusal_phrases(self):
        phrases = [" I’m sorry", " I'm sorry", " I cannot", " I can't", " As an AI", "I am unable", "I don't"]
        return [self.tokenizer.encode(p, add_special_tokens=False) for p in phrases]

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block_module.self_attn for block_module in self.model_block_modules])
    
    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block_module.block_sparse_moe for block_module in self.model_block_modules])

    # def _get_orthogonalization_mod_fn(self, direction: Float[Tensor, "d_model"]):
    #     return functools.partial(orthogonalize_mixtral_weights, direction=direction)
    
    # def _get_act_add_mod_fn(self, direction: Float[Tensor, "d_model"], coeff, layer):
    #     return functools.partial(act_add_mixtral_weights, direction=direction, coeff=coeff, layer=layer)
