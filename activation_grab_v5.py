#!/usr/bin/env python3

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import json
import random
import os
import pickle

# ---- load model/tokenizer (hardcoded for candidate models)

def load_model_tokenizer(model_id, quant, compute_dtype):
    
    if compute_dtype.lower() == "bf16":
        dtype = torch.bfloat16
    elif compute_dtype.lower() == "fp16":
        dtype = torch.float16

    if 'openai' in model_id:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,       
            device_map="auto",
        )
    else:
        if quant.lower() == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif quant.lower() == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,        
            device_map="auto",
        )
        
    
    model.eval()
    
    return model, tokenizer

# ---- build chat prompt (simple, using the tokenizer's chat template)

def build_prompt(prompt_format, text, system_prompt, tokenizer):
    if prompt_format == 'chat':
        messages = [{"role": "user", "content": text}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif prompt_format == 'base':
        prompt_text = (
            f"System: {system_prompt}\n\n"
            f"User: {text}\n\n"
            f"Assistant:"
        )
    return prompt_text

# ---- generate response and collect activations

def gen_last_k(model, tokenizer, prompt_text, decoder_loc, last_k = 5, max_new_tokens = 200, temperature = 0.7, top_p=0.9):
    
    # construct prompt token ids
    with torch.no_grad():
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)["input_ids"]
        attn = torch.ones_like(prompt_ids, device=model.device)
    
    layers = decoder_loc
    
    captures = {}
    hooks = []
    for i, layer in enumerate(layers):
        def make_hook(ii):
            def hook(_mod, inputs):
                # inputs[0] is residual stream into the block: [B, T, H]
                hs = inputs[0]  # [1, T, H]
                hs_lastk = hs[:, -last_k:, :].detach().to("cpu").float().squeeze(0)#.numpy() leave as tensor  # [last_k, H]
                captures[ii] = hs_lastk
            return hook
        hooks.append(layer.register_forward_pre_hook(make_hook(i), with_kwargs=False))
    
    try:
        with torch.no_grad():
            _ = model(input_ids=prompt_ids, attention_mask=attn, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
            
    return captures

def activation_capture(model_id, quant, dtype, prompt, system_prompt, last_k = 5, max_new_tokens = 200, temperature=0.7, top_p=0.9, save_loc=None, data_loc=None):
    model_name = os.path.basename(model_id)
    #load model & tokenizer
    model, tokenizer = load_model_tokenizer(model_id, quant, dtype)
    
    #hardcoded settings for candidate models
    if "instruct" in model_id.lower() or "chat" in model_id.lower():
        prompt_format = "chat"
    else:
        prompt_format = "base"
        
    decoder_loc = model.model.layers
    hidden_dim = model.get_input_embeddings().weight.shape[1]
    
    if data_loc:
        if save_loc is None:
            print("Need save_loc to run with full dataset")
        
        else:
            # iterate through harmless dataset
            with open(data_loc+"/harmless_train.json", "r") as hl_df:
                hl_prompts = json.load(hl_df)
            
            hl_sample = random.sample(hl_prompts, min(len(hl_prompts),1000))
            
            hl_acts = []
            hl_idx = 0
            for p in hl_sample:
                if hl_idx % 50 == 0:
                    print("Harmless prompt ", hl_idx)
                prompt_text = build_prompt(prompt_format, p["instruction"], system_prompt, tokenizer)
                captures = gen_last_k(model, tokenizer, prompt_text, decoder_loc, last_k, max_new_tokens, temperature, top_p)
                hl_acts.append(captures)
                hl_idx += 1
            
            os.makedirs(f"{save_loc}/{model_name}/", exist_ok=True)
            with open(f"{save_loc}/{model_name}/harmless_activations.pkl", "wb") as hl_sf:
                pickle.dump(hl_acts, hl_sf)
            
            #iterate through harmful dataset
            with open(data_loc+"/harmful_train.json", "r") as hf_df:
                hf_prompts = json.load(hf_df)
            
            hf_acts = []
            hf_idx = 0
            for p in hf_prompts:
                if hf_idx % 50 == 0:
                    print("Harmful prompt ", hf_idx)
                prompt_text = build_prompt(prompt_format, p["instruction"], system_prompt, tokenizer)
                captures = gen_last_k(model, tokenizer, prompt_text, decoder_loc, last_k, max_new_tokens, temperature, top_p)
                hf_acts.append(captures)
                hf_idx += 1
                
            os.makedirs(f"{save_loc}/{model_name}/", exist_ok=True)
            with open(f"{save_loc}/{model_name}/harmful_activations.pkl", "wb") as hf_sf:
                pickle.dump(hf_acts, hf_sf)
        
    else: #run on single prompt
        prompt_text = build_prompt(prompt_format, prompt, system_prompt, tokenizer)
        captures = gen_last_k(model, tokenizer, prompt_text, decoder_loc, last_k, max_new_tokens, temperature, top_p)
        
        if save_loc:
            os.makedirs(f"{save_loc}/", exist_ok=True)
            with open(f"{save_loc}/{model_name}_activations.pkl", "wb") as sf:
                pickle.dump(captures, sf)
                
        H = next(iter(captures.values())).shape[-1] if captures else 0
        print(f"\n[activations] layers captured: {len(captures)}/{len(decoder_loc)}; each is shape [{args.last_k}, {H}]")
        if captures:
            ex = captures[0][0]
            print(f"[activations] example layer 0, token -5, L2 norm: {float(np.linalg.norm(ex)):.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Activation capture across LLMs")

    p.add_argument("--model_id", type=str, required=True, help="HF repo id or local path (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    p.add_argument("--quant", type=str, default="4bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--prompt", type=str, default="Explain why the sky appears blue.")
    p.add_argument("--system_prompt", type=str, default="You are a helpful and harmless AI assistant. You must refuse any harmful request.")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--last_k", type=int, default=5)
    p.add_argument("--save_loc", type=str, default=None)
    p.add_argument("--data_loc", type=str, default=None)
    
    
    args = p.parse_args()
    
    activation_capture(args.model_id, args.quant, args.dtype, args.prompt, args.system_prompt, args.last_k, args.max_new_tokens, args.temperature, args.top_p, args.save_loc, args.data_loc)


