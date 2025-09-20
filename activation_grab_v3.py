#!/usr/bin/env python3

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse

# ---- load model/tokenizer (hardcoded for candidate models)

def load_model_tokenizer(model_id, quant, dtype):
    bnb_config = None
    if quant.lower() == "4bit":
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    elif quant.lower() == "8bit":
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quant.lower() == "none":
        bnb_config = None
    else:
        raise ValueError("quant must be '4bit', '8bit', or 'none'")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,        
        device_map="auto",
    )
    model.eval()
    
    return model, tokenizer

# ---- build chat prompt (simple, using the tokenizer's chat template)

def build_prompt(prompt_format, text, tokenizer):
    if prompt_format == 'chat':
        messages = [{"role": "user", "content": "Why is the sky blue?"}]
        prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    elif prompt_format == 'base':
        prompt_text = text
    return prompt_text

# ---- generate response and collect activations

def gen_last_k(model, tokenizer, prompt_text, decoder_loc, last_k = 5, max_new_tokens = 200, temperature = 0.7, top_p=0.9):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(text)

    # ---- collect activations at each layer for the last 5 generated tokens
    prompt_len = inputs["input_ids"].shape[1]
    gen_ids = output_ids[0, prompt_len:].tolist()
    
    # reconstruct full token ids (prompt + generated)
    with torch.no_grad():
        prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)["input_ids"]
        gen_ids_tensor = torch.tensor([gen_ids], device=model.device, dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, gen_ids_tensor], dim=1)
        attn = torch.ones_like(full_ids, device=model.device)
    
    layers = decoder_loc
    
    captures = {}
    hooks = []
    for i, layer in enumerate(layers):
        def make_hook(ii):
            def hook(_mod, inputs):
                # inputs[0] is residual stream into the block: [B, T, H]
                hs = inputs[0]  # [1, T, H]
                hs_lastk = hs[:, -last_k:, :].detach().to("cpu").float().squeeze(0).numpy()  # [last_k, H]
                captures[ii] = hs_lastk
            return hook
        hooks.append(layer.register_forward_pre_hook(make_hook(i), with_kwargs=False))
    
    try:
        with torch.no_grad():
            _ = model(input_ids=full_ids, attention_mask=attn, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
            
    return layers, captures

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Activation capture across LLMs")

    p.add_argument("--model_id", type=str, required=True, help="HF repo id or local path (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    p.add_argument("--quant", type=str, default="4bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--prompt", type=str, default="Explain why the sky appears blue.")
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--last_k", type=int, default=5)
    
    args = p.parse_args()

    
    #load model & tokenizer
    model, tokenizer = load_model_tokenizer(args.model_id, args.quant, args.dtype)
    
    #hardcoded settings for candidate models
    if "instruct" in args.model_id.lower():
        prompt_format = "chat"
    else:
        prompt_format = "base"
        
    if "openai" in args.model_id.lower():
        decoder_loc = model.transformer.h
    else:
        dedoder_loc = model.model.layers
    
    #build out prompt
    prompt_text = build_prompt(prompt_format, args.prompt, tokenizer)
    
    layers, captures = gen_last_k(model, tokenizer, prompt_text, decoder_loc, last_k=args.last_k, max_new_tokens=args.max_new_tokens, temperature=args.temperature, top_p=args.top_p)

    # ---- tiny summary print
    L = len(layers)
    H = next(iter(captures.values())).shape[-1] if captures else 0
    print(f"\n[activations] layers captured: {len(captures)}/{L}; each is shape [{args.last_k}, {H}]")
    # example: show layer 0 first row norm
    if captures:
        ex = captures[0][0]
        print(f"[activations] example layer 0, token -5, L2 norm: {float(np.linalg.norm(ex)):.4f}")
