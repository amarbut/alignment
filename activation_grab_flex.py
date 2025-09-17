#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 10:25:14 2025

@author: anna
"""

# activation_grab.py
import os
import argparse
from typing import List, Optional, Dict

import torch
import numpy as np
from collections import deque
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ------------------------- Utilities -------------------------

def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}

def parse_layers(s: Optional[str], num_layers: int) -> List[int]:
    """Parse layer indices like '0,12,23' or 'all'."""
    if s is None or s.strip().lower() == "all":
        return list(range(num_layers))
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        i = int(part)
        if not (0 <= i < num_layers):
            raise ValueError(f"Layer index {i} out of range [0, {num_layers-1}]")
        out.append(i)
    return out

def infer_decoder_layers(model) -> List[torch.nn.Module]:
    """
    Try common locations for decoder layers across HF causal LMs.
    Works for LLaMA, Mixtral, Qwen, GPT-OSS, etc.
    """
    candidates = [
        getattr(getattr(model, "model", None), "layers", None),        # LLaMA-style: model.model.layers
        getattr(getattr(model, "model", None), "decoder", None) and
        getattr(getattr(model.model, "decoder", None), "layers", None),
        getattr(getattr(model, "transformer", None), "h", None),       # GPT-style: model.transformer.h (list)
    ]
    for c in candidates:
        if isinstance(c, (list, torch.nn.ModuleList)) and len(c) > 0:
            return c
    raise RuntimeError("Could not locate decoder layers list on this model.")

# ---------------------- Model loading ------------------------

def load_model_and_tokenizer(
    model_id_or_path: str,
    quant: str = "4bit",                 # "4bit", "8bit", or "none"
    compute_dtype: str = "bf16",         # "bf16" or "fp16"
    device_map: str = "auto",
    trust_remote_code: bool = False,
):
    if compute_dtype.lower() == "bf16":
        dtype = torch.bfloat16
    elif compute_dtype.lower() == "fp16":
        dtype = torch.float16
    else:
        raise ValueError("compute_dtype must be 'bf16' or 'fp16'")

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

    tokenizer = AutoTokenizer.from_pretrained(model_id_or_path, use_fast=True, trust_remote_code=trust_remote_code)

    kwargs = dict(
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )

    model = AutoModelForCausalLM.from_pretrained(model_id_or_path, **kwargs)
    model.eval()

    # Ensure we have a pad token if needed (for teacher-forced masks)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

# ---------------- Activation capture classes -----------------

class ResidualCapture:
    """
    Capture *input hidden_states* to each decoder layer via forward_pre_hook.
    Stores only the last K steps (newly generated tokens).
    """
    def __init__(self, model, layers_list, layer_indices=None, last_token_only=True, to_cpu=True, last_k_steps=5):
        self.model = model
        self.layers_list = layers_list
        self.num_layers = len(self.layers_list)
        self.layer_indices = layer_indices if layer_indices is not None else list(range(self.num_layers))
        self.last_token_only = last_token_only
        self.to_cpu = to_cpu
        self.last_k_steps = last_k_steps

        self._hooks = []
        self._current_step_buffer: Dict[int, np.ndarray] = {}
        self.captures_deque = deque(maxlen=last_k_steps)

    def _pre_hook(self, layer_idx):
        def hook(module, inputs):
            # captures layer-specific details for each layer
            hidden_states = inputs[0]  # [batch, seq_len, hidden_dim]
            hs = hidden_states[:, -1, :] if self.last_token_only else hidden_states
            hs = hs.detach().to("cpu") if self.to_cpu else hs.detach()
            arr = hs.squeeze(0).contiguous().float().numpy()
            self._current_step_buffer[layer_idx] = arr
        return hook
    
    # the rest of these are utilities around managing the hooks and navigating between layers and tokens
    def install(self):
        for i in self.layer_indices:
            h = self.layers_list[i].register_forward_pre_hook(self._pre_hook(i), with_kwargs=False)
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def start_step(self):
        self._current_step_buffer = {}

    def end_step(self):
        self.captures_deque.append(dict(self._current_step_buffer))

    @property
    def captures(self):
        return list(self.captures_deque)

# not likely to use, but no harm in leaving the class
# per ChatGPT, incremental activation capture is more efficient, even though it creates a rolling storage of activations
# this teacher-forced pass can be used as an afterthought to capture some number of activations for the last k tokens all at once
class LastKPositionsCapture:
    """
    Teacher-forced pass: capture *input hidden_states* to each decoder layer
    for only the last K sequence positions (single shot).
    """
    def __init__(self, model, layers_list, layer_indices=None, last_k_positions=5, to_cpu=True):
        self.model = model
        self.layers_list = layers_list
        self.num_layers = len(self.layers_list)
        self.layer_indices = layer_indices if layer_indices is not None else list(range(self.num_layers))
        self.last_k_positions = last_k_positions
        self.to_cpu = to_cpu
        self._hooks = []
        self.captures: Dict[int, np.ndarray] = {}

    def _pre_hook(self, layer_idx):
        def hook(module, inputs):
            hs = inputs[0]  # [batch, seq_len, hidden_dim]
            hs_lastk = hs[:, -self.last_k_positions:, :] if self.last_k_positions > 0 else hs
            hs_lastk = hs_lastk.detach().to("cpu") if self.to_cpu else hs_lastk.detach()
            arr = hs_lastk.squeeze(0).contiguous().float().numpy()  # [last_k, hidden_dim]
            self.captures[layer_idx] = arr
        return hook

    def install(self):
        for i in self.layer_indices:
            h = self.layers_list[i].register_forward_pre_hook(self._pre_hook(i), with_kwargs=False)
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks = []

# ------------------ Generation + capture APIs ----------------

def pick_token(logits, do_sample: bool, temperature: float, top_p: float):
    # function for response generation
    # default is greedy (selects most likely token)
    # temp and do_sample allow some controls over induced stochasticity of choice
    # top_p removes very long tail to avoid choosing very unlikely token
    if do_sample and temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        if top_p < 1.0:
            sorted_p, sorted_idx = torch.sort(probs, descending=True)
            csum = torch.cumsum(sorted_p, dim=-1)
            cutoff = csum > top_p
            cutoff[..., 1:] = cutoff[..., :-1].clone()
            cutoff[..., 0] = False
            sorted_p = torch.where(cutoff, torch.zeros_like(sorted_p), sorted_p)
            sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
            idx = torch.multinomial(sorted_p, 1)
            return sorted_idx.gather(-1, idx)
        return torch.multinomial(probs, 1)
    return torch.argmax(logits, dim=-1, keepdim=True)

def generate_and_capture_last_k(
    tokenizer,
    model,
    layers_list,
    prompt: str,
    max_new_tokens: int = 64,
    last_k_steps: int = 5,
    temperature: float = 0.0,
    top_p: float = 1.0,
    do_sample: bool = False,
    layer_indices: Optional[List[int]] = None,
):
    """
    Incremental decoding with KV cache; keep only last_k_steps per-token activations.
    Returns (text, captures_list)
    """
    device = model.device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    attn = inputs.get("attention_mask", None)

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn, use_cache=True)
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

    rc = ResidualCapture(
        model=model,
        layers_list=layers_list,
        layer_indices=layer_indices,
        last_token_only=True,
        to_cpu=True,
        last_k_steps=last_k_steps,
    )
    rc.install()

    gen_ids = []
    try:
        for _ in range(max_new_tokens):
            next_id = pick_token(next_logits, do_sample, temperature, top_p)
            step_ids = next_id
            step_attn = torch.ones_like(step_ids, device=device)

            rc.start_step()
            with torch.no_grad():
                out = model(
                    input_ids=step_ids,
                    attention_mask=step_attn,
                    past_key_values=past_kv,
                    use_cache=True,
                )
            rc.end_step()

            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]
            gen_ids.append(int(next_id.item()))

            # EOS early stop
            if tokenizer.eos_token_id is not None and gen_ids[-1] == tokenizer.eos_token_id:
                break
    finally:
        rc.remove()

    full_ids = torch.cat([input_ids, torch.tensor([gen_ids], device=input_ids.device)], dim=1)
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    return text, rc.captures

def teacher_forced_capture_last_k_positions(
    tokenizer,
    model,
    layers_list,
    prompt: str,
    generated_ids: List[int],
    last_k_positions: int = 5,
    layer_indices: Optional[List[int]] = None,
):
    """
    One-shot forward over (prompt + generated) that captures only the last_k_positions
    inputs to each selected layer. Returns (text, caps_dict).
    """
    device = model.device
    with torch.no_grad():
        prompt_ids = tokenizer(prompt, return_tensors="pt").to(device)["input_ids"]
        gen_ids = torch.tensor([generated_ids], device=device, dtype=prompt_ids.dtype)
        full_ids = torch.cat([prompt_ids, gen_ids], dim=1)
        attn = torch.ones_like(full_ids, device=device)

    rc = LastKPositionsCapture(
        model=model,
        layers_list=layers_list,
        layer_indices=layer_indices,
        last_k_positions=last_k_positions,
        to_cpu=True,
    )
    rc.install()
    try:
        with torch.no_grad():
            _ = model(input_ids=full_ids, attention_mask=attn, use_cache=False)
    finally:
        rc.remove()

    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    return text, rc.captures

# ----------------------------- CLI ---------------------------

def main():
    p = argparse.ArgumentParser(description="Activation capture across LLMs")
    # Model / loading
    p.add_argument("--model_id", type=str, required=True,
                   help="HF repo id or local path (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    p.add_argument("--quant", type=str, default="4bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--compute_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--attn_impl", type=str, default=None, help="e.g., flash_attention_2")
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--trust_remote_code", type=str2bool, default=False)

    # Capture config
    p.add_argument("--capture_mode", type=str, default="incremental",
                   choices=["incremental", "teacher_forced"])
    p.add_argument("--layers", type=str, default="all",
                   help="comma list (e.g., '0,12,23') or 'all'")
    p.add_argument("--last_k", type=int, default=5)

    # Generation
    p.add_argument("--prompt", type=str, default="Explain why the sky appears blue.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", type=str2bool, default=False)

    args = p.parse_args()

    tokenizer, model = load_model_and_tokenizer(
        model_id_or_path=args.model_id,
        quant=args.quant,
        compute_dtype=args.compute_dtype,
        attn_impl=args.attn_impl,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    layers_list = infer_decoder_layers(model)
    layer_idx_list = parse_layers(args.layers, num_layers=len(layers_list))

    if args.capture_mode == "incremental":
        text, caps = generate_and_capture_last_k(
            tokenizer=tokenizer,
            model=model,
            layers_list=layers_list,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            last_k_steps=args.last_k,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.do_sample,
            layer_indices=layer_idx_list,
        )
        print(text)
        print(f"[incremental] kept steps: {len(caps)}; captured layers per step: {len(caps[-1]) if caps else 0}")

    else:
        # First generate greedily (or with your settings), then do a teacher-forced capture of the last_k positions
        device = model.device
        with torch.no_grad():
            gen_inp = tokenizer(args.prompt, return_tensors="pt").to(device)
            out = model.generate(
                **gen_inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.do_sample,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        full_text = tokenizer.decode(out[0], skip_special_tokens=True)
        # Extract only the generated part’s ids:
        generated_ids = out[0, gen_inp["input_ids"].shape[1]:].tolist()

        text, caps = teacher_forced_capture_last_k_positions(
            tokenizer=tokenizer,
            model=model,
            layers_list=layers_list,
            prompt=args.prompt,
            generated_ids=generated_ids,
            last_k_positions=args.last_k,
            layer_indices=layer_idx_list,
        )
        print(text)
        print(f"[teacher_forced] layers captured: {len(caps)}; each array is shape [last_k, hidden_dim]")

if __name__ == "__main__":
    main()
