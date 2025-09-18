#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Activation capture (incremental, last-k steps) — simplified/robust overhaul

Key changes vs your previous version:
- Removed teacher-forced capture path entirely.
- Removed attention-implementation branches (Flash/SDPA logic etc.).
- Simplified model layer discovery to a small lookup table for our target models
  (LLaMA-3.1 8B base+instruct, GPT-OSS-20B, Mixtral-8x7B-Instruct) with a
  conservative fallback.
- Added reliable chat-templating for instruct models (opt-in via --chat_template
  or auto-enabled when model_id suggests "instruct").
- Fixed incremental-attention mask shape when using past_kv.
- Added on-disk saving of activations + metadata (--save_dir). We store
  a single NPZ named run_{timestamp}.npz with arrays:
    * activations: float32 array [S, L, H] (S = kept decoding steps, L = #layers)
      If some layers were not captured, we pad with NaNs.
    * layer_indices: int32 [L]
    * generated_ids: int32 [S]
    * input_ids: int32 [prompt_len]
      Plus a JSON sidecar with metadata.
- BOS/EOS/PAD safety wiring for LLaMA-3(.1), without growing vocab.

Usage (examples):
  python activation_grab_flex_overhaul.py \
    --model_id meta-llama/Meta-Llama-3.1-8B-Instruct \
    --prompt "Why is the sky blue?" \
    --last_k 5 --save_dir ./caps_llama31_instruct --chat_template true

  python activation_grab_flex_overhaul.py \
    --model_id mistralai/Mixtral-8x7B-Instruct-v0.1 \
    --prompt "Explain diffusion models at a high level." \
    --last_k 5 --save_dir ./caps_mixtral --chat_template true

  python activation_grab_flex_overhaul.py \
    --model_id cognitivecomputations/dolphin-2.9-llama3-8b \
    --prompt "Summarize the CMB" --last_k 5 --save_dir ./caps_gptoss
"""

import os
import re
import json
import time
import argparse
from typing import List, Optional, Dict, Any

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# ------------------------- Utilities -------------------------

def str2bool(v: str) -> bool:
    return str(v).lower() in {"1", "true", "t", "yes", "y"}

# ---------------------- Model loading ------------------------

MODEL_LAYER_TABLE = {
    # family or explicit id pattern -> attribute path (resolved against model)
    # We'll match by regex against model_id and also try model.config.model_type.
    r"meta-llama/.+": "model.layers",       # LLaMA 2/3/3.1 style
    r"mistral(ai)?/.+|mistralai/.+": "model.layers",  # Mistral/Mixtral
    r"gpt[-_]?oss|cognitivecomputations/.+|tiiuae/.+": "transformer.h",  # GPT-ish
}

INSTRUCT_HINT_PAT = re.compile(r"(instruct|chat|it|assistant)", re.I)


def _pick_layer_list(model, model_id: str):
    """Return a list-like of decoder blocks using our small lookup table."""
    model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
    candidates = []

    # Prefer explicit table matches by model_id
    for pat, path in MODEL_LAYER_TABLE.items():
        if re.search(pat, model_id) or (pat == model_type):
            candidates.append(path)

    # Add robust fallbacks
    candidates.extend([
        "model.layers",           # LLaMA / Mistral / Mixtral
        "model.decoder.layers",   # some encoder-decoder styles
        "transformer.h",          # GPT-style
    ])

    for path in candidates:
        obj = model
        ok = True
        for attr in path.split('.'):
            if not hasattr(obj, attr):
                ok = False
                break
            obj = getattr(obj, attr)
        if ok and isinstance(obj, (list, torch.nn.ModuleList)) and len(obj) > 0:
            return obj
    raise RuntimeError("Could not locate decoder layers list on this model.")


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

    tokenizer = AutoTokenizer.from_pretrained(
        model_id_or_path, use_fast=True, trust_remote_code=trust_remote_code
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id_or_path,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    )
    model.eval()

    # LLaMA-3(.1) BOS/EOS are inside-vocab (128000/128001). Reuse EOS for PAD.
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # Diagnostics
    emb_rows = model.get_input_embeddings().num_embeddings
    print("[diag:model] emb_rows:", emb_rows)
    print("[diag:tok] len(tokenizer):", len(tokenizer), "vocab_size:", tokenizer.vocab_size)
    print("[diag:tok] bos/eos/pad:", tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id)

    return tokenizer, model

# ---------------- Activation capture classes -----------------

class ResidualCapture:
    """Capture *input hidden_states* to each decoder layer via forward_pre_hook.
    Stores only the last K steps (newly generated tokens).
    """
    def __init__(self, model, layers_list, layer_indices=None, last_k_steps=5, to_cpu=True):
        self.model = model
        self.layers_list = layers_list
        self.num_layers = len(self.layers_list)
        self.layer_indices = layer_indices if layer_indices is not None else list(range(self.num_layers))
        self.last_k_steps = last_k_steps
        self.to_cpu = to_cpu

        self._hooks = []
        self._cur: Dict[int, np.ndarray] = {}
        self._steps: List[Dict[int, np.ndarray]] = []

    def _pre_hook(self, layer_idx):
        def hook(module, inputs):
            hidden_states = inputs[0]  # [B, T, H]
            hs = hidden_states[:, -1, :]  # last token position only
            hs = hs.detach().to("cpu") if self.to_cpu else hs.detach()
            arr = hs.squeeze(0).contiguous().float().numpy()  # [H]
            self._cur[layer_idx] = arr
        return hook

    def install(self):
        for i in self.layer_indices:
            h = self.layers_list[i].register_forward_pre_hook(self._pre_hook(i), with_kwargs=False)
            self._hooks.append(h)

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def start_step(self):
        self._cur = {}

    def end_step(self):
        self._steps.append(dict(self._cur))
        # Trim to last_k
        if len(self._steps) > self.last_k_steps:
            self._steps = self._steps[-self.last_k_steps:]

    @property
    def captures(self) -> List[Dict[int, np.ndarray]]:
        return list(self._steps)

# ------------------ Generation + capture ---------------------

def apply_chat_template_if_needed(tokenizer, prompt: str, chat_template: bool, model_id: str) -> str:
    """For instruct models, wrap the prompt via tokenizer.apply_chat_template if available."""
    should = chat_template or bool(INSTRUCT_HINT_PAT.search(model_id))
    if should and hasattr(tokenizer, "apply_chat_template") and callable(tokenizer.apply_chat_template):
        msgs = [{"role": "user", "content": prompt}]
        try:
            rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return rendered
        except Exception as e:
            print("[warn] chat_template failed, falling back to raw prompt:", e)
    return prompt


def build_attn_mask_for_past(step_len: int, past_kv: Any, device) -> tuple[torch.Tensor, int]:
    """Return (attention_mask, past_len). 1D mask length = past_len + step_len."""
    if past_kv is None:
        return torch.ones((1, step_len), device=device), 0
    try:
        if isinstance(past_kv, (list, tuple)) and len(past_kv) > 0:
            past_len = past_kv[0][0].shape[-2]
        else:
            past_len = 0
    except Exception:
        past_len = 0
    total = past_len + step_len
    return torch.ones((1, total), device=device), past_len


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
    use_chat_template: bool = False,
):
    device = model.device
    rendered = apply_chat_template_if_needed(tokenizer, prompt, use_chat_template, getattr(model.config, 'name_or_path', ''))

    # Avoid auto specials; prepend BOS only if needed (chat template usually handles it)
    inputs = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(device)
    input_ids = inputs["input_ids"]

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids, device=device), use_cache=True)
        past_kv = out.past_key_values
        next_logits = out.logits[:, -1, :]

    rc = ResidualCapture(
        model=model,
        layers_list=layers_list,
        layer_indices=layer_indices,
        last_k_steps=last_k_steps,
        to_cpu=True,
    )
    rc.install()

    generated_ids: List[int] = []
    try:
        for _ in range(max_new_tokens):
            if do_sample and temperature > 0:
                probs = torch.softmax(next_logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_p, sorted_idx = torch.sort(probs, descending=True)
                    csum = torch.cumsum(sorted_p, dim=-1)
                    cutoff = csum > top_p
                    cutoff[..., 1:] = cutoff[..., :-1].clone()
                    cutoff[..., 0] = False
                    sorted_p = torch.where(cutoff, torch.zeros_like(sorted_p), sorted_p)
                    sorted_p = sorted_p / sorted_p.sum(dim=-1, keepdim=True)
                    idx = torch.multinomial(sorted_p, 1)
                    next_id = sorted_idx.gather(-1, idx)
                else:
                    next_id = torch.multinomial(probs, 1)
            else:
                next_id = torch.argmax(next_logits, dim=-1, keepdim=True)

            step_ids = next_id  # [1,1]
            attn_mask, past_len = build_attn_mask_for_past(step_len=1, past_kv=past_kv, device=device)
            # Crucial for correct decoding with KV cache: advance positional ids.
            position_ids = torch.tensor([[past_len]], device=device, dtype=torch.long)

            rc.start_step()
            with torch.no_grad():
                out = model(
                    input_ids=step_ids,
                    attention_mask=attn_mask,
                    past_key_values=past_kv,
                    position_ids=position_ids,
                    use_cache=True,
                )
            rc.end_step()

            past_kv = out.past_key_values
            next_logits = out.logits[:, -1, :]
            gid = int(next_id.item())
            generated_ids.append(gid)

            eos = tokenizer.eos_token_id
            if eos is not None and gid == eos:
                break
    finally:
        rc.remove()

    full_ids = torch.cat([input_ids, torch.tensor([generated_ids], device=input_ids.device)], dim=1)
    text = tokenizer.decode(full_ids[0], skip_special_tokens=True)
    return text, rc.captures, input_ids[0].tolist(), generated_ids

# -------------------------- Saving ---------------------------

def stack_captures(caps: List[Dict[int, np.ndarray]], layer_indices: List[int]) -> np.ndarray:
    """Stack capture dicts (per step) into [S, L, H] with NaN for missing layers."""
    if not caps:
        return np.zeros((0, len(layer_indices), 0), dtype=np.float32)
    L = len(layer_indices)
    H = next(iter(caps[-1].values())).shape[-1] if caps[-1] else 0
    S = len(caps)
    out = np.full((S, L, H), np.nan, dtype=np.float32)
    layer_pos = {lid: i for i, lid in enumerate(layer_indices)}
    for s, d in enumerate(caps):
        for lid, vec in d.items():
            if lid in layer_pos:
                out[s, layer_pos[lid]] = vec.astype(np.float32, copy=False)
    return out


def save_run(save_dir: str, activations: np.ndarray, layer_indices: List[int],
             input_ids: List[int], generated_ids: List[int], meta: Dict[str, Any]):
    os.makedirs(save_dir, exist_ok=True)
    ts = int(time.time())
    base = os.path.join(save_dir, f"run_{ts}")
    np.savez_compressed(
        base + ".npz",
        activations=activations,
        layer_indices=np.asarray(layer_indices, dtype=np.int32),
        input_ids=np.asarray(input_ids, dtype=np.int32),
        generated_ids=np.asarray(generated_ids, dtype=np.int32),
    )
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[save] wrote {base}.npz and {base}.json")

# ----------------------------- CLI ---------------------------

def parse_layers(s: Optional[str], num_layers: int) -> List[int]:
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


def main():
    p = argparse.ArgumentParser(description="Activation capture (incremental, last-k steps)")
    # Model / loading
    p.add_argument("--model_id", type=str, required=True,
                   help="HF repo id or local path (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct)")
    p.add_argument("--quant", type=str, default="4bit", choices=["4bit", "8bit", "none"])
    p.add_argument("--compute_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--device_map", type=str, default="auto")
    p.add_argument("--trust_remote_code", type=str2bool, default=False)

    # Capture config
    p.add_argument("--layers", type=str, default="all", help="comma list (e.g., '0,12,23') or 'all'")
    p.add_argument("--last_k", type=int, default=5)

    # Generation
    p.add_argument("--prompt", type=str, default="Explain why the sky appears blue.")
    p.add_argument("--max_new_tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--do_sample", type=str2bool, default=False)
    p.add_argument("--chat_template", type=str2bool, default=False,
                   help="Wrap prompt with tokenizer chat template (good for *Instruct* models)")

    # Saving
    p.add_argument("--save_dir", type=str, default="",
                   help="If provided, write NPZ+JSON with activations and metadata")

    args = p.parse_args()

    tokenizer, model = load_model_and_tokenizer(
        model_id_or_path=args.model_id,
        quant=args.quant,
        compute_dtype=args.compute_dtype,
        device_map=args.device_map,
        trust_remote_code=args.trust_remote_code,
    )

    layers_list = _pick_layer_list(model, args.model_id)
    layer_idx_list = parse_layers(args.layers, num_layers=len(layers_list))

    text, caps, input_ids, gen_ids = generate_and_capture_last_k(
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
        use_chat_template=args.chat_template,
    )

    print("\n[output]\n" + text)
    print(f"[capture] kept steps: {len(caps)}; captured layers/step: {len(caps[-1]) if caps else 0}")

    # Save if requested
    if args.save_dir:
        act = stack_captures(caps, layer_idx_list)
        meta = {
            "model_id": args.model_id,
            "quant": args.quant,
            "dtype": args.compute_dtype,
            "layers": layer_idx_list,
            "last_k": args.last_k,
            "prompt": args.prompt,
            "chat_template": args.chat_template,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.do_sample,
            "text": text,
        }
        save_run(args.save_dir, act, layer_idx_list, input_ids, gen_ids, meta)


if __name__ == "__main__":
    main()
