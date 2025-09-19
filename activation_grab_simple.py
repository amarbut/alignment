#!/usr/bin/env python3

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- load model/tokenizer (hardcoded for LLaMA-3.1-8B-Instruct)
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
model.eval()

# ---- build chat prompt (simple, using the tokenizer's chat template)
messages = [{"role": "user", "content": "Why is the sky blue?"}]
prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# ---- tokenize, generate, print
inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=200,
        temperature=0.7,
        top_p=0.9,
    )
text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(text)

# ---- collect activations at each layer for the last 5 generated tokens
last_k = 5
prompt_len = inputs["input_ids"].shape[1]
gen_ids = output_ids[0, prompt_len:].tolist()

# reconstruct full token ids (prompt + generated)
with torch.no_grad():
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)["input_ids"]
    gen_ids_tensor = torch.tensor([gen_ids], device=model.device, dtype=prompt_ids.dtype)
    full_ids = torch.cat([prompt_ids, gen_ids_tensor], dim=1)
    attn = torch.ones_like(full_ids, device=model.device)

# hardcode where the decoder layers live for LLaMA: model.model.layers
layers = model.model.layers

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

# ---- tiny summary print
L = len(layers)
H = next(iter(captures.values())).shape[-1] if captures else 0
print(f"\n[activations] layers captured: {len(captures)}/{L}; each is shape [{last_k}, {H}]")
# example: show layer 0 first row norm
if captures:
    ex = captures[0][0]
    print(f"[activations] example layer 0, token -5, L2 norm: {float(np.linalg.norm(ex)):.4f}")
