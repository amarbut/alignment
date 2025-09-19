#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 18:09:11 2025

@author: anna
"""

#!/usr/bin/env python3

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Use chat template for instruct models
    messages = [{"role": "user", "content": "Why is the sky blue?"}]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
        )

    # Decode and print
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(output_text)

if __name__ == "__main__":
    main()
