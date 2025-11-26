"""
Extract expert routing patterns for harmful vs harmless prompts using hooks.

This script analyzes which experts are selected in the OSS-20B MoE model
for the last 5 tokens of harmful and harmless prompts.
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
from pipeline.model_utils.model_factory import construct_model_base
import numpy as np

# Global storage for router outputs
router_outputs = {}

def create_mlp_hook(layer_idx):
    """Create a hook to capture router logits from MLP output for a specific layer."""
    def hook(module, input, output):
        # MLP output is a tuple: (hidden_states, router_logits)
        # router_logits shape: (seq_len, num_experts)
        if isinstance(output, tuple) and len(output) >= 2:
            router_logits = output[1]  # Second element is router logits
            router_outputs[layer_idx] = router_logits.detach().cpu()
    return hook

def load_dataset(path):
    """Load a dataset from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def extract_expert_routing(model_base, prompts, label, batch_size=4, last_n_tokens=5):
    """
    Extract expert routing information for a set of prompts.

    Args:
        model_base: The model wrapper
        prompts: List of prompt strings
        label: Label for this dataset ('harmful' or 'harmless')
        batch_size: Batch size for processing
        last_n_tokens: Number of tokens from the end to analyze

    Returns:
        List of dictionaries containing routing information
    """
    global router_outputs
    results = []

    # Register hooks on all MLPs to capture router logits
    hooks = []
    for layer_idx, layer in enumerate(model_base.model.model.layers):
        hook = layer.mlp.register_forward_hook(create_mlp_hook(layer_idx))
        hooks.append(hook)

    try:
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Processing {label} prompts"):
            batch_prompts = prompts[i:i + batch_size]

            # Tokenize prompts
            tokenized = model_base.tokenize_instructions_fn(instructions=batch_prompts)
            input_ids = tokenized.input_ids.to(model_base.model.device)
            attention_mask = tokenized.attention_mask.to(model_base.model.device)

            # Clear previous router outputs
            router_outputs = {}

            # Forward pass (hooks will capture router outputs)
            try:
                with torch.no_grad():
                    _ = model_base.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                    )
            except (RuntimeError, TypeError) as e:
                # Model may throw device or dtype errors at the end, but hooks still capture data
                if len(router_outputs) == 0:
                    # If no data was captured, re-raise the error
                    raise
                # Otherwise continue - we got the router logits we need

            # Process each sample in the batch
            for batch_idx in range(len(batch_prompts)):
                # Get the sequence length for this sample (accounting for padding)
                seq_len = attention_mask[batch_idx].sum().item()

                # Focus on the last N tokens
                start_pos = max(0, seq_len - last_n_tokens)
                end_pos = seq_len

                # Extract routing info for each MoE layer
                layer_routing = {}
                for layer_idx in sorted(router_outputs.keys()):
                    # router_logits shape from MLP: (batch_size * seq_len, num_experts)
                    # Need to reshape to (batch_size, seq_len, num_experts)
                    router_logits_flat = router_outputs[layer_idx]
                    batch_size_cur = len(batch_prompts)
                    seq_len_padded = input_ids.shape[1]
                    num_experts = router_logits_flat.shape[-1]

                    router_logits = router_logits_flat.view(batch_size_cur, seq_len_padded, num_experts)
                    layer_router = router_logits[batch_idx, start_pos:end_pos, :]

                    # Get top expert indices for each token position
                    top_experts = torch.argmax(layer_router, dim=-1).cpu().numpy()

                    # Get top-4 experts (since the model uses 4 experts per token)
                    routing_probs = torch.nn.functional.softmax(layer_router, dim=-1)
                    top_k_probs, top_k_experts = torch.topk(
                        routing_probs,
                        k=min(4, layer_router.shape[-1]),
                        dim=-1
                    )

                    # Store full routing distribution for later analysis
                    # Convert to float32 to avoid BFloat16 conversion issues
                    layer_routing[f"layer_{layer_idx}"] = {
                        "top_expert": top_experts.tolist(),  # Single top expert per token
                        "top_k_experts": top_k_experts.cpu().numpy().tolist(),  # Top-4 experts per token
                        "top_k_probs": top_k_probs.float().cpu().numpy().tolist(),  # Their probabilities
                        "num_experts": num_experts
                    }

                results.append({
                    "prompt": batch_prompts[batch_idx],
                    "label": label,
                    "prompt_length": seq_len,
                    "analyzed_positions": list(range(start_pos, end_pos)),
                    "layer_routing": layer_routing
                })

    finally:
        # Remove hooks
        for hook in hooks:
            hook.remove()

    return results

def main():
    # Configuration
    model_path = "openai/gpt-oss-20b"
    output_dir = Path("/media/volume/align_2_stg/alignment/expert_routing_analysis")
    output_dir.mkdir(exist_ok=True)

    # Load datasets
    print("Loading datasets...")
    harmful_data = load_dataset("dataset/splits/harmful_train.json")
    harmless_data = load_dataset("dataset/splits/harmless_train.json")

    harmful_prompts = [item["instruction"] for item in harmful_data]
    harmless_prompts = [item["instruction"] for item in harmless_data]

    print(f"Loaded {len(harmful_prompts)} harmful prompts and {len(harmless_prompts)} harmless prompts")

    # Load model
    print(f"Loading model: {model_path}")
    model_base = construct_model_base(model_path)

    # Get model info
    num_layers = len(model_base.model.model.layers)
    num_experts = model_base.model.config.num_local_experts
    experts_per_token = model_base.model.config.experts_per_token

    print(f"Model info:")
    print(f"  Number of layers: {num_layers}")
    print(f"  Number of experts per layer: {num_experts}")
    print(f"  Experts activated per token: {experts_per_token}")

    # Extract routing for harmful prompts
    print("\nExtracting routing patterns for harmful prompts...")
    harmful_results = extract_expert_routing(
        model_base,
        harmful_prompts,
        label="harmful",
        batch_size=4,
        last_n_tokens=5
    )

    # Extract routing for harmless prompts
    print("\nExtracting routing patterns for harmless prompts...")
    harmless_results = extract_expert_routing(
        model_base,
        harmless_prompts,
        label="harmless",
        batch_size=4,
        last_n_tokens=5
    )

    # Combine results
    all_results = {
        "model": model_path,
        "num_layers": num_layers,
        "num_experts": num_experts,
        "experts_per_token": experts_per_token,
        "num_harmful": len(harmful_results),
        "num_harmless": len(harmless_results),
        "last_n_tokens": 5,
        "harmful_results": harmful_results,
        "harmless_results": harmless_results
    }

    # Save results
    output_file = output_dir / "expert_routing_data.json"
    print(f"\nSaving results to {output_file}")
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Model: {model_path}")
    print(f"Number of layers: {num_layers}")
    print(f"Number of experts per layer: {num_experts}")
    print(f"Experts activated per token: {experts_per_token}")
    print(f"Processed {len(harmful_results)} harmful prompts")
    print(f"Processed {len(harmless_results)} harmless prompts")
    print(f"Results saved to: {output_file}")
    print(f"File size: {output_file.stat().st_size / (1024*1024):.2f} MB")
    print("\nYou can now download this file to your local machine for analysis.")
    print(f"\nTo analyze: python analyze_expert_routing.py {output_file}")

if __name__ == "__main__":
    main()
