"""
Test the full extraction pipeline with small dataset.
"""

import torch
import json
from pathlib import Path
from tqdm import tqdm
from pipeline.model_utils.model_factory import construct_model_base

# Global storage for router outputs
router_outputs = {}

def create_mlp_hook(layer_idx):
    """Create a hook to capture router logits from MLP output for a specific layer."""
    def hook(module, input, output):
        # MLP output is a tuple: (hidden_states, router_logits)
        # router_logits shape: (batch_size * seq_len, num_experts)
        if isinstance(output, tuple) and len(output) >= 2:
            router_logits = output[1]  # Second element is router logits
            router_outputs[layer_idx] = router_logits.detach().cpu()
    return hook

def load_dataset(path):
    """Load a dataset from JSON file."""
    with open(path, 'r') as f:
        return json.load(f)

def extract_expert_routing(model_base, prompts, label, batch_size=2, last_n_tokens=5):
    """Extract expert routing information for a set of prompts."""
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

                    # Store routing distribution
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
    # Load small test datasets
    print("Loading test datasets...")
    harmful_data = load_dataset("dataset/splits/harmful_test_small.json")
    harmless_data = load_dataset("dataset/splits/harmless_test_small.json")

    harmful_prompts = [item["instruction"] for item in harmful_data][:3]  # Just 3 samples
    harmless_prompts = [item["instruction"] for item in harmless_data][:3]

    print(f"Testing with {len(harmful_prompts)} harmful and {len(harmless_prompts)} harmless prompts")

    # Load model
    model_path = "openai/gpt-oss-20b"
    print(f"\nLoading model: {model_path}")
    model_base = construct_model_base(model_path)

    # Extract routing
    print("\nExtracting routing for harmful prompts...")
    harmful_results = extract_expert_routing(model_base, harmful_prompts, "harmful", batch_size=2)

    print("\nExtracting routing for harmless prompts...")
    harmless_results = extract_expert_routing(model_base, harmless_prompts, "harmless", batch_size=2)

    # Print sample results
    print("\n" + "="*60)
    print("SAMPLE RESULTS")
    print("="*60)

    if harmful_results:
        sample = harmful_results[0]
        print(f"\nFirst harmful prompt: {sample['prompt'][:80]}...")
        print(f"Prompt length: {sample['prompt_length']} tokens")
        print(f"Number of layers: {len(sample['layer_routing'])}")

        # Show layer 0 routing for last 5 tokens
        layer_0 = sample['layer_routing']['layer_0']
        print(f"\nLayer 0 routing (last 5 tokens):")
        print(f"  Top experts: {layer_0['top_expert']}")
        print(f"  Top-4 experts for first token: {layer_0['top_k_experts'][0]}")
        print(f"  Top-4 probs for first token: {[f'{p:.3f}' for p in layer_0['top_k_probs'][0]]}")

    print("\n✓ Test completed successfully!")
    print("\nYou can now run the full extraction with:")
    print("  python extract_expert_routing_v2.py")

if __name__ == "__main__":
    main()
