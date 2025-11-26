"""
Test version of expert routing extraction with hooks.
"""

import torch
import json
from pathlib import Path
from pipeline.model_utils.model_factory import construct_model_base

router_outputs = {}

def create_router_hook(layer_idx):
    """Create a hook to capture router logits for a specific layer."""
    def hook(module, input, output):
        # Router output - need to check what it returns
        if isinstance(output, tuple):
            # Try to get router logits from tuple
            for i, item in enumerate(output):
                if isinstance(item, torch.Tensor):
                    print(f"    Output[{i}] shape: {item.shape}")
            router_logits = output[0] if len(output) > 0 else None
        else:
            router_logits = output
            print(f"    Router output shape: {router_logits.shape}")

        if router_logits is not None:
            router_outputs[layer_idx] = router_logits.detach().cpu()
    return hook

def test_expert_routing():
    """Test expert routing extraction with hooks."""

    # Load small test datasets
    print("Loading test datasets...")
    with open("dataset/splits/harmful_test_small.json", 'r') as f:
        harmful_data = json.load(f)

    harmful_prompts = [item["instruction"] for item in harmful_data]
    print(f"Loaded {len(harmful_prompts)} harmful prompts")

    # Load model
    model_path = "openai/gpt-oss-20b"
    print(f"\nLoading model: {model_path}")
    model_base = construct_model_base(model_path)
    print("Model loaded successfully!")

    # Register hooks on all MLP routers
    print("\nRegistering hooks on router modules...")
    hooks = []
    for layer_idx, layer in enumerate(model_base.model.model.layers):
        if hasattr(layer.mlp, 'router'):
            hook = layer.mlp.router.register_forward_hook(create_router_hook(layer_idx))
            hooks.append(hook)
    print(f"Registered {len(hooks)} hooks")

    # Test with one prompt
    print("\n" + "="*60)
    print("Testing with first harmful prompt:")
    print("="*60)
    print(f"Prompt: {harmful_prompts[0][:100]}...")

    # Tokenize
    tokenized = model_base.tokenize_instructions_fn(instructions=[harmful_prompts[0]])
    input_ids = tokenized.input_ids.to(model_base.model.device)
    attention_mask = tokenized.attention_mask.to(model_base.model.device)

    print(f"Tokenized length: {input_ids.shape}")

    # Forward pass
    print("\nRunning forward pass with hooks...")
    global router_outputs
    router_outputs = {}

    with torch.no_grad():
        _ = model_base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    # Check captured outputs
    print(f"\nCaptured router outputs from {len(router_outputs)} layers")

    if router_outputs:
        print("\n✓ Success! Router outputs captured via hooks.")

        # Analyze first layer
        layer_0_router = router_outputs[0]
        print(f"\nLayer 0 router output shape: {layer_0_router.shape}")
        print(f"  Format: (batch_size={layer_0_router.shape[0]}, seq_len={layer_0_router.shape[1]}, num_experts={layer_0_router.shape[2]})")

        # Get last 5 tokens
        seq_len = attention_mask[0].sum().item()
        last_5_router = layer_0_router[0, max(0, seq_len-5):seq_len, :]
        print(f"\nLast 5 tokens routing (Layer 0):")

        # Get top experts
        top_experts = torch.argmax(last_5_router, dim=-1)
        routing_probs = torch.nn.functional.softmax(last_5_router, dim=-1)

        for tok_idx, (expert_id, probs) in enumerate(zip(top_experts, routing_probs)):
            top_4_probs, top_4_experts = torch.topk(probs, 4)
            print(f"  Token {tok_idx}: Top expert = {expert_id.item()}")
            print(f"    Top 4 experts: {top_4_experts.numpy()}")
            print(f"    Top 4 probs: {top_4_probs.numpy()}")

        print("\n✓ All checks passed! You can now run the full extraction:")
        print("  python extract_expert_routing_v2.py")
    else:
        print("\n✗ Error: No router outputs captured!")
        print("Need to debug the hook function.")

    # Clean up hooks
    for hook in hooks:
        hook.remove()

if __name__ == "__main__":
    test_expert_routing()
