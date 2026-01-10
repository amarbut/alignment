"""
Stochastic Expert Dropout for MoE Models

This module implements stochastic expert dropout during fine-tuning to diffuse
refusal responsibility across experts, making models more robust to expert-based
jailbreaking attacks.

The dropout is applied by masking router logits in the MLP output tuple during
the forward pass, ensuring different experts are randomly suppressed each iteration.

Based on the router logit access pattern from extract_beavertails_routing.py,
we hook into the MLP module and modify the (hidden_states, router_logits) tuple.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Set
import numpy as np


class StochasticExpertDropout:
    """
    Applies stochastic dropout to MoE experts during training.

    During each forward pass, randomly masks n% of experts by setting their
    router logits to a very negative value, preventing them from being selected.

    This forces the model to distribute capabilities (like refusal) across
    multiple experts rather than specializing individual experts.
    """

    def __init__(
        self,
        model,
        dropout_rate: float = 0.3,
        mask_value: float = -1e9,
        exclude_layers: Optional[List[int]] = None,
        exclude_experts: Optional[Set[int]] = None,
        seed: Optional[int] = None
    ):
        """
        Initialize expert dropout.

        Args:
            model: The transformer model (expects model.model.layers structure)
            dropout_rate: Fraction of experts to drop per layer (0.0 to 1.0)
            mask_value: Value to set for masked expert logits (large negative number)
            exclude_layers: Optional list of layer indices to exclude from dropout
            exclude_experts: Optional set of expert indices to never drop
            seed: Random seed for reproducibility (if None, uses random state)
        """
        self.model = model
        self.dropout_rate = dropout_rate
        self.mask_value = mask_value
        self.exclude_layers = set(exclude_layers or [])
        self.exclude_experts = exclude_experts or set()
        self.seed = seed

        self.hooks = []
        self.enabled = False

        # Get model architecture info
        self.num_layers = len(model.model.layers)
        self.num_experts = model.config.num_local_experts

        print(f"Initialized StochasticExpertDropout:")
        print(f"  Dropout rate: {dropout_rate:.1%}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Num experts: {self.num_experts}")
        print(f"  Excluded layers: {sorted(self.exclude_layers) if self.exclude_layers else 'None'}")
        print(f"  Excluded experts: {sorted(self.exclude_experts) if self.exclude_experts else 'None'}")

    def _create_dropout_hook(self, layer_idx: int):
        """
        Create a pre-forward hook that applies expert dropout by temporarily
        modifying router weights.

        Since OSS-20B computes router logits inline (not via router.forward()),
        we modify the router bias before the MLP forward pass.
        """
        original_bias = [None]  # Store original bias

        def pre_hook(module, input):
            # Only apply dropout during training
            if not self.enabled or not self.model.training:
                return

            # Skip excluded layers
            if layer_idx in self.exclude_layers:
                return

            # Determine how many experts to drop
            num_to_drop = int(self.num_experts * self.dropout_rate)

            if num_to_drop == 0:
                return

            # Get all expert indices
            all_experts = set(range(self.num_experts))
            available_experts = all_experts - self.exclude_experts

            # If we can't drop enough experts due to exclusions, drop what we can
            num_to_drop = min(num_to_drop, len(available_experts))

            if num_to_drop == 0:
                return

            # Randomly select experts to drop
            dropped_experts = np.random.choice(
                list(available_experts),
                size=num_to_drop,
                replace=False
            )

            # Save original bias and modify it
            router = module.router
            original_bias[0] = router.bias.data.clone()

            # Create modified bias with dropped experts masked
            modified_bias = router.bias.data.clone()
            modified_bias[dropped_experts] = self.mask_value
            router.bias.data = modified_bias

        def post_hook(module, input, output):
            # Restore original bias after forward pass
            if original_bias[0] is not None:
                module.router.bias.data = original_bias[0]
                original_bias[0] = None
            return output

        return pre_hook, post_hook

    def enable(self):
        """Enable expert dropout by registering forward hooks."""
        if self.enabled:
            return

        print(f"Enabling StochasticExpertDropout on {self.num_layers} layers...")

        # Set random seed if specified
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # Register hooks on MLP modules (since router.forward() isn't called)
        for layer_idx in range(self.num_layers):
            if layer_idx in self.exclude_layers:
                continue

            mlp = self.model.model.layers[layer_idx].mlp
            pre_hook, post_hook = self._create_dropout_hook(layer_idx)
            pre_handle = mlp.register_forward_pre_hook(pre_hook)
            post_handle = mlp.register_forward_hook(post_hook)
            self.hooks.append(pre_handle)
            self.hooks.append(post_handle)

        self.enabled = True
        print(f"Registered {len(self.hooks)} dropout hooks")

    def disable(self):
        """Disable expert dropout by removing all hooks."""
        if not self.enabled:
            return

        print("Disabling StochasticExpertDropout...")

        for hook in self.hooks:
            hook.remove()

        self.hooks = []
        self.enabled = False
        print("Removed all dropout hooks")

    def __enter__(self):
        """Context manager entry - enable dropout."""
        self.enable()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - disable dropout."""
        self.disable()
        return False


class PerTokenExpertDropout(StochasticExpertDropout):
    """
    Variant that applies independent dropout to each token position.

    Note: This is NOT implemented for OSS-20B's fused experts since we can't
    apply per-token bias modifications. Falls back to per-batch dropout.
    """

    def __init__(self, *args, **kwargs):
        print("Warning: Per-token dropout not supported for OSS-20B fused experts.")
        print("Falling back to per-batch dropout.")
        super().__init__(*args, **kwargs)


def test_expert_dropout():
    """Test the expert dropout implementation."""
    import sys
    import os

    # Add alignment directory to path
    alignment_dir = os.path.dirname(os.path.abspath(__file__))
    if alignment_dir not in sys.path:
        sys.path.insert(0, alignment_dir)

    from pipeline.model_utils.model_factory import construct_model_base

    print("="*80)
    print("Testing StochasticExpertDropout")
    print("="*80)

    # Load model
    print("\nLoading model...")
    model_base = construct_model_base("openai/gpt-oss-20b")
    model = model_base.model

    # Create dropout module
    dropout = StochasticExpertDropout(
        model=model,
        dropout_rate=0.3,
        exclude_layers=[0, 1],  # Don't drop experts in first 2 layers
        exclude_experts={0, 1}  # Never drop experts 0 and 1
    )

    # Test with context manager
    print("\n" + "="*80)
    print("Test 1: Context manager usage")
    print("="*80)

    # Create test input
    test_text = "Hello, how are you?"
    inputs = model_base.tokenizer(test_text, return_tensors="pt").to(model.device)

    print("\nRunning inference WITHOUT dropout...")
    model.eval()
    with torch.no_grad():
        outputs1 = model(**inputs)

    print("Running inference WITH dropout (should have no effect in eval mode)...")
    with dropout:
        model.eval()
        with torch.no_grad():
            outputs2 = model(**inputs)

    print(f"Outputs identical in eval mode: {torch.allclose(outputs1.logits, outputs2.logits)}")

    print("\nRunning forward pass WITH dropout in training mode...")
    with dropout:
        model.train()
        # Note: we still use no_grad since we're not optimizing, just testing
        with torch.no_grad():
            outputs3 = model(**inputs)

    model.eval()  # Set back to eval
    print(f"Training mode outputs different from eval: {not torch.allclose(outputs1.logits, outputs3.logits)}")

    # Test manual enable/disable
    print("\n" + "="*80)
    print("Test 2: Manual enable/disable")
    print("="*80)

    dropout.enable()
    print("Dropout enabled")

    model.train()
    with torch.no_grad():
        outputs4 = model(**inputs)

    dropout.disable()
    print("Dropout disabled")

    print(f"Hooks removed successfully: {len(dropout.hooks) == 0}")

    print("\n" + "="*80)
    print("Test 3: Per-token dropout")
    print("="*80)

    per_token_dropout = PerTokenExpertDropout(
        model=model,
        dropout_rate=0.3
    )

    with per_token_dropout:
        model.train()
        with torch.no_grad():
            outputs5 = model(**inputs)

    model.eval()  # Set back to eval
    print(f"Per-token dropout produces different outputs: {not torch.allclose(outputs1.logits, outputs5.logits)}")

    print("\n" + "="*80)
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    test_expert_dropout()
