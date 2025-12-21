"""
Stochastic Expert Dropout for MoE Models (Unsloth Version)

This module implements stochastic expert dropout during fine-tuning using unsloth models.
Adapted to work with unsloth's model structure and optimizations.

The dropout is applied by masking router logits before expert selection, ensuring
different experts are randomly suppressed each iteration.
"""

import torch
import torch.nn as nn
from typing import Optional, List, Set
import numpy as np


class StochasticExpertDropoutUnsloth:
    """
    Applies stochastic dropout to MoE experts during training with unsloth models.

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
        Initialize expert dropout for unsloth models.

        Args:
            model: The unsloth FastLanguageModel or base model
            dropout_rate: Fraction of experts to drop per layer (0.0 to 1.0)
            mask_value: Value to set for masked expert logits (large negative number)
            exclude_layers: Optional list of layer indices to exclude from dropout
            exclude_experts: Optional set of expert indices to never drop
            seed: Random seed for reproducibility (if None, uses random state)
        """
        # Unwrap unsloth/PEFT model if needed
        # Keep unwrapping until we find the model with 'layers' attribute
        self.model = model
        while hasattr(self.model, 'model') and not hasattr(self.model, 'layers'):
            self.model = self.model.model

        self.dropout_rate = dropout_rate
        self.mask_value = mask_value
        self.exclude_layers = set(exclude_layers or [])
        self.exclude_experts = exclude_experts or set()
        self.seed = seed

        self.hooks = []
        self.enabled = False

        # Get model architecture info
        self.num_layers = len(self.model.layers)
        self.num_experts = self.model.config.num_local_experts

        print(f"Initialized StochasticExpertDropoutUnsloth:")
        print(f"  Dropout rate: {dropout_rate:.1%}")
        print(f"  Num layers: {self.num_layers}")
        print(f"  Num experts: {self.num_experts}")
        print(f"  Excluded layers: {sorted(self.exclude_layers) if self.exclude_layers else 'None'}")
        print(f"  Excluded experts: {sorted(self.exclude_experts) if self.exclude_experts else 'None'}")

    def _create_dropout_hook(self, layer_idx: int):
        """
        Create forward hook that applies expert dropout by masking router output logits.
        This is safer than modifying bias.data as it doesn't interfere with autograd.
        """
        dropped_experts = [None]  # Store which experts to drop for this forward pass

        def router_hook(router_module, input, output):
            """Hook on the router itself to modify its output logits."""
            # Only apply dropout during training
            if not self.enabled or not router_module.training:
                return output

            # Skip excluded layers
            if layer_idx in self.exclude_layers:
                return output

            # Determine how many experts to drop
            num_to_drop = int(self.num_experts * self.dropout_rate)

            if num_to_drop == 0:
                return output

            # Get all expert indices
            all_experts = set(range(self.num_experts))
            available_experts = all_experts - self.exclude_experts

            # If we can't drop enough experts due to exclusions, drop what we can
            num_to_drop = min(num_to_drop, len(available_experts))

            if num_to_drop == 0:
                return output

            # Randomly select experts to drop
            dropped = np.random.choice(
                list(available_experts),
                size=num_to_drop,
                replace=False
            )

            # The output from the router is a tuple: (router_scores, router_indices)
            # Apply dropout like traditional dropout: binary mask + renormalization
            if isinstance(output, tuple):
                router_scores, router_indices = output
                # Clone to avoid in-place modification
                masked_scores = router_scores.clone()

                # Create binary mask: 0 for dropped experts, 1 for kept experts
                mask = torch.ones_like(masked_scores)
                mask[..., dropped] = 0.0

                # Apply mask
                masked_scores = masked_scores * mask

                # Renormalize so probabilities sum to 1
                # Add small epsilon to avoid division by zero
                score_sum = masked_scores.sum(dim=-1, keepdim=True)
                masked_scores = masked_scores / (score_sum + 1e-10)

                return (masked_scores, router_indices)
            else:
                # If it's just a tensor, apply binary mask + renormalization
                masked_output = output.clone()
                mask = torch.ones_like(masked_output)
                mask[..., dropped] = 0.0
                masked_output = masked_output * mask
                score_sum = masked_output.sum(dim=-1, keepdim=True)
                masked_output = masked_output / (score_sum + 1e-10)
                return masked_output

        return router_hook

    def enable(self):
        """Enable expert dropout by registering forward hooks."""
        if self.enabled:
            return

        print(f"Enabling StochasticExpertDropoutUnsloth on {self.num_layers} layers...")

        # Set random seed if specified
        if self.seed is not None:
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)

        # Register hooks on router modules (not MLP)
        # The router outputs logits, which we'll mask to drop experts
        for layer_idx in range(self.num_layers):
            if layer_idx in self.exclude_layers:
                continue

            # Get the router module
            mlp = self.model.layers[layer_idx].mlp
            if hasattr(mlp, 'router'):
                router = mlp.router
                # Create and register the hook
                router_hook = self._create_dropout_hook(layer_idx)
                handle = router.register_forward_hook(router_hook)
                self.hooks.append(handle)

        self.enabled = True
        print(f"Registered {len(self.hooks)} dropout hooks")

    def disable(self):
        """Disable expert dropout by removing all hooks."""
        if not self.enabled:
            return

        print("Disabling StochasticExpertDropoutUnsloth...")

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


class PerTokenExpertDropoutUnsloth(StochasticExpertDropoutUnsloth):
    """
    Variant that would apply independent dropout to each token position.

    Note: Per-token dropout is not currently implemented for fused experts.
    Falls back to per-batch dropout.
    """

    def __init__(self, *args, **kwargs):
        print("Warning: Per-token dropout not supported for fused MoE experts.")
        print("Falling back to per-batch dropout.")
        super().__init__(*args, **kwargs)


def test_expert_dropout_unsloth():
    """Test the expert dropout implementation with unsloth."""
    import sys
    import os

    # Add alignment directory to path
    alignment_dir = os.path.dirname(os.path.abspath(__file__))
    if alignment_dir not in sys.path:
        sys.path.insert(0, alignment_dir)

    # Fix HF cache paths
    import fix_hf_cache

    from unsloth import FastLanguageModel

    print("="*80)
    print("Testing StochasticExpertDropoutUnsloth")
    print("="*80)

    # Load model using unsloth
    print("\nLoading model with unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        max_seq_length=512,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    print(f"Model loaded: {type(model).__name__}")

    # Create dropout module
    dropout = StochasticExpertDropoutUnsloth(
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
    inputs = tokenizer(test_text, return_tensors="pt").to(model.device)

    print("\nRunning inference WITHOUT dropout...")
    model.eval()
    with torch.no_grad():
        outputs1 = model(**inputs)

    print("Running inference WITH dropout (should have no effect in eval mode)...")
    with dropout:
        model.eval()
        with torch.no_grad():
            outputs2 = model(**inputs)

    print(f"Outputs identical in eval mode: {torch.allclose(outputs1.logits, outputs2.logits, rtol=1e-3)}")

    print("\nRunning forward pass WITH dropout in training mode...")
    with dropout:
        model.train()
        with torch.no_grad():
            outputs3 = model(**inputs)

    model.eval()  # Set back to eval
    print(f"Training mode outputs different from eval: {not torch.allclose(outputs1.logits, outputs3.logits, rtol=1e-3)}")

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
    print("All tests completed!")
    print("="*80)


if __name__ == "__main__":
    test_expert_dropout_unsloth()
