"""
Model Card for unsloth/gpt-oss-20b-unsloth-bnb-4bit. and 120b

This card handles the Unsloth/PEFT wrapped version of GPT-OSS-20B,120B
which has different layer navigation and router access patterns.
"""

import torch
import torch.nn as nn
from model_utils.model_card import ModelCard


class UnslothOSSModelCard(ModelCard):
    """Model card for unsloth/gpt-oss-20b-unsloth-bnb-4bit."""

    def _navigate_to_layers(self):
        """
        Navigate PEFT/Unsloth wrapping to find layers.

        Unsloth models are wrapped in PEFT layers, requiring navigation
        through model.model or base_model to reach the actual layers.
        """
        current = self.model

        # Navigate through PEFT/unsloth wrappers
        # Try various paths that unsloth may use
        while hasattr(current, 'model') and not hasattr(current, 'layers'):
            current = current.model

        if hasattr(current, 'layers'):
            return current.layers

        # Alternative: try base_model path
        current = self.model
        while hasattr(current, 'base_model'):
            current = current.base_model
            if hasattr(current, 'layers'):
                return current.layers

        raise AttributeError(f"Could not find layers in unsloth model: {type(self.model)}")

    def get_num_experts(self, layer_idx: int) -> int:
        """OSS-20B has 32 experts per layer. OSS-120B has 128 experts per layer"""
        if '120b' in str(self.model_base.model_name_or_path).lower():
            return 128
        else:
            return 32

    def get_num_layers(self) -> int:
        """Return total number of transformer layers."""
        return len(self.layers)

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        """Get MLP module at specified layer."""
        return self.layers[layer_idx].mlp

    def get_router(self, layer_idx: int) -> nn.Module:
        """Get router module at specified layer."""
        return self.layers[layer_idx].mlp.router

    def get_router_bias(self, router: nn.Module, layer_idx: int = None) -> torch.Tensor:
        """
        Get router bias parameter.

        Unsloth wraps the router in a linear layer, so bias may be at:
        - router.linear.bias (most common for unsloth)
        - router.bias (fallback)

        Args:
            router: Router module
            layer_idx: Layer index (not used for unsloth OSS, but kept for API consistency)
        """
        # Try unsloth-wrapped path first
        if hasattr(router, 'linear') and hasattr(router.linear, 'bias'):
            if router.linear.bias is not None:
                return router.linear.bias
            else:
                # Create bias on the linear layer
                num_experts = 32
                bias = torch.zeros(num_experts, device=router.linear.weight.device,
                                 dtype=router.linear.weight.dtype)
                router.linear.bias = nn.Parameter(bias)
                return router.linear.bias

        # Fallback to direct router.bias
        elif hasattr(router, 'bias') and router.bias is not None:
            return router.bias

        else:
            # Create at router level as last resort
            num_experts = 32
            # Find weight to get device/dtype
            if hasattr(router, 'linear') and hasattr(router.linear, 'weight'):
                weight = router.linear.weight
            elif hasattr(router, 'weight'):
                weight = router.weight
            else:
                raise AttributeError(f"Could not find weight in router: {type(router)}")

            bias = torch.zeros(num_experts, device=weight.device, dtype=weight.dtype)
            router.bias = nn.Parameter(bias)
            return router.bias

    def set_router_bias(self, router: nn.Module, bias: torch.Tensor, layer_idx: int = None):
        """
        Set router bias parameter.

        Handles both unsloth-wrapped (router.linear.bias) and standard paths.

        Args:
            router: Router module
            bias: New bias values
            layer_idx: Layer index (not used for unsloth OSS, but kept for API consistency)
        """
        if hasattr(router, 'linear') and hasattr(router.linear, 'bias'):
            if router.linear.bias is not None:
                router.linear.bias.data = bias
            else:
                router.linear.bias = nn.Parameter(bias)
        elif hasattr(router, 'bias') and router.bias is not None:
            router.bias.data = bias
        else:
            router.bias = nn.Parameter(bias)

    def is_moe_layer(self, layer_idx: int) -> bool:
        """All OSS layers are MoE."""
        return True

    def get_expert_routing_mode(self) -> str:
        """OSS uses top-2 routing."""
        return "top-2"

    def get_expert_diffs_filename(self) -> str:
        """Return filename for OSS expert diffs (same as standard OSS, same model)."""
        return "oss_expert_diffs.json"

    def get_expert_steering_thresholds(self) -> dict:
        """
        Return thresholds for expert steering selection.

        OSS models use less aggressive thresholds than Mixtral/OLMoE/DeepSeek.
        """
        return {
            "kl_threshold": 1.0,
            "steering_score_threshold": -20.0,
            "prune_layer_percentage": 0.0,
            "expert_diff_threshold": 10.0,
        }
