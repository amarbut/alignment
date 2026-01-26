"""
Model Card for openai/gpt-oss-20b (non-unsloth variant).

This card handles the standard HuggingFace loading of GPT-OSS-20B
without PEFT/Unsloth wrapping.
"""

import torch
import torch.nn as nn
from pipeline.model_utils.model_card import ModelCard


class OSSModelCard(ModelCard):
    """Model card for openai/gpt-oss-20b (non-unsloth)."""

    def _navigate_to_layers(self):
        """Direct access for non-unsloth OSS."""
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            return self.model.model.layers
        elif hasattr(self.model, 'layers'):
            return self.model.layers
        raise AttributeError(f"Could not find layers in model: {type(self.model)}")

    def get_num_experts(self, layer_idx: int) -> int:
        """OSS has 32 experts per layer."""
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

        For standard OSS, the bias is directly on the router.
        Creates bias if it doesn't exist.

        Args:
            router: Router module
            layer_idx: Layer index (not used for OSS, but kept for API consistency)
        """
        if hasattr(router, 'bias') and router.bias is not None:
            return router.bias
        else:
            # Create bias if missing
            num_experts = 32
            bias = torch.zeros(num_experts, device=router.weight.device,
                             dtype=router.weight.dtype)
            router.bias = nn.Parameter(bias)
            return router.bias

    def set_router_bias(self, router: nn.Module, bias: torch.Tensor, layer_idx: int = None):
        """
        Set router bias parameter.

        Args:
            router: Router module
            bias: New bias values
            layer_idx: Layer index (not used for OSS, but kept for API consistency)
        """
        if not hasattr(router, 'bias') or router.bias is None:
            router.bias = nn.Parameter(bias)
        else:
            router.bias.data = bias

    def is_moe_layer(self, layer_idx: int) -> bool:
        """All OSS layers are MoE."""
        return True

    def get_expert_routing_mode(self) -> str:
        """OSS uses top-2 routing."""
        return "top-2"

    def get_expert_diffs_filename(self) -> str:
        """Return filename for OSS expert diffs."""
        return "oss_expert_diffs.json"
