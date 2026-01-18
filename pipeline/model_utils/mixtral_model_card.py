"""
Model Card for Mixtral 8x7B.

Key differences from OSS:
- MLP module: block_sparse_moe (not mlp)
- Router: gate (not router)
- Router has NO BIAS by default - we dynamically add one
"""

import torch
import torch.nn as nn
from pipeline.model_utils.model_card import ModelCard


class MixtralModelCard(ModelCard):
    """Model card for Mixtral 8x7B."""

    def _navigate_to_layers(self):
        """Standard navigation for Mixtral."""
        current = self.model
        while hasattr(current, 'model') and not hasattr(current, 'layers'):
            current = current.model
        if hasattr(current, 'layers'):
            return current.layers
        raise AttributeError(f"Could not find layers in Mixtral model: {type(self.model)}")

    def get_num_experts(self, layer_idx: int) -> int:
        """Mixtral has 8 experts per layer."""
        return 8

    def get_num_layers(self) -> int:
        """Return total number of transformer layers."""
        return len(self.layers)

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        """
        Get MLP module at specified layer.

        KEY DIFFERENCE: Mixtral uses block_sparse_moe, not mlp.
        """
        return self.layers[layer_idx].block_sparse_moe

    def get_router(self, layer_idx: int) -> nn.Module:
        """
        Get router module at specified layer.

        KEY DIFFERENCE: Mixtral calls it 'gate', not 'router'.
        """
        return self.layers[layer_idx].block_sparse_moe.gate

    def get_router_bias(self, router: nn.Module) -> torch.Tensor:
        """
        Get router bias parameter.

        CRITICAL: Mixtral routers don't have bias by default.
        We dynamically add a bias parameter for expert forcing.

        If this causes issues (NaN, unexpected routing), alternative
        strategies are documented in the plan:
        - Weight temperature scaling
        - Pre-hook input shifting
        """
        if hasattr(router, 'bias') and router.bias is not None:
            return router.bias
        else:
            print("Note: Mixtral router has no bias, adding parameter for expert forcing")
            num_experts = 8
            bias = torch.zeros(num_experts, device=router.weight.device,
                             dtype=router.weight.dtype)
            router.bias = nn.Parameter(bias)
            return router.bias

    def set_router_bias(self, router: nn.Module, bias: torch.Tensor):
        """Set router bias parameter."""
        if not hasattr(router, 'bias') or router.bias is None:
            router.bias = nn.Parameter(bias)
        else:
            router.bias.data = bias

    def is_moe_layer(self, layer_idx: int) -> bool:
        """All Mixtral layers are MoE."""
        return True

    def get_expert_routing_mode(self) -> str:
        """Mixtral uses top-2 routing."""
        return "top-2"

    def get_expert_diffs_filename(self) -> str:
        """Return filename for Mixtral expert diffs."""
        return "mixtral_expert_diffs.json"
