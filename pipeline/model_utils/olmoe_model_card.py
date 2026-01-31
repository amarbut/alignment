"""
Model Card for OLMoE-1B-7B-Instruct.

Special features:
- 64 experts per MoE layer
- Top-8 routing (8 of 64 experts per token)
- 16 layers total, all are MoE
- Router is .gate (Linear), returns logits
- MLP output is (hidden_states, router_logits) tuple
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from pipeline.model_utils.model_card import ModelCard


class OLMoEModelCard(ModelCard):
    """Model card for OLMoE-1B-7B-Instruct."""

    def _navigate_to_layers(self):
        """Navigate to model layers."""
        return self.model.model.layers

    def get_expert_routing_mode(self) -> str:
        """OLMoE uses top-8 routing."""
        return "top-8"

    def get_num_layers(self) -> int:
        """Return number of layers."""
        return len(self.layers)

    def is_moe_layer(self, layer_idx: int) -> bool:
        """All OLMoE layers are MoE."""
        return True

    def get_num_experts(self, layer_idx: int) -> int:
        """Return number of experts (64 for OLMoE)."""
        return 64

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        """Get MLP (OlmoeSparseMoeBlock) for a layer."""
        return self.layers[layer_idx].mlp

    def get_router(self, layer_idx: int) -> nn.Module:
        """Get router (gate) for a layer."""
        mlp = self.get_mlp_module(layer_idx)
        return mlp.gate

    def get_router_bias(self, router: nn.Module, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get or create router bias for expert forcing.

        OLMoE's gate is a Linear layer without bias by default.
        We add a bias parameter if it doesn't exist.
        """
        if not hasattr(router, 'bias') or router.bias is None:
            num_experts = router.weight.shape[0]
            bias = torch.zeros(num_experts, dtype=router.weight.dtype, device=router.weight.device)
            router.register_parameter('bias', nn.Parameter(bias))
        return router.bias

    def set_router_bias(self, router: nn.Module, bias: torch.Tensor, layer_idx: Optional[int] = None):
        """Set router bias."""
        if not hasattr(router, 'bias') or router.bias is None:
            self.get_router_bias(router, layer_idx)
        router.bias.data = bias.to(router.weight.device, router.weight.dtype)

    def parse_mlp_output(self, output) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parse MLP output into (hidden_states, router_logits).

        OLMoE returns tuple: (hidden_states [batch, seq, hidden], router_logits [batch, num_experts])
        """
        if isinstance(output, tuple) and len(output) >= 2:
            hidden_states = output[0]
            router_logits = output[1]
            return hidden_states, router_logits
        return output, None

    def wrap_mlp_output(self, hidden_states: torch.Tensor,
                        router_logits: Optional[torch.Tensor] = None):
        """Reconstruct MLP output in expected format."""
        if router_logits is not None:
            return (hidden_states, router_logits)
        return hidden_states

    def get_expert_diffs_filename(self) -> str:
        """Return filename for OLMoE expert diffs."""
        return "olmoe_expert_diffs.json"

    # OLMoE uses standard MLP hooks (returns router_logits in MLP output)
    def uses_router_hook_for_routing(self) -> bool:
        """OLMoE returns router_logits in MLP output, so use MLP hooks."""
        return False

    def get_router_output_format(self) -> str:
        """OLMoE returns logits from MLP output."""
        return "logits"

    def get_expert_steering_thresholds(self) -> dict:
        """
        Return thresholds for expert steering selection.

        OLMoE uses more permissive thresholds due to different routing dynamics.
        """
        return {
            "kl_threshold": 2.0,
            "steering_score_threshold": -20.0,
            "prune_layer_percentage": 0.0
        }
