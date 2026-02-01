"""
Model Card for DeepSeek-MoE-16B-Chat.

Special features:
- First layer is DENSE (not MoE)
- 64 routed + 2 shared experts per MoE layer
- Top-6 routing (6 of 64 routed experts per token)
- Standard gating network (linear projection + softmax)

KEY DIFFERENCE from OSS/Mixtral:
- Router returns (top_k_indices, top_k_weights, aux_loss) NOT raw logits
- MLP output is just a tensor, NOT (hidden_states, router_logits)
- Must hook router directly to capture routing decisions
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
from model_utils.model_card import ModelCard


class DeepSeekModelCard(ModelCard):
    """
    Model card for DeepSeek-MoE-16B-Chat.

    Note: This is for the original DeepSeek-MoE architecture,
    NOT the V2/V3 which uses more advanced routing mechanisms.
    """

    def __init__(self, model, model_base):
        # DeepSeek-specific config (set before super().__init__ calls methods)
        self.num_routed_experts = 64
        self.num_shared_experts = 2
        self.num_experts_per_tok = 6
        self.first_k_dense_replace = 1  # First layer is dense

        super().__init__(model, model_base)

    def _navigate_to_layers(self):
        """Standard navigation for DeepSeek."""
        current = self.model
        while hasattr(current, 'model') and not hasattr(current, 'layers'):
            current = current.model
        if hasattr(current, 'layers'):
            return current.layers
        raise AttributeError(f"Could not find layers in DeepSeek model: {type(self.model)}")

    def get_num_experts(self, layer_idx: int) -> int:
        """
        Return number of routed experts.

        Note: Does not include shared experts (which are always active).
        Returns 0 for dense layers.
        """
        if not self.is_moe_layer(layer_idx):
            return 0
        return self.num_routed_experts  # 64

    def get_num_layers(self) -> int:
        """Return total number of transformer layers (28)."""
        return len(self.layers)

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        """
        Get MLP/MoE module at specified layer.

        DeepSeek module name may vary - try common patterns.
        """
        layer = self.layers[layer_idx]

        # Try common names
        if hasattr(layer, 'mlp'):
            return layer.mlp
        elif hasattr(layer, 'moe'):
            return layer.moe
        elif hasattr(layer, 'ffn'):
            return layer.ffn
        else:
            raise AttributeError(
                f"Could not find MLP/MoE module in layer {layer_idx}. "
                f"Available attributes: {[a for a in dir(layer) if not a.startswith('_')]}"
            )

    def get_router(self, layer_idx: int) -> nn.Module:
        """
        Get router/gate module at specified layer.

        Raises error for dense layers (layer 0).
        """
        if not self.is_moe_layer(layer_idx):
            raise ValueError(f"Layer {layer_idx} is dense, has no router")

        mlp = self.get_mlp_module(layer_idx)

        # Try common names
        if hasattr(mlp, 'gate'):
            return mlp.gate
        elif hasattr(mlp, 'router'):
            return mlp.router
        elif hasattr(mlp, 'expert_gate'):
            return mlp.expert_gate
        else:
            raise AttributeError(
                f"Could not find router in layer {layer_idx} MLP. "
                f"Available attributes: {[a for a in dir(mlp) if not a.startswith('_')]}"
            )

    def get_router_bias(self, router: nn.Module, layer_idx: int = None) -> torch.Tensor:
        """
        Get router bias parameter.

        DeepSeek may not have bias by default (like Mixtral).
        Creates one if needed for expert forcing.

        Args:
            router: Router module
            layer_idx: Layer index (not used currently, but kept for API consistency)
        """
        if hasattr(router, 'bias') and router.bias is not None:
            return router.bias
        else:
            # Create bias
            bias = torch.zeros(self.num_routed_experts,
                             device=router.weight.device,
                             dtype=router.weight.dtype)
            router.bias = nn.Parameter(bias)
            return router.bias

    def set_router_bias(self, router: nn.Module, bias: torch.Tensor, layer_idx: int = None):
        """
        Set router bias parameter.

        Args:
            router: Router module
            bias: New bias values
            layer_idx: Layer index (not used currently, but kept for API consistency)
        """
        if not hasattr(router, 'bias') or router.bias is None:
            router.bias = nn.Parameter(bias)
        else:
            router.bias.data = bias

    def is_moe_layer(self, layer_idx: int) -> bool:
        """
        Check if layer is MoE.

        First layer (layer 0) is dense in DeepSeek-MoE architecture.
        """
        return layer_idx >= self.first_k_dense_replace

    def get_expert_routing_mode(self) -> str:
        """DeepSeek uses top-6 routing."""
        return "top-6"

    def get_expert_diffs_filename(self) -> str:
        """Return filename for DeepSeek expert diffs."""
        return "deepseek_expert_diffs.json"

    def get_router_output_format(self) -> str:
        """
        Return how router outputs routing information.

        DeepSeek returns 'top_k_indices' - the router outputs (indices, weights, aux)
        instead of raw logits for all experts.
        """
        return "top_k_indices"

    def parse_router_output(self, output) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Parse router output into (top_k_indices, top_k_weights).

        DeepSeek's MoEGate returns:
        - [0]: top_k_indices [batch*seq, k] - which experts were selected
        - [1]: top_k_weights [batch*seq, k] - their routing weights
        - [2]: aux_loss (usually None)

        Returns:
            (top_k_indices, top_k_weights) or (None, None) if not available
        """
        if isinstance(output, tuple) and len(output) >= 2:
            top_k_indices = output[0]  # [batch*seq, k]
            top_k_weights = output[1]  # [batch*seq, k]
            return top_k_indices, top_k_weights
        return None, None

    def create_router_hook(self, layer_idx: int, storage_dict: dict):
        """
        Create a hook for capturing router outputs.

        For DeepSeek, we hook the router directly since MLP doesn't
        return router_logits.

        Args:
            layer_idx: Layer index
            storage_dict: Dictionary to store captured outputs

        Returns:
            Hook function
        """
        model_card = self

        def hook(module, input, output):
            top_k_indices, top_k_weights = model_card.parse_router_output(output)
            if top_k_indices is not None:
                storage_dict[layer_idx] = {
                    'indices': top_k_indices.detach().cpu(),
                    'weights': top_k_weights.detach().cpu() if top_k_weights is not None else None
                }

        return hook

    def uses_router_hook_for_routing(self) -> bool:
        """
        Return True if this model requires hooking the router directly
        to capture routing decisions (vs getting them from MLP output).
        """
        return True

    def get_expert_steering_thresholds(self) -> dict:
        """
        Return thresholds for expert steering selection.

        DeepSeek uses more permissive thresholds due to different routing dynamics.
        """
        return {
            "kl_threshold": 2.0,
            "steering_score_threshold": -20.0,
            "prune_layer_percentage": 0.0
        }
