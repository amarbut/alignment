"""
Model Card for Mixtral 8x7B.

Key differences from OSS:
- MLP module: block_sparse_moe (not mlp)
- Router: gate (not router)
- Router has NO BIAS by default - we use forward hooks to inject bias
  (Adding parameters breaks accelerate offloading)
"""

import torch
import torch.nn as nn
from typing import Optional, Dict
from pipeline.model_utils.model_card import ModelCard


class MixtralModelCard(ModelCard):
    """Model card for Mixtral 8x7B."""

    def __init__(self, model, model_base):
        # Store bias values and hooks per layer
        self._router_biases: Dict[int, torch.Tensor] = {}
        self._router_hooks: Dict[int, any] = {}
        super().__init__(model, model_base)

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

    def get_router_bias(self, router: nn.Module, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get router bias value.

        CRITICAL: Mixtral routers don't have bias by default.
        We store bias values separately and inject them via forward hooks.
        This avoids breaking accelerate's offloading system.
        """
        if layer_idx is None:
            # Try to find layer_idx from router
            for idx in range(self.get_num_layers()):
                if self.get_router(idx) is router:
                    layer_idx = idx
                    break

        if layer_idx is None:
            raise ValueError("Could not determine layer_idx for router")

        # Initialize bias if not exists
        if layer_idx not in self._router_biases:
            num_experts = 8
            # Store bias on CPU to avoid meta tensor issues with accelerate offloading
            bias = torch.zeros(num_experts, dtype=router.weight.dtype, device='cpu')
            self._router_biases[layer_idx] = bias
            self._register_bias_hook(layer_idx, router)
            print(f"Note: Mixtral router has no bias, using forward hook for expert forcing (layer {layer_idx})")

        return self._router_biases[layer_idx]

    def _register_bias_hook(self, layer_idx: int, router: nn.Module):
        """Register a forward hook to inject bias into router output."""
        # Store reference to self for closure
        model_card = self

        def bias_hook(module, input, output):
            bias = model_card._router_biases.get(layer_idx)
            if bias is None:
                return output
            # Bias is stored on CPU - check if non-zero there (safe operation)
            if bias.device.type == 'cpu' and not (bias != 0).any():
                return output
            # Move bias to output device and apply
            return output + bias.to(device=output.device, dtype=output.dtype)

        # Remove existing hook if any
        if layer_idx in self._router_hooks:
            self._router_hooks[layer_idx].remove()

        handle = router.register_forward_hook(bias_hook)
        self._router_hooks[layer_idx] = handle

    def set_router_bias(self, router: nn.Module, bias: torch.Tensor, layer_idx: Optional[int] = None):
        """Set router bias value (stored separately, injected via hook)."""
        if layer_idx is None:
            # Try to find layer_idx from router
            for idx in range(self.get_num_layers()):
                if self.get_router(idx) is router:
                    layer_idx = idx
                    break

        if layer_idx is None:
            raise ValueError("Could not determine layer_idx for router")

        # Ensure hook is registered
        if layer_idx not in self._router_hooks:
            self._register_bias_hook(layer_idx, router)

        # Store bias on CPU to avoid meta tensor issues with accelerate offloading
        self._router_biases[layer_idx] = bias.cpu()

    def remove_bias_hooks(self):
        """Remove all bias injection hooks (cleanup)."""
        for layer_idx, handle in self._router_hooks.items():
            handle.remove()
        self._router_hooks.clear()
        self._router_biases.clear()

    def is_moe_layer(self, layer_idx: int) -> bool:
        """All Mixtral layers are MoE."""
        return True

    def get_expert_routing_mode(self) -> str:
        """Mixtral uses top-2 routing."""
        return "top-2"

    def get_expert_diffs_filename(self) -> str:
        """Return filename for Mixtral expert diffs."""
        return "mixtral_expert_diffs.json"

    def get_expert_steering_thresholds(self) -> dict:
        """
        Return thresholds for expert steering selection.

        Mixtral uses more permissive thresholds due to different routing dynamics.
        """
        return {
            "kl_threshold": 2.0,
            "steering_score_threshold": -20.0,
            "prune_layer_percentage": 0.0
        }
