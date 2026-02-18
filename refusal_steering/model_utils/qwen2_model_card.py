"""
Model Card for Qwen2-57B-A14B-Instruct MoE.

Special features:
- 64 routed experts + 1 shared expert per MoE layer
- Top-4 routing (4 of 64 experts per token)
- 28 layers total (all MoE when mlp_only_layers is empty)
- Router is .gate (Linear, no bias) - uses forward hooks to inject bias
- MLP output is (hidden_states, router_logits) tuple for MoE layers
- Shared expert always active (not included in expert count/routing)

Router bias strategy:
  Uses hook-based bias injection (like Mixtral) rather than register_parameter
  because BnB 4-bit quantization + device_map="auto" can cause issues with
  adding parameters to already-dispatched modules.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple
from model_utils.model_card import ModelCard


class Qwen2ModelCard(ModelCard):
    """Model card for Qwen2-57B-A14B-Instruct."""

    def __init__(self, model, model_base):
        # Store bias values and hooks per layer (hook-based for quantized models)
        self._router_biases: Dict[int, torch.Tensor] = {}
        self._router_hooks: Dict[int, any] = {}

        # Read MoE config from model
        config = model.config
        self._num_experts = getattr(config, 'num_experts', 64)
        self._num_experts_per_tok = getattr(config, 'num_experts_per_tok', 4)
        self._mlp_only_layers = set(getattr(config, 'mlp_only_layers', None) or [])

        super().__init__(model, model_base)

    def _navigate_to_layers(self):
        """Navigate to model layers (handles potential wrapping)."""
        current = self.model
        while hasattr(current, 'model') and not hasattr(current, 'layers'):
            current = current.model
        if hasattr(current, 'layers'):
            return current.layers
        raise AttributeError(f"Could not find layers in Qwen2 model: {type(self.model)}")

    def get_expert_routing_mode(self) -> str:
        """Qwen2 MoE uses top-4 routing."""
        return f"top-{self._num_experts_per_tok}"

    def get_num_layers(self) -> int:
        """Return number of layers."""
        return len(self.layers)

    def is_moe_layer(self, layer_idx: int) -> bool:
        """
        Check if layer is MoE (vs dense MLP).

        Uses config.mlp_only_layers first, then verifies by checking
        for gate module (robust across transformers versions).
        """
        if self._mlp_only_layers and layer_idx in self._mlp_only_layers:
            return False
        # Verify the layer actually has a router/gate
        return hasattr(self.layers[layer_idx].mlp, 'gate')

    def get_num_experts(self, layer_idx: int) -> int:
        """Return number of routed experts (not including shared expert)."""
        if not self.is_moe_layer(layer_idx):
            return 0
        return self._num_experts

    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        """Get MLP (Qwen2MoeSparseMoeBlock or Qwen2MoeMLP) for a layer."""
        return self.layers[layer_idx].mlp

    def get_router(self, layer_idx: int) -> nn.Module:
        """Get router (gate) for an MoE layer."""
        if not self.is_moe_layer(layer_idx):
            raise ValueError(f"Layer {layer_idx} is not an MoE layer (dense MLP only)")
        return self.layers[layer_idx].mlp.gate

    def get_router_bias(self, router: nn.Module, layer_idx: Optional[int] = None) -> torch.Tensor:
        """
        Get router bias value.

        Qwen2 MoE routers use nn.Linear(..., bias=False) by default.
        We store bias values separately and inject them via forward hooks
        to avoid breaking accelerate/BnB dispatch.
        """
        if layer_idx is None:
            for idx in range(self.get_num_layers()):
                if self.is_moe_layer(idx) and self.get_router(idx) is router:
                    layer_idx = idx
                    break

        if layer_idx is None:
            raise ValueError("Could not determine layer_idx for router")

        if layer_idx not in self._router_biases:
            bias = torch.zeros(self._num_experts, dtype=torch.float32, device='cpu')
            self._router_biases[layer_idx] = bias
            self._register_bias_hook(layer_idx, router)
            print(f"Note: Qwen2 router has no bias, using forward hook for expert forcing (layer {layer_idx})")

        return self._router_biases[layer_idx]

    def _register_bias_hook(self, layer_idx: int, router: nn.Module):
        """Register a forward hook to inject bias into router output."""
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
            for idx in range(self.get_num_layers()):
                if self.is_moe_layer(idx) and self.get_router(idx) is router:
                    layer_idx = idx
                    break

        if layer_idx is None:
            raise ValueError("Could not determine layer_idx for router")

        if layer_idx not in self._router_hooks:
            self._register_bias_hook(layer_idx, router)

        self._router_biases[layer_idx] = bias.cpu()

    def remove_bias_hooks(self):
        """Remove all bias injection hooks (cleanup)."""
        for layer_idx, handle in self._router_hooks.items():
            handle.remove()
        self._router_hooks.clear()
        self._router_biases.clear()

    def parse_mlp_output(self, output) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parse MLP output into (hidden_states, router_logits).

        Qwen2 MoE layers return tuple: (hidden_states, router_logits)
        Dense MLP layers return just a hidden_states tensor.
        """
        if isinstance(output, tuple) and len(output) >= 2:
            return output[0], output[1]
        return output, None

    def wrap_mlp_output(self, hidden_states: torch.Tensor,
                        router_logits: Optional[torch.Tensor] = None):
        """Reconstruct MLP output in expected format."""
        if router_logits is not None:
            return (hidden_states, router_logits)
        return hidden_states

    def uses_router_hook_for_routing(self) -> bool:
        """Qwen2 MoE returns router_logits in MLP output, so use MLP hooks."""
        return False

    def get_router_output_format(self) -> str:
        """Qwen2 returns logits from MLP output."""
        return "logits"

    def get_expert_diffs_filename(self) -> str:
        """Return filename for Qwen2 expert diffs."""
        return "qwen2_expert_diffs.json"

    def get_expert_steering_thresholds(self) -> dict:
        """
        Return thresholds for expert steering selection.

        Uses permissive thresholds consistent with other MoE models.
        """
        return {
            "kl_threshold": 2.0,
            "steering_score_threshold": -20.0,
            "prune_layer_percentage": 0.0
        }
