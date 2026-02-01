"""
Model Card Abstract Base Class.

Encapsulates MoE-specific logic for expert intervention, separating concerns:
- ModelBase: handles tokenization, generation, base API
- ModelCard: handles MoE internals (routing, experts, output format)

Design principles:
- One card per model variant (OSS and UnslothOSS are separate)
- MLP output format detected ONCE at init (not per forward pass)
- Expert diff generation abstracted (shared utility)
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn


class ModelCard(ABC):
    """
    Abstract base class for model cards that encapsulate MoE architecture specifics.

    Each model variant should have its own model card file (like model bases).
    """

    def __init__(self, model, model_base):
        """
        Initialize model card.

        Args:
            model: Underlying model (unwrapped if needed)
            model_base: ModelBase instance (for tokenizer, etc.)
        """
        self.model = model
        self.model_base = model_base
        self.layers = self._navigate_to_layers()

        # Detect MLP output format once at initialization
        self._mlp_output_format = self._detect_mlp_output_format()

    @abstractmethod
    def _navigate_to_layers(self):
        """Navigate to model.layers (handles PEFT wrapping)."""
        pass

    def _detect_mlp_output_format(self) -> str:
        """
        Detect if MLP returns tuple or tensor.
        Runs once at init to avoid runtime checks.

        Returns:
            'tuple' or 'tensor'
        """
        try:
            import warnings
            warnings.filterwarnings('ignore')

            # Find first MoE layer
            layer_idx = 0
            while not self.is_moe_layer(layer_idx) and layer_idx < self.get_num_layers():
                layer_idx += 1

            if layer_idx >= self.get_num_layers():
                return 'tensor'

            mlp = self.get_mlp_module(layer_idx)
            hidden_size = self.model.config.hidden_size

            # Test forward pass
            test_input = torch.randn(
                1, 1, hidden_size,
                device=next(mlp.parameters()).device,
                dtype=next(mlp.parameters()).dtype
            )

            with torch.no_grad():
                output = mlp(test_input)

            return 'tuple' if isinstance(output, tuple) and len(output) >= 2 else 'tensor'

        except Exception as e:
            print(f"Warning: MLP format detection failed: {e}")
            return 'tensor'  # Safe default

    # === Abstract Methods (Must Implement) ===

    @abstractmethod
    def get_num_experts(self, layer_idx: int) -> int:
        """Number of routed experts (not including shared experts)."""
        pass

    @abstractmethod
    def get_num_layers(self) -> int:
        """Total transformer layers."""
        pass

    @abstractmethod
    def get_mlp_module(self, layer_idx: int) -> nn.Module:
        """Get MLP/MoE module."""
        pass

    @abstractmethod
    def get_router(self, layer_idx: int) -> nn.Module:
        """Get router module."""
        pass

    @abstractmethod
    def get_router_bias(self, router: nn.Module, layer_idx: Optional[int] = None) -> torch.Tensor:
        """Get router bias (may create if doesn't exist). layer_idx needed for hook-based bias (Mixtral)."""
        pass

    @abstractmethod
    def set_router_bias(self, router: nn.Module, bias: torch.Tensor, layer_idx: Optional[int] = None):
        """Set router bias. layer_idx needed for hook-based bias (Mixtral)."""
        pass

    @abstractmethod
    def is_moe_layer(self, layer_idx: int) -> bool:
        """Check if layer is MoE (some models have dense layers)."""
        pass

    @abstractmethod
    def get_expert_routing_mode(self) -> str:
        """Return routing strategy: 'top-2', 'top-6', etc."""
        pass

    @abstractmethod
    def get_expert_diffs_filename(self) -> str:
        """Return filename for expert diffs: 'oss_expert_diffs.json', etc."""
        pass

    # === Concrete Methods (Use Pre-Detected Format) ===

    def parse_mlp_output(self, output) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Parse MLP output into (hidden_states, router_logits).
        Uses pre-detected format (no runtime checks).
        """
        if self._mlp_output_format == 'tuple':
            if isinstance(output, tuple) and len(output) >= 2:
                return output[0], output[1]
            elif isinstance(output, tuple):
                return output[0], None
            else:
                return output, None
        else:
            return output, None

    def wrap_mlp_output(self, hidden_states: torch.Tensor,
                        router_logits: Optional[torch.Tensor] = None):
        """
        Reconstruct MLP output in expected format.
        Uses pre-detected format.
        """
        if self._mlp_output_format == 'tuple' and router_logits is not None:
            return (hidden_states, router_logits)
        return hidden_states

    def generate_expert_diffs(
        self,
        harmful_dataset_path: str = "dataset/splits/harmful_train.json",
        harmless_dataset_path: str = "dataset/splits/harmless_train.json",
        output_path: Optional[str] = None,
        batch_size: int = 4,
        last_n_tokens: int = 5,
        num_harmful: Optional[int] = None,
        num_harmless: int = 200
    ) -> Dict:
        """
        Generate expert activation frequency differences.

        Uses shared utility (expert_diff_generator.py) that works for all models.
        Can be overridden if model needs special handling.

        Args:
            harmful_dataset_path: Path to harmful prompts JSON
            harmless_dataset_path: Path to harmless prompts JSON
            output_path: Where to save results (default: expert_explore/{filename})
            batch_size: Batch size for processing
            last_n_tokens: Number of tokens from end to analyze
            num_harmful: Number of harmful samples (None = all)
            num_harmless: Number of harmless samples

        Returns:
            Dictionary with expert activation frequency differences
        """
        from submodules.expert_diff_generator import generate_expert_diffs_for_model

        if output_path is None:
            output_path = f"expert_explore/{self.get_expert_diffs_filename()}"

        return generate_expert_diffs_for_model(
            model_base=self.model_base,
            model_card=self,
            harmful_dataset_path=harmful_dataset_path,
            harmless_dataset_path=harmless_dataset_path,
            output_path=output_path,
            batch_size=batch_size,
            last_n_tokens=last_n_tokens,
            num_harmful=num_harmful,
            num_harmless=num_harmless
        )

    # === Router Output Handling (for models with different routing output formats) ===

    def uses_router_hook_for_routing(self) -> bool:
        """
        Return True if this model requires hooking the router directly
        to capture routing decisions (vs getting them from MLP output).

        Default: False (most models return router_logits in MLP output)
        Override in subclasses for models like DeepSeek that don't.
        """
        return False

    def get_router_output_format(self) -> str:
        """
        Return how router outputs routing information.

        Options:
        - 'logits': Raw logits for all experts [batch*seq, num_experts]
        - 'top_k_indices': Top-k expert indices and weights

        Default: 'logits'
        """
        return "logits"

    def parse_router_output(self, output):
        """
        Parse router output. Default assumes raw logits.
        Override for models with different formats.

        Returns:
            For 'logits': router_logits tensor
            For 'top_k_indices': (indices, weights) tuple
        """
        if isinstance(output, torch.Tensor):
            return output
        elif isinstance(output, tuple) and len(output) > 0:
            return output[0]
        return None

    def create_router_hook(self, layer_idx: int, storage_dict: dict):
        """
        Create a hook for capturing router outputs.
        Default implementation for models returning logits.

        Args:
            layer_idx: Layer index
            storage_dict: Dictionary to store captured outputs

        Returns:
            Hook function
        """
        def hook(module, input, output):
            router_logits = self.parse_router_output(output)
            if router_logits is not None:
                storage_dict[layer_idx] = router_logits.detach().cpu()

        return hook
