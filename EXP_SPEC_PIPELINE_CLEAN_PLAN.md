# Implementation Plan: Expert-Specific Intervention for Mixtral and DeepSeek

## Executive Summary

Extend the expert-specific intervention pipeline to support:
1. **Mixtral 8x7B-Instruct** - 8 experts per layer, top-2 routing
2. **DeepSeek-MoE-16B-Chat** - 64 routed + 2 shared experts, top-6 routing, standard gating

**Key Requirements (Addressing User Feedback):**
1. ✅ Use DeepSeek-V2-Lite instead of Llama 4 (better hardware fit)
2. ✅ Use `dataset/splits` for expert diffs (based on `extract_expert_routing_unsloth.py`)
3. ✅ Support both `openai/gpt-oss-20b` AND `unsloth/gpt-oss-20b-unsloth-bnb-4bit` (separate model cards)
4. ✅ Separate file per model card (like model bases: `oss_model_card.py`, `unsloth_oss_model_card.py`, etc.)
5. ✅ Abstract expert diff generation into model cards (single `expert_diff_generator.py` utility)
6. ✅ Address Mixtral's lack of router bias carefully (test multiple approaches)
7. ✅ Check MLP output format at initialization, not runtime

**Strategy:** Implement a **Model Card Architecture** to centralize model-specific MoE logic.

**Estimated Work:** 3-4 days

---

## Current State

### Working Pipeline (GPT-OSS-20B)
- `run_pipeline_expert_specific.py` - Main orchestrator
- `expert_selection.py` - Loads pre-computed expert diffs
- `expert_specific_activations.py` - Forces experts, extracts MLP outputs
- `expert_intervention.py` - Weighted intervention hooks
- `expert_selection_mlp.py` - Direction selection

### Problems

**Hardcoded Logic Scattered Across Files:**
- `get_model_layers()` duplicated 3+ times
- Router access: `layers[i].mlp.router` breaks for Mixtral (`block_sparse_moe`)
- Bias access: Multiple fallback patterns
- Output parsing: Conditional checks everywhere

**Result:** Adding new models requires modifying 5+ files with complex fallbacks.

---

## Target Models

### OSS (Two Variants - Separate Model Cards)

**openai/gpt-oss-20b:**
- Loader: `AutoModelForCausalLM`
- Model Class: `OSSModel`
- Model Card: **`OSSModelCard`** (new)
- Layers access: Direct `model.model.layers`

**unsloth/gpt-oss-20b-unsloth-bnb-4bit:**
- Loader: `FastLanguageModel` with 4-bit quantization
- Model Class: `UnslothOSSModel`
- Model Card: **`UnslothOSSModelCard`** (new, separate from OSS)
- Layers access: Navigate PEFT wrapping to `base_model.layers`
- Router access: `router.linear.bias` (wrapped)

**Shared Specs:**
- 22 layers, 32 experts per layer, top-2 routing
- MLP module: `layers[i].mlp`
- MLP output: Tuple `(hidden_states, router_logits)`

**Key Insight:** These need SEPARATE model cards because layer/router access differs significantly.

### Mixtral 8x7B-Instruct-v0.1
- Model Class: `MixtralModel` (exists)
- Model Card: **`MixtralModelCard`** (new)
- 32 layers, 8 experts per layer, top-2 routing
- MLP module: `layers[i].block_sparse_moe` ⚠️
- Router: `block_sparse_moe.gate` ⚠️
- **⚠️ CRITICAL:** Router has **NO BIAS by default**

**Mixtral Bias Solution Strategy:**
We'll implement and test three approaches:
1. **Dynamic bias addition** - Add `nn.Parameter` when forcing (simplest)
2. **Weight temperature scaling** - Temporarily scale router weights (preserves architecture)
3. **Pre-hook input shifting** - Shift router input to favor expert (cleanest)

Start with #1, keep #2/#3 as fallbacks if #1 causes issues.

### DeepSeek-MoE-16B-Chat
- Model Class: **`DeepSeekModel`** (new)
- Model Card: **`DeepSeekModelCard`** (new)
- 28 layers, 2048 hidden size
- **64 routed + 2 shared experts** per MoE layer
- Top-6 routing (6 of 64 routed experts, standard top-K selection)
- **First layer is DENSE** (not MoE)
- Total: 16.4B parameters
- MLP module: Inspect at runtime (likely `layers[i].mlp`)
- Router: Standard gating network (linear projection + softmax)
- BF16, fits in 40GB A100
- **Routing compatible** with expert-specific intervention (standard top-K, unlike V2's advanced routing)

---

## Design: Model Card Architecture

### File Structure (Separate Files Per Card)

```
pipeline/model_utils/
├── model_card.py                    # Abstract base class
├── oss_model_card.py               # OSSModelCard (openai variant)
├── unsloth_oss_model_card.py      # UnslothOSSModelCard (unsloth variant)
├── mixtral_model_card.py          # MixtralModelCard
├── deepseek_model_card.py         # DeepSeekModelCard
├── model_card_factory.py          # Factory function
└── (existing model files)
```

**Rationale:** Mirrors the structure of model bases (separate file per model).

### Abstract Base: model_card.py

```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn

class ModelCard(ABC):
    """
    Encapsulates MoE-specific logic for expert intervention.

    Design principles:
    - ModelBase: handles tokenization, generation, base API
    - ModelCard: handles MoE internals (routing, experts, format)
    - One card per model variant (OSS and UnslothOSS are separate)
    - MLP output format detected ONCE at init (not per forward pass)
    - Expert diff generation abstracted (shared utility)
    """

    def __init__(self, model, model_base):
        """
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
    def get_router_bias(self, router: nn.Module) -> torch.Tensor:
        """Get router bias (may create if doesn't exist)."""
        pass

    @abstractmethod
    def set_router_bias(self, router: nn.Module, bias: torch.Tensor):
        """Set router bias."""
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
        """
        from expert_diff_generator import generate_expert_diffs_for_model

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
```

### Concrete Cards (Separate Files)

#### oss_model_card.py

```python
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
        raise AttributeError("Could not find layers")

    def get_num_experts(self, layer_idx: int) -> int:
        return 32

    def get_num_layers(self) -> int:
        return len(self.layers)

    def get_mlp_module(self, layer_idx: int):
        return self.layers[layer_idx].mlp

    def get_router(self, layer_idx: int):
        return self.layers[layer_idx].mlp.router

    def get_router_bias(self, router):
        if hasattr(router, 'bias') and router.bias is not None:
            return router.bias
        else:
            # Create bias if missing
            num_experts = 32
            bias = torch.zeros(num_experts, device=router.weight.device,
                             dtype=router.weight.dtype)
            router.bias = nn.Parameter(bias)
            return router.bias

    def set_router_bias(self, router, bias):
        if not hasattr(router, 'bias') or router.bias is None:
            router.bias = nn.Parameter(bias)
        else:
            router.bias.data = bias

    def is_moe_layer(self, layer_idx: int) -> bool:
        return True  # All OSS layers are MoE

    def get_expert_routing_mode(self) -> str:
        return "top-2"

    def get_expert_diffs_filename(self) -> str:
        return "oss_expert_diffs.json"
```

#### unsloth_oss_model_card.py

```python
import torch
import torch.nn as nn
from pipeline.model_utils.model_card import ModelCard

class UnslothOSSModelCard(ModelCard):
    """Model card for unsloth/gpt-oss-20b-unsloth-bnb-4bit."""

    def _navigate_to_layers(self):
        """Navigate PEFT wrapping for unsloth."""
        current = self.model

        # Navigate through PEFT/unsloth wrappers
        while hasattr(current, 'model') and not hasattr(current, 'layers'):
            current = current.model

        if hasattr(current, 'layers'):
            return current.layers

        raise AttributeError("Could not find layers in unsloth model")

    def get_num_experts(self, layer_idx: int) -> int:
        return 32

    def get_num_layers(self) -> int:
        return len(self.layers)

    def get_mlp_module(self, layer_idx: int):
        return self.layers[layer_idx].mlp

    def get_router(self, layer_idx: int):
        return self.layers[layer_idx].mlp.router

    def get_router_bias(self, router):
        """Unsloth wraps bias in router.linear.bias."""
        if hasattr(router, 'linear') and hasattr(router.linear, 'bias'):
            if router.linear.bias is not None:
                return router.linear.bias
            else:
                # Create bias
                num_experts = 32
                bias = torch.zeros(num_experts, device=router.linear.weight.device,
                                 dtype=router.linear.weight.dtype)
                router.linear.bias = nn.Parameter(bias)
                return router.linear.bias
        elif hasattr(router, 'bias') and router.bias is not None:
            return router.bias
        else:
            # Fallback: create at router level
            num_experts = 32
            bias = torch.zeros(num_experts, device=router.weight.device,
                             dtype=router.weight.dtype)
            router.bias = nn.Parameter(bias)
            return router.bias

    def set_router_bias(self, router, bias):
        """Set bias in wrapped router."""
        if hasattr(router, 'linear') and hasattr(router.linear, 'bias'):
            router.linear.bias.data = bias
        elif hasattr(router, 'bias'):
            router.bias.data = bias
        else:
            router.bias = nn.Parameter(bias)

    def is_moe_layer(self, layer_idx: int) -> bool:
        return True

    def get_expert_routing_mode(self) -> str:
        return "top-2"

    def get_expert_diffs_filename(self) -> str:
        return "unsloth_oss_expert_diffs.json"
```

#### mixtral_model_card.py

```python
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
        raise AttributeError("Could not find layers")

    def get_num_experts(self, layer_idx: int) -> int:
        return 8

    def get_num_layers(self) -> int:
        return len(self.layers)

    def get_mlp_module(self, layer_idx: int):
        # KEY DIFFERENCE: block_sparse_moe
        return self.layers[layer_idx].block_sparse_moe

    def get_router(self, layer_idx: int):
        # KEY DIFFERENCE: gate (not router)
        return self.layers[layer_idx].block_sparse_moe.gate

    def get_router_bias(self, router):
        """
        ⚠️ CRITICAL: Mixtral routers don't have bias by default.

        Strategy 1 (current): Dynamically add bias parameter.
        If this causes issues, we'll implement:
        - Strategy 2: Weight temperature scaling
        - Strategy 3: Pre-hook input shifting
        """
        if hasattr(router, 'bias') and router.bias is not None:
            return router.bias
        else:
            print("Warning: Mixtral router has no bias, adding parameter")
            num_experts = 8
            bias = torch.zeros(num_experts, device=router.weight.device,
                             dtype=router.weight.dtype)
            router.bias = nn.Parameter(bias)
            return router.bias

    def set_router_bias(self, router, bias):
        if not hasattr(router, 'bias') or router.bias is None:
            router.bias = nn.Parameter(bias)
        else:
            router.bias.data = bias

    def is_moe_layer(self, layer_idx: int) -> bool:
        return True

    def get_expert_routing_mode(self) -> str:
        return "top-2"

    def get_expert_diffs_filename(self) -> str:
        return "mixtral_expert_diffs.json"
```

#### deepseek_model_card.py

```python
import torch
import torch.nn as nn
from pipeline.model_utils.model_card import ModelCard

class DeepSeekModelCard(ModelCard):
    """
    Model card for DeepSeek-V2-Lite (16B).

    Special features:
    - First layer is dense (not MoE)
    - 64 routed + 2 shared experts per MoE layer
    - Top-6 routing
    """

    def __init__(self, model, model_base):
        super().__init__(model, model_base)

        # DeepSeek-specific config
        self.num_routed_experts = 64
        self.num_shared_experts = 2
        self.num_experts_per_tok = 6
        self.first_k_dense_replace = 1  # First layer is dense

    def _navigate_to_layers(self):
        """Standard navigation."""
        current = self.model
        while hasattr(current, 'model') and not hasattr(current, 'layers'):
            current = current.model
        if hasattr(current, 'layers'):
            return current.layers
        raise AttributeError("Could not find layers")

    def get_num_experts(self, layer_idx: int) -> int:
        """Return routed experts only (not shared)."""
        if not self.is_moe_layer(layer_idx):
            return 0
        return self.num_routed_experts  # 64

    def get_num_layers(self) -> int:
        return len(self.layers)  # 28

    def get_mlp_module(self, layer_idx: int):
        """DeepSeek MLP module name TBD (inspect at runtime)."""
        layer = self.layers[layer_idx]

        # Try common names
        if hasattr(layer, 'mlp'):
            return layer.mlp
        elif hasattr(layer, 'moe'):
            return layer.moe
        elif hasattr(layer, 'ffn'):
            return layer.ffn
        else:
            raise AttributeError(f"Could not find MLP in layer {layer_idx}")

    def get_router(self, layer_idx: int):
        """DeepSeek router name TBD (inspect at runtime)."""
        if not self.is_moe_layer(layer_idx):
            raise ValueError(f"Layer {layer_idx} is dense, no router")

        mlp = self.get_mlp_module(layer_idx)

        # Try common names
        if hasattr(mlp, 'gate'):
            return mlp.gate
        elif hasattr(mlp, 'router'):
            return mlp.router
        elif hasattr(mlp, 'expert_gate'):
            return mlp.expert_gate
        else:
            raise AttributeError(f"Could not find router in layer {layer_idx}")

    def get_router_bias(self, router):
        """DeepSeek may not have bias (like Mixtral)."""
        if hasattr(router, 'bias') and router.bias is not None:
            return router.bias
        else:
            # Create bias
            bias = torch.zeros(self.num_routed_experts,
                             device=router.weight.device,
                             dtype=router.weight.dtype)
            router.bias = nn.Parameter(bias)
            return router.bias

    def set_router_bias(self, router, bias):
        if not hasattr(router, 'bias') or router.bias is None:
            router.bias = nn.Parameter(bias)
        else:
            router.bias.data = bias

    def is_moe_layer(self, layer_idx: int) -> bool:
        """First layer is dense."""
        return layer_idx >= self.first_k_dense_replace

    def get_expert_routing_mode(self) -> str:
        return "top-6"

    def get_expert_diffs_filename(self) -> str:
        return "deepseek_expert_diffs.json"
```

### Factory: model_card_factory.py

```python
from pipeline.model_utils.model_card import ModelCard

def create_model_card(model_base) -> ModelCard:
    """
    Create appropriate model card for a model.

    Detection order:
    1. Check ModelBase class name (most reliable)
    2. Check model_name_or_path
    3. Check model.config.architectures
    """
    model_type = model_base.__class__.__name__
    model_path = model_base.model_name_or_path.lower()

    # Get architecture from config
    if hasattr(model_base.model, 'config'):
        arch = getattr(model_base.model.config, 'architectures', [])
        arch_str = ''.join(arch).lower() if arch else ''
    else:
        arch_str = ''

    # === OSS Detection (Two Variants) ===

    # Unsloth OSS (check first - more specific)
    if 'UnslothOSSModel' in model_type or 'unsloth' in model_path:
        from pipeline.model_utils.unsloth_oss_model_card import UnslothOSSModelCard
        return UnslothOSSModelCard(model_base.model, model_base)

    # Standard OSS
    elif 'OSSModel' in model_type or 'oss' in model_path or 'oss' in arch_str:
        from pipeline.model_utils.oss_model_card import OSSModelCard
        return OSSModelCard(model_base.model, model_base)

    # === Mixtral Detection ===
    elif 'MixtralModel' in model_type or 'mixtral' in model_path or 'mixtral' in arch_str:
        from pipeline.model_utils.mixtral_model_card import MixtralModelCard
        return MixtralModelCard(model_base.model, model_base)

    # === DeepSeek Detection ===
    elif 'DeepSeekModel' in model_type or 'deepseek' in model_path or 'deepseek' in arch_str:
        from pipeline.model_utils.deepseek_model_card import DeepSeekModelCard
        return DeepSeekModelCard(model_base.model, model_base)

    else:
        raise ValueError(
            f"No model card for: {model_type}, {model_path}, {arch_str}\n"
            f"To add support, create a new model card file."
        )
```

---

## Implementation Sequence

### Phase 1: Model Card Infrastructure (Day 1)

**Goal:** Create architecture and validate with existing OSS models.

#### Step 1.1: Create Base Files
- `pipeline/model_utils/model_card.py` - Abstract base class
- `pipeline/model_utils/oss_model_card.py` - For openai/gpt-oss-20b
- `pipeline/model_utils/unsloth_oss_model_card.py` - For unsloth variant
- `pipeline/model_utils/model_card_factory.py` - Factory function

#### Step 1.2: Create Expert Diff Generator Utility
**File:** `expert_diff_generator.py` (root directory)

Extract logic from `extract_expert_routing_unsloth.py` into reusable function:

```python
def generate_expert_diffs_for_model(
    model_base,
    model_card,
    harmful_dataset_path,
    harmless_dataset_path,
    output_path,
    batch_size=4,
    last_n_tokens=5,
    num_harmful=None,
    num_harmless=200
):
    """
    Generate expert activation frequency differences for any MoE model.

    Works with any model card. Algorithm:
    1. Load datasets from dataset/splits
    2. For each prompt, extract router logits from last N tokens
    3. Compute top expert for each token position
    4. Calculate activation frequency: harmful vs harmless
    5. Save differences per layer-expert pair
    """
    # Implementation based on extract_expert_routing_unsloth.py
    # ...
```

#### Step 1.3: Refactor expert_specific_activations.py

Replace hardcoded logic with model card calls:

```python
from pipeline.model_utils.model_card_factory import create_model_card

def force_expert_via_bias(
    model_base,
    layer_idx,
    expert_id,
    force_strength=100.0,
    model_card=None
):
    """Force expert using model card."""
    if model_card is None:
        model_card = create_model_card(model_base)

    if not model_card.is_moe_layer(layer_idx):
        raise ValueError(f"Layer {layer_idx} is not MoE")

    router = model_card.get_router(layer_idx)
    original_bias = model_card.get_router_bias(router).clone()

    # Create forced bias
    num_experts = model_card.get_num_experts(layer_idx)
    modified_bias = torch.full((num_experts,), -force_strength,
                              device=original_bias.device,
                              dtype=original_bias.dtype)
    modified_bias[expert_id] = force_strength

    model_card.set_router_bias(router, modified_bias)

    return original_bias, model_card
```

Update all functions in the file to use model card.

#### Step 1.4: Refactor expert_intervention.py

```python
def get_expert_weighted_activation_addition_hook(
    direction,
    expert_id,
    coeff,
    model_card
):
    def hook_fn(module, input, output):
        # Use model card for parsing (no runtime checks)
        hidden_states, router_logits = model_card.parse_mlp_output(output)

        if router_logits is None:
            return output  # Not MoE

        # Apply weighted intervention
        router_probs = torch.nn.functional.softmax(router_logits, dim=-1)
        expert_weight = router_probs[:, expert_id]

        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        expert_weight = expert_weight.view(batch_size, seq_len, 1)
        weighted_direction = expert_weight * direction.to(hidden_states.device, hidden_states.dtype)
        modified_output = hidden_states + coeff * weighted_direction

        # Use model card for wrapping (no runtime checks)
        return model_card.wrap_mlp_output(modified_output, router_logits)

    return hook_fn
```

#### Step 1.5: Validation

Run OSS pipeline with both model variants:

```bash
# Test openai/gpt-oss-20b
python test_model_card_oss.py --model_path openai/gpt-oss-20b

# Test unsloth/gpt-oss-20b-unsloth-bnb-4bit
python test_model_card_oss.py --model_path unsloth/gpt-oss-20b-unsloth-bnb-4bit
```

**Success criteria:**
- Both variants work with their respective model cards
- Expert forcing successful
- Intervention hooks apply correctly
- Results identical to baseline

---

### Phase 2: Mixtral Support (Day 2)

#### Step 2.1: Create Model Card
**File:** `pipeline/model_utils/mixtral_model_card.py` (implemented above)

#### Step 2.2: Test Mixtral Bias Handling

Create `test_mixtral_bias.py`:

```python
# Test Strategy 1: Dynamic bias addition
model_base = construct_model_base("mistralai/Mixtral-8x7B-Instruct-v0.1")
card = create_model_card(model_base)

# Test forcing expert
original_bias, card = force_expert_via_bias(model_base, 0, 0)
# Generate with forced expert
# Restore
restore_router_bias(model_base, 0, original_bias, card)

# Verify no side effects
```

If Strategy 1 causes issues (NaN, unexpected routing), implement alternatives:

**Strategy 2: Weight Temperature Scaling**
```python
def force_expert_via_weight_scaling(model_base, layer_idx, expert_id, strength=10.0):
    """Scale router weights instead of bias."""
    router = card.get_router(layer_idx)
    original_weights = router.weight.data.clone()

    # Scale target expert's weights
    router.weight.data[expert_id] *= strength
    # Scale others down
    for i in range(8):
        if i != expert_id:
            router.weight.data[i] /= strength

    return original_weights
```

**Strategy 3: Pre-Hook Input Shifting**
```python
def get_router_input_shift_hook(expert_id, strength):
    """Shift router input to favor expert."""
    def hook_fn(module, input):
        shifted = input[0].clone()
        # Add offset that increases logit for expert_id
        # (requires analyzing router weight structure)
        return (shifted,)
    return hook_fn
```

#### Step 2.3: Generate Mixtral Expert Diffs

```bash
python -c "
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.model_utils.model_card_factory import create_model_card

model_base = construct_model_base('mistralai/Mixtral-8x7B-Instruct-v0.1')
card = create_model_card(model_base)

# Generate diffs (uses shared utility)
card.generate_expert_diffs(
    harmful_dataset_path='dataset/splits/harmful_train.json',
    harmless_dataset_path='dataset/splits/harmless_train.json',
    num_harmful=None,  # All
    num_harmless=200
)
"
```

Outputs: `expert_explore/mixtral_expert_diffs.json`

#### Step 2.4: Run Mixtral Pipeline

```bash
python run_pipeline_expert_specific.py \
  --model_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --threshold 15.0 \
  --n_train 100 \
  --n_val 50 \
  --n_test 50
```

**Monitor:**
- Expert forcing works (no NaN or crashes)
- Direction extraction succeeds
- Interventions apply correctly

---

### Phase 3: DeepSeek Support (Day 3-4)

#### Step 3.1: Inspect DeepSeek Architecture

**Script:** `inspect_deepseek_architecture.py`

```python
from transformers import AutoConfig, AutoModelForCausalLM
import torch

config = AutoConfig.from_pretrained("deepseek-ai/deepseek-moe-16b-chat")

print(f"Num layers: {config.num_hidden_layers}")
print(f"Hidden size: {config.hidden_size}")
print(f"Num experts: {getattr(config, 'num_experts', 'N/A')}")
print(f"Routed experts: {getattr(config, 'n_routed_experts', 'N/A')}")
print(f"Shared experts: {getattr(config, 'n_shared_experts', 'N/A')}")
print(f"Experts per token: {getattr(config, 'num_experts_per_tok', 'N/A')}")
print(f"First dense: {getattr(config, 'first_k_dense_replace', 'N/A')}")

# Load model (BF16)
model = AutoModelForCausalLM.from_pretrained(
    "deepseek-ai/deepseek-moe-16b-chat",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

# Inspect layer 0 (dense)
layer0 = model.model.layers[0]
print(f"\nLayer 0 (dense): {dir(layer0)}")
if hasattr(layer0, 'mlp'):
    print(f"  Has .mlp: {type(layer0.mlp)}")

# Inspect layer 1 (MoE)
layer1 = model.model.layers[1]
print(f"\nLayer 1 (MoE): {dir(layer1)}")
if hasattr(layer1, 'mlp'):
    print(f"  MLP type: {type(layer1.mlp)}")
    print(f"  MLP contents: {dir(layer1.mlp)}")
    if hasattr(layer1.mlp, 'gate'):
        print(f"  Has .gate (router)")
    if hasattr(layer1.mlp, 'router'):
        print(f"  Has .router")

# Test MLP forward
test_input = torch.randn(1, 1, config.hidden_size,
                        device='cuda', dtype=torch.bfloat16)
with torch.no_grad():
    output = layer1.mlp(test_input)
    print(f"\nMLP output type: {type(output)}")
    if isinstance(output, tuple):
        print(f"  Tuple length: {len(output)}")
        print(f"  Element 0 shape: {output[0].shape}")
        if len(output) > 1:
            print(f"  Element 1 shape: {output[1].shape}")
```

**Output:** Update `DeepSeekModelCard` based on findings.

#### Step 3.2: Create DeepSeek Model Class

**File:** `pipeline/model_utils/deepseek_model.py`

```python
import torch
import functools
from transformers import AutoTokenizer, AutoModelForCausalLM
from pipeline.model_utils.model_base import ModelBase

DEEPSEEK_REFUSAL_TOKS = [40, 357, 2305]  # 'I', 'Sorry', 'As' (TBD)

def format_instruction_deepseek(instruction: str, output: str=None):
    formatted = f"User: {instruction}\n\nAssistant:"
    if output:
        formatted += f" {output}"
    return formatted

def tokenize_instructions_deepseek(tokenizer, instructions, outputs=None):
    if outputs:
        prompts = [format_instruction_deepseek(i, o) for i, o in zip(instructions, outputs)]
    else:
        prompts = [format_instruction_deepseek(i) for i in instructions]

    return tokenizer(prompts, padding=True, truncation=False, return_tensors="pt")

class DeepSeekModel(ModelBase):
    def _load_model(self, model_path, dtype=torch.bfloat16):
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        ).eval()
        model.config.use_cache = False
        model.requires_grad_(False)
        torch.set_grad_enabled(False)
        return model

    def _load_tokenizer(self, model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _get_tokenize_instructions_fn(self):
        return functools.partial(tokenize_instructions_deepseek, tokenizer=self.tokenizer)

    def _get_eoi_toks(self):
        return self.tokenizer.encode("\n\n", add_special_tokens=False)

    def _get_refusal_toks(self):
        return DEEPSEEK_REFUSAL_TOKS

    def _get_model_block_modules(self):
        return self.model.model.layers

    def _get_attn_modules(self):
        return torch.nn.ModuleList([block.self_attn for block in self.model_block_modules])

    def _get_mlp_modules(self):
        return torch.nn.ModuleList([block.mlp for block in self.model_block_modules])
```

**Add to model factory:**
```python
# In model_factory.py
elif 'deepseek' in model_path_lower:
    return DeepSeekModel(model_path)
```

#### Step 3.3: Create DeepSeek Model Card

**File:** `pipeline/model_utils/deepseek_model_card.py` (implemented above)

Update based on inspection results.

#### Step 3.4: Generate DeepSeek Expert Diffs

```bash
python -c "
from pipeline.model_utils.model_factory import construct_model_base
from pipeline.model_utils.model_card_factory import create_model_card

model_base = construct_model_base('deepseek-ai/deepseek-moe-16b-chat')
card = create_model_card(model_base)

card.generate_expert_diffs(
    harmful_dataset_path='dataset/splits/harmful_train.json',
    harmless_dataset_path='dataset/splits/harmless_train.json',
    num_harmful=None,
    num_harmless=200
)
"
```

**Note:** With 64 experts per layer (27 MoE layers, layer 0 is dense), the diff file will be large.

#### Step 3.5: Run DeepSeek Pipeline

```bash
python run_pipeline_expert_specific.py \
  --model_path deepseek-ai/DeepSeek-V2-Lite \
  --threshold 15.0 \
  --n_train 100 \
  --n_val 50 \
  --n_test 50
```

**Monitor:**
- Memory usage (should fit in 40GB)
- Top-6 routing works correctly
- Dense layer (layer 0) is skipped

---

## Critical Files

### New Files (Create)
1. `pipeline/model_utils/model_card.py` - Abstract base
2. `pipeline/model_utils/oss_model_card.py` - OSS openai variant
3. `pipeline/model_utils/unsloth_oss_model_card.py` - OSS unsloth variant
4. `pipeline/model_utils/mixtral_model_card.py` - Mixtral
5. `pipeline/model_utils/deepseek_model_card.py` - DeepSeek
6. `pipeline/model_utils/model_card_factory.py` - Factory
7. `pipeline/model_utils/deepseek_model.py` - DeepSeek ModelBase
8. `expert_diff_generator.py` - Shared utility
9. `inspect_deepseek_architecture.py` - Architecture inspection
10. `test_model_card_oss.py` - Phase 1 validation
11. `test_mixtral_bias.py` - Phase 2 bias testing

### Modified Files
1. `expert_specific_activations.py` - Use model cards
2. `expert_intervention.py` - Use model cards
3. `expert_selection_mlp.py` - Use model cards (if needed)
4. `run_pipeline_expert_specific.py` - Auto-detect model card
5. `pipeline/model_utils/model_factory.py` - Add DeepSeek

---

## Verification Plan

After implementation:

```bash
# 1. Validate OSS (both variants)
python run_pipeline_expert_specific.py \
  --model_path openai/gpt-oss-20b \
  --threshold 15.0 --n_train 100 --n_val 50 --n_test 50

python run_pipeline_expert_specific.py \
  --model_path unsloth/gpt-oss-20b-unsloth-bnb-4bit \
  --threshold 15.0 --n_train 100 --n_val 50 --n_test 50

# 2. Validate Mixtral
python run_pipeline_expert_specific.py \
  --model_path mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --threshold 15.0 --n_train 100 --n_val 50 --n_test 50

# 3. Validate DeepSeek
python run_pipeline_expert_specific.py \
  --model_path deepseek-ai/deepseek-moe-16b-chat \
  --threshold 15.0 --n_train 100 --n_val 50 --n_test 50

# 4. Compare results
python compare_models.py \
  --models oss unsloth_oss mixtral deepseek
```

**Success Criteria:**
- All 4 model variants complete without errors
- Expert diffs generated automatically if missing
- MLP output format detected correctly at init
- No runtime format checks
- Memory usage acceptable
- Intervention effectiveness varies by model (expected)
