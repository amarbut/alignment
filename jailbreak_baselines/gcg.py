"""
GCG: Greedy Coordinate Gradient Attack

Based on: Zou et al., 2023
Paper: "Universal and Transferable Adversarial Attacks on Aligned Language Models"
ArXiv: https://arxiv.org/abs/2307.15043
GitHub: https://github.com/llm-attacks/llm-attacks

This is a standalone implementation that does not require nanogcg (which has
incompatible dependencies with unsloth/transformers). If nanogcg is available,
it can optionally be used via use_nanogcg=True.

GCG is a white-box attack that optimizes an adversarial suffix by:
1. Computing gradients of loss w.r.t. one-hot token encodings
2. Selecting top-k candidates based on gradient information
3. Evaluating candidates and keeping the best one
4. Iterating until convergence or max steps

Memory considerations for A100-40GB with 4-bit quantized models:
- The model weights take ~10GB in 4-bit
- Gradient computation requires additional memory for activations
- Use search_width=128-256 and batch_size=64-128 for stability
"""

import gc
import json
import os
import random
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm

from .base import JailbreakMethod


@dataclass
class GCGConfig:
    """Configuration for GCG attack."""

    # Optimization parameters
    num_steps: int = 250
    search_width: int = 256  # Reduced from 512 for memory efficiency
    topk: int = 128  # Reduced from 256 for memory efficiency
    batch_size: int = 64  # Explicit batch size for evaluation

    # Token replacement
    n_replace: int = 1  # Number of tokens to replace per candidate
    optim_str_init: str = "! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !"

    # Loss function
    use_mellowmax: bool = False
    mellowmax_alpha: float = 1.0

    # Optimization controls
    early_stop: bool = False
    use_prefix_cache: bool = True
    allow_non_ascii: bool = False

    # Target strings for optimization (model should output these)
    target_strings: List[str] = field(default_factory=lambda: [
        "Sure, here is",
        "Sure, I can help",
        "Of course, here's",
        "Absolutely, let me",
        "Certainly! Here's",
    ])

    # Logging
    verbosity: str = "INFO"
    seed: Optional[int] = 42


class GCGAttack(JailbreakMethod):
    """
    Greedy Coordinate Gradient attack for jailbreaking LLMs.

    This is a white-box attack that requires gradient access to the model.
    It optimizes an adversarial suffix that, when appended to a harmful prompt,
    increases the probability of the model producing an affirmative response.
    """

    def __init__(
        self,
        model=None,
        tokenizer=None,
        config: Optional[GCGConfig] = None,
        use_nanogcg: bool = False,
        device: str = "cuda",
    ):
        """
        Initialize GCG attack.

        Args:
            model: The target language model (required for optimization)
            tokenizer: The tokenizer for the model
            config: GCG configuration options
            use_nanogcg: Whether to use nanogcg library if available (default: False)
            device: Device to run on ("cuda" or "cpu")
        """
        super().__init__("GCG")
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or GCGConfig()
        self.use_nanogcg = use_nanogcg
        self.device = device

        # Cache for optimized suffixes (prompt -> suffix)
        self.suffix_cache: Dict[str, str] = {}

        # Set random seed
        if self.config.seed is not None:
            random.seed(self.config.seed)
            torch.manual_seed(self.config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.config.seed)

    def set_model(self, model, tokenizer):
        """Set or update the model and tokenizer."""
        self.model = model
        self.tokenizer = tokenizer

    def _get_embedding_layer(self):
        """Get the embedding layer from the model."""
        # Handle different model architectures
        if hasattr(self.model, 'model'):
            if hasattr(self.model.model, 'embed_tokens'):
                return self.model.model.embed_tokens
            elif hasattr(self.model.model, 'model') and hasattr(self.model.model.model, 'embed_tokens'):
                return self.model.model.model.embed_tokens
        if hasattr(self.model, 'get_input_embeddings'):
            return self.model.get_input_embeddings()
        raise ValueError("Could not find embedding layer in model")

    def _compute_loss(
        self,
        input_ids: torch.Tensor,
        target_ids: torch.Tensor,
        target_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cross-entropy loss for target tokens.

        Args:
            input_ids: Full input sequence [batch, seq_len]
            target_ids: Target token IDs to match
            target_mask: Mask indicating target positions

        Returns:
            Loss value
        """
        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits

        # Shift logits and labels for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        # Compute loss only on target positions
        loss = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            reduction='none'
        )
        loss = loss.reshape(shift_logits.size(0), shift_logits.size(1))

        # Apply mask and average
        if target_mask is not None:
            target_mask = target_mask[:, 1:]  # Shift mask too
            loss = (loss * target_mask).sum(dim=1) / target_mask.sum(dim=1).clamp(min=1)
        else:
            loss = loss.mean(dim=1)

        return loss

    def _compute_token_gradients(
        self,
        input_ids: torch.Tensor,
        suffix_slice: slice,
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute gradients w.r.t. one-hot token encodings.

        This is the core of the GCG algorithm - we compute how changing
        each token in the suffix affects the loss.

        Args:
            input_ids: Input sequence without target [1, seq_len]
            suffix_slice: Slice indicating suffix positions in input_ids
            target_ids: Target token IDs [target_len]

        Returns:
            Gradients [suffix_len, vocab_size]
        """
        embed_layer = self._get_embedding_layer()
        vocab_size = embed_layer.weight.size(0)
        embed_dtype = embed_layer.weight.dtype

        # Ensure target_ids is 1D
        if target_ids.dim() > 1:
            target_ids = target_ids.squeeze(0)

        # Append target to input for full sequence
        full_ids = torch.cat([input_ids, target_ids.unsqueeze(0)], dim=1)

        # Get embeddings for the full sequence
        input_embeds = embed_layer(full_ids)

        # Create one-hot for suffix tokens with gradients
        suffix_ids = full_ids[:, suffix_slice]
        one_hot = F.one_hot(suffix_ids, num_classes=vocab_size).to(dtype=embed_dtype)
        one_hot.requires_grad_(True)

        # Replace suffix embeddings with one-hot @ embedding_weights
        suffix_embeds = one_hot @ embed_layer.weight
        input_embeds = input_embeds.clone()
        input_embeds[:, suffix_slice, :] = suffix_embeds

        # Forward pass
        outputs = self.model(inputs_embeds=input_embeds)
        logits = outputs.logits

        # Compute loss on target positions (predicting target tokens)
        # Target positions are after the original input_ids
        target_len = target_ids.size(0)
        # We want logits at positions [input_len-1, input_len, ..., input_len+target_len-2]
        # to predict tokens at positions [input_len, input_len+1, ..., input_len+target_len-1]
        input_len = input_ids.size(1)
        target_logits = logits[:, input_len-1:input_len+target_len-1, :]

        loss = F.cross_entropy(
            target_logits.reshape(-1, target_logits.size(-1)),
            target_ids,
        )

        # Backward pass
        loss.backward()

        # Return gradients w.r.t. one-hot
        return one_hot.grad.squeeze(0)

    def _sample_candidates(
        self,
        current_ids: torch.Tensor,
        gradients: torch.Tensor,
        suffix_slice: slice,
    ) -> torch.Tensor:
        """
        Sample candidate token replacements based on gradients.

        Args:
            current_ids: Current input sequence
            gradients: Gradients [suffix_len, vocab_size]
            suffix_slice: Slice for suffix positions

        Returns:
            Candidate sequences [search_width, seq_len]
        """
        suffix_len = gradients.size(0)

        # Find top-k tokens with most negative gradient (decrease loss)
        topk_tokens = (-gradients).topk(self.config.topk, dim=1).indices

        # Generate candidates by random replacement
        candidates = current_ids.repeat(self.config.search_width, 1)

        for i in range(self.config.search_width):
            # Randomly select positions to replace
            positions = random.sample(range(suffix_len), self.config.n_replace)

            for pos in positions:
                # Randomly select from top-k tokens for this position
                new_token = topk_tokens[pos, random.randint(0, self.config.topk - 1)]
                candidates[i, suffix_slice.start + pos] = new_token

        return candidates

    def optimize_suffix(
        self,
        prompt: str,
        target: Optional[str] = None,
    ) -> Tuple[str, float, List[Dict]]:
        """
        Optimize an adversarial suffix for a given prompt.

        Args:
            prompt: The harmful prompt to attack
            target: Target string for the model to output (default: "Sure, here is")

        Returns:
            Tuple of (best_suffix, best_loss, optimization_history)
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model and tokenizer must be set before optimization")

        if target is None:
            target = random.choice(self.config.target_strings)

        # Check if we can use nanogcg
        if self.use_nanogcg:
            try:
                return self._optimize_with_nanogcg(prompt, target)
            except ImportError:
                print("nanogcg not available, falling back to custom implementation")
            except Exception as e:
                print(f"nanogcg failed: {e}, falling back to custom implementation")

        return self._optimize_custom(prompt, target)

    def _optimize_with_nanogcg(
        self,
        prompt: str,
        target: str,
    ) -> Tuple[str, float, List[Dict]]:
        """Use nanogcg library for optimization."""
        import nanogcg
        from nanogcg import GCGConfig as NanoGCGConfig

        # Create nanogcg config
        nano_config = NanoGCGConfig(
            num_steps=self.config.num_steps,
            search_width=self.config.search_width,
            topk=self.config.topk,
            batch_size=self.config.batch_size,
            n_replace=self.config.n_replace,
            optim_str_init=self.config.optim_str_init,
            use_mellowmax=self.config.use_mellowmax,
            mellowmax_alpha=self.config.mellowmax_alpha,
            early_stop=self.config.early_stop,
            use_prefix_cache=self.config.use_prefix_cache,
            allow_non_ascii=self.config.allow_non_ascii,
            seed=self.config.seed,
            verbosity=self.config.verbosity,
        )

        # Format message for nanogcg (include optim_str placeholder)
        message = f"{prompt}{{optim_str}}"

        # Run optimization
        result = nanogcg.run(
            self.model,
            self.tokenizer,
            message,
            target,
            nano_config,
        )

        # Build history from result
        history = [
            {"step": i, "loss": loss, "suffix": suffix}
            for i, (loss, suffix) in enumerate(zip(result.losses, result.strings))
        ]

        return result.best_string, result.best_loss, history

    def _optimize_custom(
        self,
        prompt: str,
        target: str,
    ) -> Tuple[str, float, List[Dict]]:
        """Custom GCG implementation (fallback)."""
        # Tokenize prompt and target
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = self.tokenizer.encode(target, add_special_tokens=False)
        suffix_ids = self.tokenizer.encode(self.config.optim_str_init, add_special_tokens=False)

        # Build full sequence: prompt + suffix + target
        full_ids = prompt_ids + suffix_ids
        suffix_slice = slice(len(prompt_ids), len(prompt_ids) + len(suffix_ids))

        input_ids = torch.tensor([full_ids], device=self.device)
        target_tensor = torch.tensor([target_ids], device=self.device)

        best_suffix = self.config.optim_str_init
        best_loss = float('inf')
        history = []

        # Enable gradients for optimization
        was_training = self.model.training
        self.model.train()

        try:
            for step in tqdm(range(self.config.num_steps), desc="GCG Optimization"):
                # Compute gradients
                self.model.zero_grad()
                gradients = self._compute_token_gradients(
                    input_ids, suffix_slice, target_tensor.squeeze(0)
                )

                # Sample candidates
                candidates = self._sample_candidates(input_ids, gradients, suffix_slice)

                # Evaluate candidates in batches
                candidate_losses = []
                for i in range(0, candidates.size(0), self.config.batch_size):
                    batch = candidates[i:i + self.config.batch_size]

                    # Append target to each candidate
                    batch_with_target = torch.cat([
                        batch,
                        target_tensor.expand(batch.size(0), -1)
                    ], dim=1)

                    # Create target mask (only compute loss on target positions)
                    target_mask = torch.zeros_like(batch_with_target, dtype=torch.float)
                    target_mask[:, -target_tensor.size(1):] = 1.0

                    with torch.no_grad():
                        losses = self._compute_loss(batch_with_target, target_tensor, target_mask)
                    candidate_losses.append(losses)

                candidate_losses = torch.cat(candidate_losses)

                # Select best candidate
                best_idx = candidate_losses.argmin()
                current_loss = candidate_losses[best_idx].item()

                if current_loss < best_loss:
                    best_loss = current_loss
                    input_ids = candidates[best_idx:best_idx+1]
                    best_suffix = self.tokenizer.decode(
                        input_ids[0, suffix_slice].tolist(),
                        skip_special_tokens=True
                    )

                history.append({
                    "step": step,
                    "loss": current_loss,
                    "best_loss": best_loss,
                    "suffix": best_suffix,
                })

                # Early stopping
                if self.config.early_stop and current_loss < 0.1:
                    break

                # Clear cache periodically
                if step % 10 == 0:
                    torch.cuda.empty_cache()

        finally:
            if not was_training:
                self.model.eval()

        return best_suffix, best_loss, history

    def transform(self, prompt: str) -> str:
        """
        Transform a prompt by appending an optimized adversarial suffix.

        Note: This requires the model to be set. For batch processing,
        consider using optimize_suffix directly and caching results.

        Args:
            prompt: Original harmful prompt

        Returns:
            Prompt with adversarial suffix appended
        """
        # Check cache first
        if prompt in self.suffix_cache:
            return f"{prompt} {self.suffix_cache[prompt]}"

        # Optimize suffix
        suffix, loss, _ = self.optimize_suffix(prompt)
        self.suffix_cache[prompt] = suffix

        return f"{prompt} {suffix}"

    def transform_batch(self, prompts: List[str]) -> List[str]:
        """
        Transform a batch of prompts with optimized suffixes.

        For efficiency, this optimizes each prompt and caches results.
        """
        results = []
        for prompt in tqdm(prompts, desc=f"Optimizing {self.name} suffixes"):
            results.append(self.transform(prompt))
            gc.collect()
            torch.cuda.empty_cache()
        return results

    def save_cache(self, path: str):
        """Save suffix cache to file."""
        with open(path, 'w') as f:
            json.dump(self.suffix_cache, f, indent=2)

    def load_cache(self, path: str):
        """Load suffix cache from file."""
        if os.path.exists(path):
            with open(path, 'r') as f:
                self.suffix_cache = json.load(f)

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata.update({
            "paper": "Universal and Transferable Adversarial Attacks on Aligned Language Models",
            "arxiv": "https://arxiv.org/abs/2307.15043",
            "venue": "arXiv 2023",
            "type": "white-box",
            "num_steps": self.config.num_steps,
            "search_width": self.config.search_width,
            "topk": self.config.topk,
            "use_nanogcg": self.use_nanogcg,
        })
        return metadata


def run_gcg_attack(
    model,
    tokenizer,
    prompts: List[Dict[str, str]],
    output_dir: str,
    config: Optional[GCGConfig] = None,
    max_new_tokens: int = 512,
    use_nanogcg: bool = False,
) -> Dict[str, Any]:
    """
    Run GCG attack on a set of prompts.

    This is a convenience function that handles the full attack pipeline:
    1. Initialize GCG attack
    2. Optimize suffix for each prompt
    3. Generate completions with adversarial prompts
    4. Save results

    Args:
        model: Target language model
        tokenizer: Model tokenizer
        prompts: List of prompt dicts with 'instruction' and 'category'
        output_dir: Directory to save results
        config: GCG configuration
        max_new_tokens: Max tokens to generate
        use_nanogcg: Whether to use nanogcg library

    Returns:
        Results dictionary with completions and optimization stats
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize attack
    attack = GCGAttack(
        model=model,
        tokenizer=tokenizer,
        config=config or GCGConfig(),
        use_nanogcg=use_nanogcg,
    )

    results = {
        "prompts": [],
        "optimization_stats": [],
    }

    device = next(model.parameters()).device

    for i, prompt_dict in enumerate(tqdm(prompts, desc="Running GCG attack")):
        original_prompt = prompt_dict["instruction"]

        # Optimize suffix
        suffix, loss, history = attack.optimize_suffix(original_prompt)

        # Create adversarial prompt
        adversarial_prompt = f"{original_prompt} {suffix}"

        # Generate completion
        # Format with chat template
        formatted = f"<|start|>user<|message|>{adversarial_prompt}<|end|><|start|>assistant<|channel|>final<|message|>"

        inputs = tokenizer(formatted, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        completion = tokenizer.decode(
            outputs[0][inputs["input_ids"].size(1):],
            skip_special_tokens=True,
        ).strip()

        results["prompts"].append({
            "prompt": original_prompt,
            "adversarial_prompt": adversarial_prompt,
            "suffix": suffix,
            "response": completion,
            "category": prompt_dict.get("category", "unknown"),
            "final_loss": loss,
        })

        results["optimization_stats"].append({
            "prompt_idx": i,
            "final_loss": loss,
            "num_steps": len(history),
            "history": history[-10:],  # Keep last 10 steps
        })

        # Save intermediate results
        if (i + 1) % 5 == 0:
            with open(os.path.join(output_dir, "gcg_results_partial.json"), "w") as f:
                json.dump(results, f, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    # Save final results
    with open(os.path.join(output_dir, "gcg_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    return results
