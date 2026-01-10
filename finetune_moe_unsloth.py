"""
Controlled Fine-tuning of MoE Models with Expert Dropout (Unsloth Version)

This script implements controlled experimental conditions for studying
how to diffuse refusal behavior across MoE experts using the unsloth library
for optimized training.

Training modes:
1. ROUTER-ONLY: Train only routing parameters, freeze experts
2. EXPERT-ONLY: Train only expert weights, freeze routers
3. COMBINED: Train both routers and experts

All modes use stochastic expert dropout during training.
"""

import unsloth
import os
import sys

# Add alignment directory to path
alignment_dir = os.path.dirname(os.path.abspath(__file__))
if alignment_dir not in sys.path:
    sys.path.insert(0, alignment_dir)

# Fix HF cache paths BEFORE importing any HF libraries
import fix_hf_cache

import torch
import argparse
from pathlib import Path
from typing import Optional
import json

from datasets import load_dataset
from transformers import TrainingArguments, Trainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from dataclasses import dataclass
from typing import Dict, List, Any
import torch

from expert_dropout_unsloth import StochasticExpertDropoutUnsloth, PerTokenExpertDropoutUnsloth


@dataclass
class DataCollatorForCausalLM:
    """
    Custom data collator that properly pads sequences for causal language modeling.
    """
    tokenizer: Any

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Get max length in batch
        max_length = max(len(f["input_ids"]) for f in features)

        # Pad to multiple of 8 for efficiency
        max_length = ((max_length + 7) // 8) * 8

        # Prepare batch
        batch = {
            "input_ids": [],
            "attention_mask": [],
            "labels": []
        }

        for feature in features:
            input_ids = feature["input_ids"]
            labels = feature["labels"]

            # Calculate padding length
            padding_length = max_length - len(input_ids)

            # Pad input_ids and attention_mask
            padded_input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = [1] * len(input_ids) + [0] * padding_length

            # Pad labels with -100 (ignore index)
            padded_labels = labels + [-100] * padding_length

            batch["input_ids"].append(padded_input_ids)
            batch["attention_mask"].append(attention_mask)
            batch["labels"].append(padded_labels)

        # Convert to tensors
        return {
            "input_ids": torch.tensor(batch["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(batch["attention_mask"], dtype=torch.long),
            "labels": torch.tensor(batch["labels"], dtype=torch.long)
        }


def prepare_wildjailbreak_dataset(
    tokenizer,
    split: str = "train",
    max_samples: Optional[int] = None,
    max_length: int = 512,
    seed: int = 42
):
    """
    Prepare WildJailbreak dataset for alignment fine-tuning.

    WildJailbreak contains adversarial prompts with vanilla (safe) and adversarial tactics.
    We'll use the vanilla prompts with refusal responses to teach proper refusal behavior.

    Args:
        tokenizer: The model tokenizer
        split: Dataset split to use
        max_samples: Maximum number of samples (None for all)
        max_length: Maximum sequence length
        seed: Random seed for sampling

    Returns:
        Tokenized dataset ready for training
    """
    print(f"Loading WildJailbreak dataset (split: {split})...")

    # Load dataset
    dataset = load_dataset("allenai/wildjailbreak", split, split="train", delimiter="\t", keep_default_na=False)

    print(f"Loaded {len(dataset)} examples")

    # Filter to examples with refusal responses
    dataset = dataset.filter(lambda x: x['data_type'] == 'vanilla_harmful')

    print(f"Filtered to {len(dataset)} refusal examples")

    if max_samples and max_samples < len(dataset):
        # Shuffle and sample
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        print(f"Sampled {len(dataset)} examples")

    def tokenize_example(example):
        """
        Tokenize example for standard Trainer.

        WildJailbreak format:
        - 'vanilla': The base harmful request
        - 'completion': The model's response (we filtered to refusals)
        """
        prompt = example['vanilla']
        response = example['completion']

        # Format using OSS chat template
        formatted_text = f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>{response}"

        # Tokenize
        tokenized = tokenizer(
            formatted_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # Labels are the same as input_ids for causal LM
        # Make sure labels is a proper list (not a reference)
        tokenized["labels"] = list(tokenized["input_ids"])

        return tokenized

    # Tokenize all examples
    print("Tokenizing examples...")
    dataset = dataset.map(
        tokenize_example,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    # Don't set format - let the custom data collator handle conversion
    return dataset


def setup_model_for_training(
    model,
    training_mode: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
):
    """
    Configure model for one of three training modes using LoRA.

    Args:
        model: The unsloth model to configure
        training_mode: One of 'router', 'expert', 'combined'
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate

    Returns:
        Model with LoRA adapters configured
    """
    print(f"\n{'='*80}")
    print(f"SETTING UP MODEL FOR {training_mode.upper()} TRAINING")
    print(f"{'='*80}")

    # Determine target modules based on training mode
    # Note: For OSS-20B MoE model:
    # - Router is at: mlp.router.linear
    # - Experts are at: mlp.experts.gate_up_projs.X and mlp.experts.down_projs.X (X=0..31)
    # - Attention is at: self_attn.{q,k,v,o}_proj

    # Get number of experts from config
    num_experts = model.config.num_local_experts

    if training_mode == "router":
        print("\nMode: ROUTER-ONLY")
        print("Training: Router parameters")
        print("Frozen: All expert weights, attention, embeddings")

        # Target the linear layer inside the router
        target_modules = ["router.linear"]

    elif training_mode == "expert":
        print("\nMode: EXPERT-ONLY")
        print("Training: Expert MLP weights only")
        print("Frozen: Routers, attention, embeddings")

        # GPT-OSS MoE requires indexing each expert explicitly
        # Unsloth's simple syntax only works when training attention + MLP together
        target_modules = []
        for i in range(num_experts):
            target_modules.append(f"gate_up_projs.{i}")
            target_modules.append(f"down_projs.{i}")

        print(f"  Targeting {num_experts} experts × 2 projections = {len(target_modules)} modules")

    elif training_mode == "combined":
        print("\nMode: COMBINED")
        print("Training: Both routers AND expert weights")
        print("Frozen: Attention, embeddings")

        # Target router and all expert layers (indexed)
        target_modules = ["router.linear"]
        for i in range(num_experts):
            target_modules.append(f"gate_up_projs.{i}")
            target_modules.append(f"down_projs.{i}")

        print(f"  Targeting router + {num_experts} experts × 2 projections")

    else:
        raise ValueError(f"Unknown training mode: {training_mode}")

    # Use gradient checkpointing for all modes
    use_grad_checkpoint = "unsloth"

    # Apply LoRA using unsloth's optimized method
    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_rank,
        target_modules=target_modules,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing=use_grad_checkpoint,
        random_state=42,
    )

    return model


class ExpertDropoutTrainer(Trainer):
    """
    Custom Trainer that applies expert dropout during training.
    """

    def __init__(
        self,
        expert_dropout: StochasticExpertDropoutUnsloth,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.expert_dropout = expert_dropout

    def training_step(self, *args, **kwargs):
        """
        Override training step to enable expert dropout.
        """
        # Enable dropout for this training step
        if not self.expert_dropout.enabled:
            self.expert_dropout.enable()

        # Perform standard training step
        loss = super().training_step(*args, **kwargs)

        return loss

    def evaluate(self, *args, **kwargs):
        """
        Override evaluate to disable expert dropout during evaluation.
        """
        # Disable dropout during evaluation
        self.expert_dropout.disable()

        return super().evaluate(*args, **kwargs)

    def prediction_step(self, *args, **kwargs):
        """
        Override prediction to ensure dropout is disabled.
        """
        # Disable dropout during prediction
        if self.expert_dropout.enabled:
            self.expert_dropout.disable()

        return super().prediction_step(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Controlled fine-tuning of MoE models with expert dropout (unsloth)"
    )

    # Training mode (required)
    parser.add_argument(
        "--training_mode",
        type=str,
        required=True,
        choices=["router", "expert", "combined"],
        help="Training mode: 'router' (router-only), 'expert' (expert-only), or 'combined' (both)"
    )

    # Expert dropout arguments
    parser.add_argument(
        "--expert_dropout_rate",
        type=float,
        default=0.3,
        help="Fraction of experts to drop per layer (0.0 to 1.0)"
    )
    parser.add_argument(
        "--use_per_token_dropout",
        action="store_true",
        help="Use per-token expert dropout instead of per-batch"
    )
    parser.add_argument(
        "--exclude_first_n_layers",
        type=int,
        default=0,
        help="Number of initial layers to exclude from dropout"
    )
    parser.add_argument(
        "--exclude_last_n_layers",
        type=int,
        default=0,
        help="Number of final layers to exclude from dropout"
    )

    # LoRA arguments
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha parameter"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.0,
        help="LoRA dropout rate"
    )

    # Dataset arguments
    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train",
        help="WildJailbreak split to use"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to use"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="unsloth/gpt-oss-20b-unsloth-bnb-4bit",
        help="Model to fine-tune (unsloth model)"
    )

    # Training arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for checkpoints (auto-generated if None)"
    )
    parser.add_argument(
        "--num_train_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=8,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate (default: auto-select based on mode)"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=10,
        help="Warmup steps"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging frequency"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Checkpoint save frequency"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )

    args = parser.parse_args()

    # Auto-select learning rate based on mode if not specified
    if args.learning_rate is None:
        if args.training_mode == "router":
            args.learning_rate = 1e-5  # Much lower for router-only (very sensitive)
        elif args.training_mode == "expert":
            args.learning_rate = 2e-4  # Standard for expert training
        else:  # combined
            args.learning_rate = 5e-5  # Medium for both

    # Auto-generate output dir based on training mode
    if args.output_dir is None:
        args.output_dir = f"./checkpoints/moe_unsloth_{args.training_mode}_dropout{int(args.expert_dropout_rate*100)}"

    # Set random seeds
    torch.manual_seed(args.seed)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("="*80)
    print("CONTROLLED MoE FINE-TUNING WITH EXPERT DROPOUT (UNSLOTH)")
    print("="*80)
    print(f"Training mode: {args.training_mode.upper()}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: WildJailbreak")
    print(f"Expert dropout rate: {args.expert_dropout_rate:.1%}")
    print(f"Per-token dropout: {args.use_per_token_dropout}")
    print(f"Learning rate: {args.learning_rate:.2e}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Load model and tokenizer using unsloth
    print("\nLoading model with unsloth...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,  # Auto-detect optimal dtype
        load_in_4bit=True,
    )

    print(f"Model loaded: {type(model).__name__}")

    # Setup model for specific training mode
    model = setup_model_for_training(
        model=model,
        training_mode=args.training_mode,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )

    # Prepare dataset
    print("\n" + "="*80)
    print("PREPARING DATASET")
    print("="*80)

    train_dataset = prepare_wildjailbreak_dataset(
        tokenizer=tokenizer,
        split=args.dataset_split,
        max_samples=args.max_samples,
        max_length=args.max_seq_length,
        seed=args.seed
    )

    print(f"\nTraining samples: {len(train_dataset)}")

    # Create expert dropout module
    print("\n" + "="*80)
    print("INITIALIZING EXPERT DROPOUT")
    print("="*80)

    # Determine layers to exclude
    # Get base model (unwrap from PEFT if needed)
    # After PEFT wrapping, structure is: PeftModel -> base_model -> model -> layers
    base_model = model
    while hasattr(base_model, 'model') and not hasattr(base_model, 'layers'):
        base_model = base_model.model

    num_layers = len(base_model.layers)
    exclude_layers = []

    if args.exclude_first_n_layers > 0:
        exclude_layers.extend(range(args.exclude_first_n_layers))

    if args.exclude_last_n_layers > 0:
        exclude_layers.extend(range(num_layers - args.exclude_last_n_layers, num_layers))

    DropoutClass = PerTokenExpertDropoutUnsloth if args.use_per_token_dropout else StochasticExpertDropoutUnsloth

    expert_dropout = DropoutClass(
        model=base_model,
        dropout_rate=args.expert_dropout_rate,
        exclude_layers=exclude_layers,
        seed=args.seed
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        fp16=False,  # Disable mixed precision for router training stability
        bf16=False,
        logging_dir=f"{args.output_dir}/logs",
        report_to=[],  # Disable reporting (tensorboard not installed)
        seed=args.seed,
        dataloader_num_workers=0,  # Disable multiprocessing to avoid batching issues
        # Optimizer settings (using unsloth tutorial configuration to avoid CUDA errors)
        optim="adamw_8bit",  # Use unsloth's optimized 8-bit AdamW
        weight_decay=0.001,  # Tutorial setting (prevents CUDA errors at step 285+)
        lr_scheduler_type="linear",  # Tutorial setting for stability
        max_grad_norm=0.5,  # Aggressive gradient clipping for stability
        # Logging for diagnostics
        log_level="warning",
        logging_first_step=True,
    )

    # Save configuration
    config_path = Path(args.output_dir) / "training_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nSaved training configuration to: {config_path}")

    # Data collator for padding
    data_collator = DataCollatorForCausalLM(tokenizer=tokenizer)

    # Create trainer with expert dropout
    trainer = ExpertDropoutTrainer(
        expert_dropout=expert_dropout,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    # Train
    print("\n" + "="*80)
    print("STARTING TRAINING")
    print("="*80)

    trainer.train()

    # Save final model
    print("\n" + "="*80)
    print("SAVING FINAL MODEL")
    print("="*80)

    final_output_dir = os.path.join(args.output_dir, "final")
    model.save_pretrained(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print(f"Model saved to: {final_output_dir}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nTraining mode: {args.training_mode.upper()}")
    print(f"Expert dropout rate: {args.expert_dropout_rate:.1%}")
    print(f"\nNext steps:")
    print("  1. Test the fine-tuned model on harmful prompts")
    print("  2. Use expert intervention tools to test robustness")
    print("  3. Compare expert routing patterns before/after fine-tuning")
    print("  4. Run expert interventions to test jailbreak resistance")
    print("\nExperimental comparisons:")
    print("  - Router-only vs Expert-only vs Combined")
    print("  - With vs without expert dropout")
    print("  - Before vs after fine-tuning")
    print("="*80)


if __name__ == "__main__":
    main()
