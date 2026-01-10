"""
Controlled Fine-tuning of MoE Models with Expert Dropout

This script implements three controlled experimental conditions for studying
how to diffuse refusal behavior across MoE experts:

1. ROUTER-ONLY: Train only routing parameters, freeze experts
   - Tests if changing routing patterns alone can diffuse refusal

2. EXPERT-ONLY: Train only expert weights, freeze routers
   - Tests if teaching all experts to refuse (with frozen routing) works

3. COMBINED: Train both routers and experts
   - Tests the full effect and interaction between mechanisms

All modes use stochastic expert dropout during training to prevent
refusal specialization in specific experts.
"""

import os
import sys
import torch
import argparse
from pathlib import Path
from typing import Optional, Dict, List
import json

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, TaskType

# Add alignment directory to path
alignment_dir = os.path.dirname(os.path.abspath(__file__))
if alignment_dir not in sys.path:
    sys.path.insert(0, alignment_dir)

from expert_dropout import StochasticExpertDropout, PerTokenExpertDropout


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
    dataset = load_dataset("allenai/wildjailbreak", split=split)

    print(f"Loaded {len(dataset)} examples")

    # Filter to examples with refusal responses
    # We want to train the model to refuse simple harmful requests
    dataset = dataset.filter(lambda x: x['data_type'] == 'vanilla_harmful')

    print(f"Filtered to {len(dataset)} refusal examples")

    if max_samples and max_samples < len(dataset):
        # Shuffle and sample
        dataset = dataset.shuffle(seed=seed).select(range(max_samples))
        print(f"Sampled {len(dataset)} examples")

    def prepare_example(example):
        """
        Prepare a single example for training.

        WildJailbreak format:
        - 'vanilla_prompt': The base harmful request
        - 'response': The model's response (we filtered to refusals)
        """
        prompt = example['vanilla']
        response = example['completion']

        # Format using OSS chat template
        # <|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>{response}
        formatted_text = f"<|start|>user<|message|>{prompt}<|end|><|start|>assistant<|channel|>final<|message|>{response}"

        return {"text": formatted_text, "prompt": prompt, "response": response}

    # Prepare all examples
    print("Formatting examples...")
    dataset = dataset.map(
        prepare_example,
        desc="Formatting"
    )

    # Tokenize with labels for training only on responses
    def tokenize_function(examples):
        """
        Tokenize and create labels that mask the prompt tokens.
        We only want to train on the response portion.
        """
        # Tokenize full text
        full_encoding = tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # Tokenize just the prompt to know where it ends
        prompt_text = [
            f"<|start|>user<|message|>{p}<|end|><|start|>assistant<|channel|>final<|message|>"
            for p in examples["prompt"]
        ]

        prompt_encoding = tokenizer(
            prompt_text,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # Create labels that mask prompt tokens (set to -100)
        labels = []
        for i, (input_ids, prompt_ids) in enumerate(zip(full_encoding["input_ids"], prompt_encoding["input_ids"])):
            label = input_ids.copy()
            prompt_len = len(prompt_ids)

            # Mask prompt tokens
            label[:prompt_len] = [-100] * prompt_len
            labels.append(label)

        full_encoding["labels"] = labels

        return full_encoding

    print("Tokenizing dataset (masking prompt tokens in loss)...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )

    return tokenized_dataset


def setup_model_for_training(
    model,
    training_mode: str,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
):
    """
    Configure model for one of three training modes.

    Args:
        model: The model to configure
        training_mode: One of 'router', 'expert', 'combined'
        lora_rank: LoRA rank for expert training
        lora_alpha: LoRA alpha parameter
        lora_dropout: LoRA dropout rate

    Returns:
        Configured model (with PEFT if needed)
    """
    num_layers = len(model.model.layers)
    num_experts = model.config.num_local_experts

    print(f"\n{'='*80}")
    print(f"SETTING UP MODEL FOR {training_mode.upper()} TRAINING")
    print(f"{'='*80}")

    if training_mode == "router":
        print("\nMode: ROUTER-ONLY")
        print("Training: Router parameters only")
        print("Frozen: All expert weights, attention, embeddings")

        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze only router parameters
        trainable_params = 0
        for layer_idx in range(num_layers):
            router = model.model.layers[layer_idx].mlp.router
            for param in router.parameters():
                param.requires_grad = True
                trainable_params += param.numel()

        print(f"\nTrainable parameters: {trainable_params:,}")
        print(f"Router params per layer: {trainable_params // num_layers:,}")

        return model

    elif training_mode == "expert":
        print("\nMode: EXPERT-ONLY")
        print("Training: Expert weights only (via LoRA)")
        print("Frozen: Routers, attention, embeddings")

        # Apply LoRA to expert projections only
        # Target all expert linear layers: gate_proj, up_proj, down_proj
        target_modules = []
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts):
                target_modules.extend([
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj",
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj",
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj",
                ])

        print(f"\nApplying LoRA to {len(target_modules)} expert projections...")
        print(f"LoRA rank: {lora_rank}")
        print(f"LoRA alpha: {lora_alpha}")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

        return model

    elif training_mode == "combined":
        print("\nMode: COMBINED")
        print("Training: Both routers AND expert weights (via LoRA)")
        print("Frozen: Attention, embeddings")

        # Apply LoRA to experts
        target_modules = []
        for layer_idx in range(num_layers):
            for expert_idx in range(num_experts):
                target_modules.extend([
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj",
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj",
                    f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj",
                ])

        print(f"\nApplying LoRA to {len(target_modules)} expert projections...")
        print(f"LoRA rank: {lora_rank}")
        print(f"LoRA alpha: {lora_alpha}")

        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            modules_to_save=None  # Don't auto-save router, we'll handle manually
        )

        model = get_peft_model(model, lora_config)

        # Additionally unfreeze routers
        trainable_router_params = 0
        for layer_idx in range(num_layers):
            router = model.base_model.model.model.layers[layer_idx].mlp.router
            for param in router.parameters():
                param.requires_grad = True
                trainable_router_params += param.numel()

        print(f"\nAlso unfreezing {trainable_router_params:,} router parameters")
        model.print_trainable_parameters()

        return model

    else:
        raise ValueError(f"Unknown training mode: {training_mode}")


class ExpertDropoutTrainer(Trainer):
    """
    Custom Trainer that applies expert dropout during training.
    """

    def __init__(
        self,
        expert_dropout: StochasticExpertDropout,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.expert_dropout = expert_dropout

    def training_step(self, model, inputs):
        """
        Override training step to enable expert dropout.
        """
        # Enable dropout for this training step
        if not self.expert_dropout.enabled:
            self.expert_dropout.enable()

        # Perform standard training step
        model.train()
        loss = super().training_step(model, inputs)

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
        description="Controlled fine-tuning of MoE models with expert dropout"
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

    # LoRA arguments (for expert/combined modes)
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="LoRA rank for expert training"
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
        default=0.05,
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
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length"
    )

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="openai/gpt-oss-20b",
        help="Model to fine-tune"
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
        default=5e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
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

    # Auto-generate output dir based on training mode
    if args.output_dir is None:
        args.output_dir = f"./checkpoints/moe_{args.training_mode}_dropout{int(args.expert_dropout_rate*100)}"

    # Set random seeds
    torch.manual_seed(args.seed)
    import random
    import numpy as np
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("="*80)
    print("CONTROLLED MoE FINE-TUNING WITH EXPERT DROPOUT")
    print("="*80)
    print(f"Training mode: {args.training_mode.upper()}")
    print(f"Model: {args.model_name}")
    print(f"Dataset: WildJailbreak")
    print(f"Expert dropout rate: {args.expert_dropout_rate:.1%}")
    print(f"Per-token dropout: {args.use_per_token_dropout}")
    print(f"Output directory: {args.output_dir}")
    print("="*80)

    # Load tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"\nLoading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()

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
        max_length=args.max_length,
        seed=args.seed
    )

    print(f"\nTraining samples: {len(train_dataset)}")

    # Create expert dropout module
    print("\n" + "="*80)
    print("INITIALIZING EXPERT DROPOUT")
    print("="*80)

    # Determine layers to exclude
    num_layers = len(model.model.layers) if args.training_mode == "router" else len(model.base_model.model.model.layers)
    exclude_layers = []

    if args.exclude_first_n_layers > 0:
        exclude_layers.extend(range(args.exclude_first_n_layers))

    if args.exclude_last_n_layers > 0:
        exclude_layers.extend(range(num_layers - args.exclude_last_n_layers, num_layers))

    # Create dropout module
    # Note: For PEFT models, we need to access the base model
    dropout_model = model if args.training_mode == "router" else model.base_model.model

    DropoutClass = PerTokenExpertDropout if args.use_per_token_dropout else StochasticExpertDropout

    expert_dropout = DropoutClass(
        model=dropout_model,
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
        bf16=True,
        logging_dir=f"{args.output_dir}/logs",
        report_to=["tensorboard"],
        seed=args.seed,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        evaluation_strategy="no",
        # Optimizer settings
        optim="adamw_torch",
        weight_decay=0.01,
        max_grad_norm=1.0,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Save configuration
    config_path = Path(args.output_dir) / "training_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nSaved training configuration to: {config_path}")

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
    trainer.save_model(final_output_dir)
    tokenizer.save_pretrained(final_output_dir)

    print(f"Model saved to: {final_output_dir}")

    print("\n" + "="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print(f"\nTraining mode: {args.training_mode.upper()}")
    print(f"Expert dropout rate: {args.expert_dropout_rate:.1%}")
    print(f"\nNext steps:")
    print("  1. Test the fine-tuned model on harmful prompts")
    print("  2. Use expert_intervention_hooks_v3.py to test robustness")
    print("  3. Compare expert routing patterns before/after fine-tuning")
    print("  4. Run expert interventions to test jailbreak resistance")
    print("\nExperimental comparisons:")
    print("  - Router-only vs Expert-only vs Combined")
    print("  - With vs without expert dropout")
    print("  - Before vs after fine-tuning")
    print("="*80)


if __name__ == "__main__":
    main()
