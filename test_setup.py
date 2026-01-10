"""
Quick validation script to test the setup before running full experiments.

This script:
1. Tests expert dropout implementation
2. Verifies WildJailbreak dataset access
3. Checks model loading and parameter counting
4. Validates training mode configurations

Run this before your full experiments to catch issues early!
"""

import os
import sys
import torch
import argparse

# Add alignment directory to path
alignment_dir = os.path.dirname(os.path.abspath(__file__))
if alignment_dir not in sys.path:
    sys.path.insert(0, alignment_dir)


def test_expert_dropout():
    """Test that expert dropout works correctly."""
    print("\n" + "="*80)
    print("TEST 1: Expert Dropout Implementation")
    print("="*80)

    try:
        from expert_dropout import StochasticExpertDropout
        from pipeline.model_utils.model_factory import construct_model_base

        print("Loading model...")
        model_base = construct_model_base("openai/gpt-oss-20b")
        model = model_base.model

        print("Creating expert dropout module...")
        dropout = StochasticExpertDropout(
            model=model,
            dropout_rate=0.3,
            exclude_layers=[0, 1]
        )

        # Test forward pass without dropout
        print("\nTesting forward pass WITHOUT dropout...")
        test_text = "Hello, how are you?"
        inputs = model_base.tokenizer(test_text, return_tensors="pt").to(model.device)

        model.eval()
        with torch.no_grad():
            outputs1 = model(**inputs)

        # Test forward pass with dropout (should have no effect in eval)
        print("Testing forward pass WITH dropout (eval mode - should be same)...")
        with dropout:
            model.eval()
            with torch.no_grad():
                outputs2 = model(**inputs)

        assert torch.allclose(outputs1.logits, outputs2.logits), "Outputs should be identical in eval mode!"
        print("✓ Eval mode works correctly")

        # Test forward pass with dropout in training mode
        print("Testing forward pass WITH dropout (training mode - should differ)...")
        with dropout:
            model.train()
            with torch.no_grad():
                outputs3 = model(**inputs)

        assert not torch.allclose(outputs1.logits, outputs3.logits), "Outputs should differ in training mode!"
        print("✓ Training mode dropout works correctly")

        print("\n✅ Expert dropout test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Expert dropout test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_wildjailbreak_access():
    """Test that we can access the WildJailbreak dataset."""
    print("\n" + "="*80)
    print("TEST 2: WildJailbreak Dataset Access")
    print("="*80)

    try:
        from datasets import load_dataset

        print("Loading WildJailbreak dataset...")
        dataset = load_dataset("allenai/wildjailbreak", split="train")

        print(f"✓ Loaded {len(dataset)} examples")

        # Check for refusal examples
        refusal_count = sum(1 for x in dataset if x['response_type'] == 'refusal')
        print(f"✓ Found {refusal_count} refusal examples")

        # Show a sample
        refusal_example = next(x for x in dataset if x['response_type'] == 'refusal')
        print(f"\nSample refusal example:")
        print(f"  Prompt: {refusal_example['vanilla'][:100]}...")
        print(f"  Response: {refusal_example['response'][:100]}...")

        print("\n✅ WildJailbreak access test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ WildJailbreak access test FAILED: {e}")
        print("\nMake sure you:")
        print("  1. Have accepted the dataset terms: https://huggingface.co/datasets/allenai/wildjailbreak")
        print("  2. Are logged in: huggingface-cli login")
        import traceback
        traceback.print_exc()
        return False


def test_training_modes():
    """Test that all three training modes configure correctly."""
    print("\n" + "="*80)
    print("TEST 3: Training Mode Configurations")
    print("="*80)

    try:
        from transformers import AutoModelForCausalLM
        from finetune_moe_controlled import setup_model_for_training

        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        # Test router-only mode
        print("\n--- Testing ROUTER-ONLY mode ---")
        router_model = setup_model_for_training(
            model=model,
            training_mode="router",
            lora_rank=8
        )

        # Count trainable params
        trainable_params = sum(p.numel() for p in router_model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in router_model.parameters())
        print(f"✓ Trainable: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.3f}%)")

        # Test expert-only mode (need fresh model)
        print("\n--- Testing EXPERT-ONLY mode ---")
        model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        expert_model = setup_model_for_training(
            model=model,
            training_mode="expert",
            lora_rank=8
        )
        print("✓ Expert-only configuration created")

        # Test combined mode (need fresh model)
        print("\n--- Testing COMBINED mode ---")
        model = AutoModelForCausalLM.from_pretrained(
            "openai/gpt-oss-20b",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

        combined_model = setup_model_for_training(
            model=model,
            training_mode="combined",
            lora_rank=8
        )
        print("✓ Combined configuration created")

        print("\n✅ Training mode configuration test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Training mode configuration test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_preparation():
    """Test that dataset preparation works."""
    print("\n" + "="*80)
    print("TEST 4: Dataset Preparation")
    print("="*80)

    try:
        from transformers import AutoTokenizer
        from finetune_moe_controlled import prepare_wildjailbreak_dataset

        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-20b", trust_remote_code=True)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        print("Preparing dataset (10 samples for testing)...")
        dataset = prepare_wildjailbreak_dataset(
            tokenizer=tokenizer,
            split="train",
            max_samples=10,
            max_length=512,
            seed=42
        )

        print(f"✓ Prepared {len(dataset)} examples")

        # Check that labels are properly masked
        sample = dataset[0]
        labels = sample['labels']
        masked_count = sum(1 for l in labels if l == -100)
        print(f"✓ Sample has {masked_count} masked tokens (prompt tokens)")
        print(f"✓ Sample has {len(labels) - masked_count} training tokens (response tokens)")

        assert masked_count > 0, "Should have some masked tokens!"
        assert len(labels) - masked_count > 0, "Should have some training tokens!"

        print("\n✅ Dataset preparation test PASSED")
        return True

    except Exception as e:
        print(f"\n❌ Dataset preparation test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Test setup before running full experiments")
    parser.add_argument(
        "--skip_heavy",
        action="store_true",
        help="Skip heavy tests (model loading)"
    )
    args = parser.parse_args()

    print("="*80)
    print("SETUP VALIDATION TESTS")
    print("="*80)
    print("This will test your setup before running full experiments.")
    print("Estimated time: 5-10 minutes")
    print("="*80)

    results = {}

    # Always test dataset access (lightweight)
    results['wildjailbreak'] = test_wildjailbreak_access()

    if not args.skip_heavy:
        # Test expert dropout
        results['expert_dropout'] = test_expert_dropout()

        # Test training modes
        results['training_modes'] = test_training_modes()

        # Test dataset preparation
        results['dataset_prep'] = test_dataset_preparation()

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s}: {status}")

    all_passed = all(results.values())

    if all_passed:
        print("\n" + "="*80)
        print("🎉 ALL TESTS PASSED!")
        print("="*80)
        print("\nYou're ready to run the full experiments:")
        print("  bash run_controlled_experiments.sh")
        print("\nOr run individual experiments:")
        print("  python finetune_moe_controlled.py --training_mode router ...")
        print("="*80)
    else:
        print("\n" + "="*80)
        print("⚠️  SOME TESTS FAILED")
        print("="*80)
        print("\nPlease fix the failing tests before running full experiments.")
        print("See error messages above for details.")
        print("="*80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
