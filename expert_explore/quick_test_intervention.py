"""
Quick test script to verify expert intervention pipeline works.

This runs a minimal test with just a few examples to verify:
1. Model loads correctly
2. Interventions can be applied and removed
3. Completions can be generated
4. No crashes occur

Run this before running the full pipeline.
"""

import sys
sys.path.insert(0, '/media/volume/align_2_stg/alignment')

from expert_explore.run_expert_intervention_pipeline import run_expert_intervention_pipeline

if __name__ == "__main__":
    print("Running quick test of expert intervention pipeline...")
    print("This will test a few interventions with only 5 examples each.\n")

    # Run with minimal settings
    run_expert_intervention_pipeline(
        model_path='openai/gpt-oss-20b',
        n_test=5,  # Only 5 examples for quick test
        intervention_types=['suppress', 'force'],  # Test both types
        expert_types=['l10e5'],  # Test just one expert config
        skip_baseline=False  # Include baseline
    )

    print("\n" + "=" * 80)
    print("Quick test completed successfully!")
    print("You can now run the full pipeline with more examples.")
    print("=" * 80)
