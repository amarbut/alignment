#!/bin/bash

# Script to compile evaluation results across all pipeline runs
# Usage: ./compile_results.sh [runs_directory]

RUNS_DIR="${1:-pipeline/runs}"

echo "Compiling evaluation results from: $RUNS_DIR"
echo "=================================================================="
echo ""

# Function to extract score from JSON file
extract_score() {
    local file=$1
    local key=$2

    if [ ! -f "$file" ]; then
        echo "N/A"
        return
    fi

    # Extract the score
    score=$(jq -r ".$key // \"N/A\"" "$file" 2>/dev/null)

    if [ "$score" = "N/A" ] || [ -z "$score" ] || [ "$score" = "null" ]; then
        echo "N/A"
    else
        printf "%.3f" "$score"
    fi
}

# Process newer subspace pipeline results (with k/a/t structure)
find "$RUNS_DIR" -type f -name "*_evaluations.json" -path "*/k*/a*/t*/*" | while read -r eval_file; do
    # Extract path components
    # Path: pipeline/runs/{model}/{method}/completions/k{topk}/a{coeff}/t{tau}/{dataset}_{intervention}_evaluations.json

    relative_path="${eval_file#$RUNS_DIR/}"
    model=$(echo "$relative_path" | cut -d'/' -f1)
    method=$(echo "$relative_path" | cut -d'/' -f2)

    # Extract k, a, t from path (match only directory names starting with these letters)
    k=$(echo "$relative_path" | grep -oP '/k\K[^/]+')
    a=$(echo "$relative_path" | grep -oP '/a\K[^/]+')
    t=$(echo "$relative_path" | grep -oP '/t\K[^/]+')

    # Get dataset and intervention from filename
    filename=$(basename "$eval_file")
    dataset_intervention=$(echo "$filename" | sed 's/_evaluations.json//')

    # Group by model/method/k/a/t - only print header once per combination
    cache_key="${model}|${method}|${k}|${a}|${t}"
    if [ ! -f "/tmp/compiled_${cache_key//|/_}" ]; then
        touch "/tmp/compiled_${cache_key//|/_}"

        echo "Model: $model | Method: $method | k=$k, a=$a, t=$t"
        echo "--------------------------------------------------"

        # Get directory containing all evaluation files for this combo
        eval_dir=$(dirname "$eval_file")

        # Extract scores (prefer llamaguard2, fallback to substring_matching)
        harmless_baseline=$(extract_score "$eval_dir/harmless_baseline_evaluations.json" "substring_matching_success_rate")
        harmless_actadd=$(extract_score "$eval_dir/harmless_actadd_evaluations.json" "substring_matching_success_rate")

        jbb_baseline_lg=$(extract_score "$eval_dir/jailbreakbench_baseline_evaluations.json" "llamaguard2_success_rate")
        jbb_baseline_sm=$(extract_score "$eval_dir/jailbreakbench_baseline_evaluations.json" "substring_matching_success_rate")
        jbb_baseline="${jbb_baseline_lg}"
        if [ "$jbb_baseline" = "N/A" ]; then
            jbb_baseline="${jbb_baseline_sm}"
        fi

        jbb_ablation_lg=$(extract_score "$eval_dir/jailbreakbench_ablation_evaluations.json" "llamaguard2_success_rate")
        jbb_ablation_sm=$(extract_score "$eval_dir/jailbreakbench_ablation_evaluations.json" "substring_matching_success_rate")
        jbb_ablation="${jbb_ablation_lg}"
        if [ "$jbb_ablation" = "N/A" ]; then
            jbb_ablation="${jbb_ablation_sm}"
        fi

        jbb_actadd_lg=$(extract_score "$eval_dir/jailbreakbench_actadd_evaluations.json" "llamaguard2_success_rate")
        jbb_actadd_sm=$(extract_score "$eval_dir/jailbreakbench_actadd_evaluations.json" "substring_matching_success_rate")
        jbb_actadd="${jbb_actadd_lg}"
        if [ "$jbb_actadd" = "N/A" ]; then
            jbb_actadd="${jbb_actadd_sm}"
        fi

        # Print results
        printf "%-25s %s\n" "Harmless (baseline):" "$harmless_baseline"
        printf "%-25s %s\n" "Harmless (actadd):" "$harmless_actadd"
        printf "%-25s %s\n" "JailbreakBench (baseline):" "$jbb_baseline"
        printf "%-25s %s\n" "JailbreakBench (ablation):" "$jbb_ablation"
        printf "%-25s %s\n" "JailbreakBench (actadd):" "$jbb_actadd"
        echo ""
    fi
done

# Clean up temp files
rm -f /tmp/compiled_*

# Process older pipeline results (direct in completions/)
find "$RUNS_DIR" -type f -name "*_evaluations.json" ! -path "*/k*/*" -path "*/completions/*" | head -1 | while read -r eval_file; do
    if [ -n "$eval_file" ]; then
        relative_path="${eval_file#$RUNS_DIR/}"
        model=$(echo "$relative_path" | cut -d'/' -f1)
        eval_dir=$(dirname "$eval_file")

        echo "Model: $model | Method: arditi (original)"
        echo "--------------------------------------------------"

        harmless_baseline=$(extract_score "$eval_dir/harmless_baseline_evaluations.json" "substring_matching_success_rate")
        harmless_actadd=$(extract_score "$eval_dir/harmless_actadd_evaluations.json" "substring_matching_success_rate")

        jbb_baseline_lg=$(extract_score "$eval_dir/jailbreakbench_baseline_evaluations.json" "llamaguard2_success_rate")
        jbb_baseline_sm=$(extract_score "$eval_dir/jailbreakbench_baseline_evaluations.json" "substring_matching_success_rate")
        jbb_baseline="${jbb_baseline_lg}"
        if [ "$jbb_baseline" = "N/A" ]; then
            jbb_baseline="${jbb_baseline_sm}"
        fi

        jbb_ablation_lg=$(extract_score "$eval_dir/jailbreakbench_ablation_evaluations.json" "llamaguard2_success_rate")
        jbb_ablation_sm=$(extract_score "$eval_dir/jailbreakbench_ablation_evaluations.json" "substring_matching_success_rate")
        jbb_ablation="${jbb_ablation_lg}"
        if [ "$jbb_ablation" = "N/A" ]; then
            jbb_ablation="${jbb_ablation_sm}"
        fi

        jbb_actadd_lg=$(extract_score "$eval_dir/jailbreakbench_actadd_evaluations.json" "llamaguard2_success_rate")
        jbb_actadd_sm=$(extract_score "$eval_dir/jailbreakbench_actadd_evaluations.json" "substring_matching_success_rate")
        jbb_actadd="${jbb_actadd_lg}"
        if [ "$jbb_actadd" = "N/A" ]; then
            jbb_actadd="${jbb_actadd_sm}"
        fi

        printf "%-25s %s\n" "Harmless (baseline):" "$harmless_baseline"
        printf "%-25s %s\n" "Harmless (actadd):" "$harmless_actadd"
        printf "%-25s %s\n" "JailbreakBench (baseline):" "$jbb_baseline"
        printf "%-25s %s\n" "JailbreakBench (ablation):" "$jbb_ablation"
        printf "%-25s %s\n" "JailbreakBench (actadd):" "$jbb_actadd"
        echo ""
    fi
done

echo "=================================================================="
echo "Compilation complete!"
