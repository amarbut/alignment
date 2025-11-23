#!/bin/bash

# Script to show detailed results for a specific model/method combination
# Usage: ./show_results.sh MODEL METHOD K A T

if [ $# -ne 5 ]; then
    echo "Usage: $0 MODEL METHOD K A T"
    echo "Example: $0 Meta-Llama-3.1-8B-Instruct cpca_arditi 1 1.0 1.0"
    exit 1
fi

MODEL=$1
METHOD=$2
K=$3
A=$4
T=$5

RUNS_DIR="pipeline/runs"
RESULTS_DIR="$RUNS_DIR/$MODEL/$METHOD/completions/k$K/a$A/t$T"

echo "=================================================================="
echo "Results for: $MODEL | $METHOD | k=$K, a=$A, t=$T"
echo "=================================================================="
echo ""

if [ ! -d "$RESULTS_DIR" ]; then
    echo "ERROR: Results directory not found: $RESULTS_DIR"
    exit 1
fi

# Function to extract and display scores
show_scores() {
    local eval_file=$1
    local label=$2

    if [ ! -f "$eval_file" ]; then
        echo "$label: N/A (file not found)"
        return
    fi

    substring_score=$(jq -r '.substring_matching_success_rate // "N/A"' "$eval_file" 2>/dev/null)
    llamaguard_score=$(jq -r '.llamaguard2_success_rate // "N/A"' "$eval_file" 2>/dev/null)

    printf "%-30s substring: %.3f, llamaguard: %.3f\n" "$label:" "$substring_score" "$llamaguard_score" 2>/dev/null || \
    printf "%-30s substring: %s, llamaguard: %s\n" "$label:" "$substring_score" "$llamaguard_score"
}

# Display scores
echo "=== EVALUATION SCORES ==="
show_scores "$RESULTS_DIR/harmless_baseline_evaluations.json" "Harmless (baseline)"
show_scores "$RESULTS_DIR/harmless_actadd_evaluations.json" "Harmless (actadd)"
show_scores "$RESULTS_DIR/jailbreakbench_baseline_evaluations.json" "JailbreakBench (baseline)"
show_scores "$RESULTS_DIR/jailbreakbench_ablation_evaluations.json" "JailbreakBench (ablation)"
show_scores "$RESULTS_DIR/jailbreakbench_actadd_evaluations.json" "JailbreakBench (actadd)"
echo ""

# Function to show first N completions
show_completions() {
    local completion_file=$1
    local label=$2
    local n=${3:-2}

    if [ ! -f "$completion_file" ]; then
        echo "$label: N/A (file not found)"
        return
    fi

    echo "=== $label ==="
    echo ""

    for i in $(seq 0 $((n-1))); do
        prompt=$(jq -r ".[$i].prompt // \"N/A\"" "$completion_file" 2>/dev/null)
        response=$(jq -r ".[$i].response // \"N/A\"" "$completion_file" 2>/dev/null)
        category=$(jq -r ".[$i].category // \"N/A\"" "$completion_file" 2>/dev/null)

        if [ "$prompt" = "N/A" ] || [ "$prompt" = "null" ]; then
            break
        fi

        echo "--- Example $((i+1)) ---"
        if [ "$category" != "N/A" ] && [ "$category" != "null" ]; then
            echo "Category: $category"
        fi
        echo "Prompt: $prompt"
        echo ""
        echo "Response: $response"
        echo ""
    done
}

# Show sample completions for interventions only (not baseline)
echo ""
show_completions "$RESULTS_DIR/harmless_actadd_completions.json" "HARMLESS - ACTADD INTERVENTION" 2
show_completions "$RESULTS_DIR/jailbreakbench_ablation_completions.json" "JAILBREAKBENCH - ABLATION INTERVENTION" 2
show_completions "$RESULTS_DIR/jailbreakbench_actadd_completions.json" "JAILBREAKBENCH - ACTADD INTERVENTION" 2

echo "=================================================================="
echo "End of results"
echo "=================================================================="
