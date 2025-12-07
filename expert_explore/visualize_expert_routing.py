"""
Visualize expert routing patterns as a node-link diagram.

This script creates a visualization showing which experts in each layer
transition to which experts in the next layer, with edge thickness
representing transition frequency across a dataset of prompts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import argparse


def load_routing_data(json_path):
    """Load expert routing data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def build_transition_matrices(routing_data, use_top_k=True, filter_by_label=None):
    """
    Build transition matrices counting expert-to-expert transitions.

    Args:
        routing_data: Dictionary containing routing information
        use_top_k: If True, use all top-k experts; if False, use only top-1
        filter_by_label: If specified ('harmful' or 'harmless'), only use those prompts

    Returns:
        Dictionary mapping (layer_from, layer_to) -> transition_matrix
        where transition_matrix[i, j] = count of transitions from expert i to expert j
    """
    num_layers = routing_data['num_layers']
    num_experts = routing_data['num_experts']

    # Initialize transition matrices for each layer pair
    # transition_counts[(layer_i, layer_i+1)][expert_from, expert_to] = count
    transition_counts = {}
    for layer_i in range(num_layers - 1):
        transition_counts[(layer_i, layer_i + 1)] = np.zeros((num_experts, num_experts))

    # Combine harmful and harmless results
    all_results = []
    if filter_by_label == 'harmful' or filter_by_label is None:
        all_results.extend(routing_data.get('harmful_results', []))
    if filter_by_label == 'harmless' or filter_by_label is None:
        all_results.extend(routing_data.get('harmless_results', []))

    # Process each prompt
    for result in all_results:
        layer_routing = result['layer_routing']

        # For each consecutive layer pair
        for layer_i in range(num_layers - 1):
            layer_from_key = f"layer_{layer_i}"
            layer_to_key = f"layer_{layer_i + 1}"

            if layer_from_key not in layer_routing or layer_to_key not in layer_routing:
                continue

            # Get expert selections for both layers
            if use_top_k:
                # Use all top-k experts (usually k=4)
                experts_from = layer_routing[layer_from_key]['top_k_experts']
                experts_to = layer_routing[layer_to_key]['top_k_experts']
            else:
                # Use only the top expert
                experts_from = [[e] for e in layer_routing[layer_from_key]['top_expert']]
                experts_to = [[e] for e in layer_routing[layer_to_key]['top_expert']]

            # For each token position
            for token_experts_from, token_experts_to in zip(experts_from, experts_to):
                # Count all transitions between active experts
                for expert_from in token_experts_from:
                    for expert_to in token_experts_to:
                        transition_counts[(layer_i, layer_i + 1)][expert_from, expert_to] += 1

    return transition_counts, num_layers, num_experts


def visualize_expert_routing(
    transition_counts,
    num_layers,
    num_experts,
    output_path=None,
    x_per_layer=1.5,
    max_line_width=3.0,
    min_line_width=0.1,
    edge_alpha=0.6,
    layer_height_exponent=0.5,
    min_edge_threshold=0,
    figsize=(20, 12),
    title="Expert Routing Visualization"
):
    """
    Create a node-link diagram showing expert routing patterns.

    Args:
        transition_counts: Dictionary of transition matrices
        num_layers: Number of layers in the model
        num_experts: Number of experts per layer
        output_path: Path to save the figure (if None, displays instead)
        x_per_layer: Horizontal distance between layers
        max_line_width: Maximum line width for edges
        min_line_width: Minimum line width for edges
        edge_alpha: Transparency of edges
        layer_height_exponent: Controls vertical spacing of nodes
        min_edge_threshold: Only draw edges with count >= this threshold
        figsize: Figure size as (width, height)
        title: Title for the plot
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Find maximum transition count for normalization
    max_transitions = max(
        matrix.max() for matrix in transition_counts.values()
        if matrix.max() > 0
    )

    if max_transitions == 0:
        print("Warning: No transitions found in the data")
        return

    def node_vertical_position(i_node, n_nodes):
        """Calculate vertical position for a node."""
        y_offset = i_node - 0.5 * (n_nodes - 1)
        if n_nodes > 1:
            return y_offset / (n_nodes - 1) ** layer_height_exponent
        else:
            return 0

    def edge_width(count):
        """Calculate edge width based on transition count."""
        if count <= 0:
            return 0
        normalized = count / max_transitions
        return min_line_width + (max_line_width - min_line_width) * normalized

    # Draw edges between layers
    for layer_i in range(num_layers - 1):
        x_from = layer_i * x_per_layer
        x_to = (layer_i + 1) * x_per_layer

        transition_matrix = transition_counts.get((layer_i, layer_i + 1))
        if transition_matrix is None:
            continue

        # Draw edges
        for expert_from in range(num_experts):
            for expert_to in range(num_experts):
                count = transition_matrix[expert_from, expert_to]

                if count < min_edge_threshold:
                    continue

                width = edge_width(count)
                if width <= 0:
                    continue

                y_from = node_vertical_position(expert_from, num_experts)
                y_to = node_vertical_position(expert_to, num_experts)

                # Color based on frequency (darker = more frequent)
                color_intensity = count / max_transitions
                color = plt.cm.viridis(color_intensity)

                ax.plot(
                    [x_from, x_to],
                    [y_from, y_to],
                    lw=width,
                    color=color,
                    alpha=edge_alpha,
                    zorder=1
                )

    # Draw nodes
    for layer_i in range(num_layers):
        x = layer_i * x_per_layer
        for expert_i in range(num_experts):
            y = node_vertical_position(expert_i, num_experts)
            ax.scatter(
                x, y,
                s=30,
                c='black',
                marker='o',
                zorder=10,
                edgecolors='white',
                linewidths=0.5
            )

    # Add layer labels
    for layer_i in range(num_layers):
        x = layer_i * x_per_layer
        y_max = node_vertical_position(num_experts - 1, num_experts)
        ax.text(
            x, y_max + 0.15,
            f'Layer {layer_i}',
            ha='center',
            va='bottom',
            fontsize=10,
            fontweight='bold'
        )

    # Add colorbar
    sm = plt.cm.ScalarMappable(
        cmap=plt.cm.viridis,
        norm=plt.Normalize(vmin=0, vmax=max_transitions)
    )
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Transition Frequency', fontsize=12)

    # Clean up axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_path}")
    else:
        plt.show()

    return fig, ax


def print_statistics(transition_counts, num_layers, num_experts):
    """Print statistics about the expert routing patterns."""
    print("\n" + "="*60)
    print("Expert Routing Statistics")
    print("="*60)

    for layer_i in range(num_layers - 1):
        matrix = transition_counts.get((layer_i, layer_i + 1))
        if matrix is None:
            continue

        total_transitions = matrix.sum()
        num_nonzero = np.count_nonzero(matrix)
        max_transition = matrix.max()

        # Find most common transition
        max_idx = np.unravel_index(matrix.argmax(), matrix.shape)

        print(f"\nLayer {layer_i} -> Layer {layer_i + 1}:")
        print(f"  Total transitions: {int(total_transitions)}")
        print(f"  Active pathways: {num_nonzero} / {num_experts * num_experts} ({100 * num_nonzero / (num_experts * num_experts):.1f}%)")
        print(f"  Most common transition: Expert {max_idx[0]} -> Expert {max_idx[1]} ({int(max_transition)} times)")

        # Calculate concentration (what % of transitions go through top 10% of pathways)
        sorted_transitions = np.sort(matrix.flatten())[::-1]
        top_10_percent_idx = max(1, int(0.1 * len(sorted_transitions)))
        top_10_percent_sum = sorted_transitions[:top_10_percent_idx].sum()
        concentration = 100 * top_10_percent_sum / total_transitions
        print(f"  Concentration: {concentration:.1f}% of transitions use top 10% of pathways")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize expert routing patterns from MoE models"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='/media/volume/align_2_stg/alignment/expert_routing_analysis/expert_routing_data.json',
        help='Path to expert routing JSON data'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Path to save visualization (default: display instead)'
    )
    parser.add_argument(
        '--use-top-k',
        action='store_true',
        default=True,
        help='Use all top-k experts instead of just top-1'
    )
    parser.add_argument(
        '--filter-label',
        type=str,
        choices=['harmful', 'harmless', None],
        default=None,
        help='Filter by prompt label (default: use all)'
    )
    parser.add_argument(
        '--min-threshold',
        type=int,
        default=0,
        help='Minimum transition count to display an edge'
    )
    parser.add_argument(
        '--max-line-width',
        type=float,
        default=3.0,
        help='Maximum line width for edges'
    )
    parser.add_argument(
        '--figsize',
        type=int,
        nargs=2,
        default=[20, 12],
        help='Figure size as width height'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading routing data from {args.input}...")
    routing_data = load_routing_data(args.input)

    # Build transition matrices
    print("Building transition matrices...")
    transition_counts, num_layers, num_experts = build_transition_matrices(
        routing_data,
        use_top_k=args.use_top_k,
        filter_by_label=args.filter_label
    )

    # Print statistics
    print_statistics(transition_counts, num_layers, num_experts)

    # Create visualization
    print("\nCreating visualization...")
    title = "Expert Routing Visualization"
    if args.filter_label:
        title += f" ({args.filter_label.capitalize()} Prompts)"

    visualize_expert_routing(
        transition_counts,
        num_layers,
        num_experts,
        output_path=args.output,
        min_edge_threshold=args.min_threshold,
        max_line_width=args.max_line_width,
        figsize=tuple(args.figsize),
        title=title
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
