"""
Visualize expert routing patterns as a node-link diagram.

This script creates a visualization showing which experts in each layer
transition to which experts in the next layer, with edge thickness
representing transition frequency across a dataset of prompts.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from collections import defaultdict
import argparse


def load_routing_data(json_path):
    """Load expert routing data from JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)


def build_transition_matrices(routing_data, use_top_k=True, separate_labels=False):
    """
    Build transition matrices counting expert-to-expert transitions.

    Args:
        routing_data: Dictionary containing routing information
        use_top_k: If True, use all top-k experts; if False, use only top-1
        separate_labels: If True, return separate matrices for harmful/harmless

    Returns:
        If separate_labels=False:
            transition_counts: Dictionary mapping (layer_from, layer_to) -> transition_matrix
            num_layers: Number of layers
            num_experts: Number of experts per layer
        If separate_labels=True:
            harmful_counts, harmless_counts, num_layers, num_experts
    """
    num_layers = routing_data['num_layers']
    num_experts = routing_data['num_experts']

    def process_results(results):
        """Process a list of results and return transition counts."""
        transition_counts = {}
        for layer_i in range(num_layers - 1):
            transition_counts[(layer_i, layer_i + 1)] = np.zeros((num_experts, num_experts))

        # Process each prompt
        for result in results:
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

        return transition_counts

    if separate_labels:
        # Build separate matrices for harmful and harmless
        harmful_results = routing_data.get('harmful_results', [])
        harmless_results = routing_data.get('harmless_results', [])

        harmful_counts = process_results(harmful_results)
        harmless_counts = process_results(harmless_results)

        return harmful_counts, harmless_counts, num_layers, num_experts
    else:
        # Build combined matrix
        all_results = []
        all_results.extend(routing_data.get('harmful_results', []))
        all_results.extend(routing_data.get('harmless_results', []))

        transition_counts = process_results(all_results)
        return transition_counts, num_layers, num_experts


def apply_edge_filtering(transition_counts, top_k_edges=None, top_percent_edges=None, min_threshold=0, verbose=False):
    """
    Filter edges based on transition frequency.

    Args:
        transition_counts: Dictionary of transition matrices
        top_k_edges: If specified, keep only top K edges per layer pair
        top_percent_edges: If specified, keep edges that carry this % of total traffic (0-100)
        min_threshold: Minimum count to keep an edge
        verbose: If True, print filtering statistics

    Returns:
        Filtered transition_counts dictionary
    """
    filtered_counts = {}

    for layer_pair, matrix in transition_counts.items():
        filtered_matrix = matrix.copy()
        original_edges = np.count_nonzero(matrix)

        if top_percent_edges is not None and 0 < top_percent_edges <= 100:
            # Keep only edges that account for X% of total traffic
            total_traffic = matrix.sum()
            if total_traffic > 0:
                # Sort edges by frequency (descending)
                flat_values = matrix.flatten()
                sorted_indices = np.argsort(flat_values)[::-1]

                # Compute cumulative percentage
                cumsum = 0
                threshold_value = 0
                target_traffic = (top_percent_edges / 100.0) * total_traffic

                for idx in sorted_indices:
                    cumsum += flat_values[idx]
                    if cumsum >= target_traffic:
                        threshold_value = flat_values[idx]
                        break

                # Keep only edges at or above threshold
                filtered_matrix[matrix < threshold_value] = 0

                if verbose:
                    kept_edges = np.count_nonzero(filtered_matrix)
                    actual_percent = 100 * filtered_matrix.sum() / total_traffic if total_traffic > 0 else 0
                    print(f"  Layer {layer_pair[0]}->{layer_pair[1]}: Kept {kept_edges}/{original_edges} edges "
                          f"({100*kept_edges/original_edges:.1f}%) carrying {actual_percent:.1f}% of traffic")

        elif top_k_edges is not None and top_k_edges > 0:
            # Keep only top K edges
            flat_values = matrix.flatten()
            if len(flat_values) > top_k_edges:
                threshold = np.partition(flat_values, -top_k_edges)[-top_k_edges]
                filtered_matrix[matrix < threshold] = 0

        if min_threshold > 0:
            # Apply minimum threshold
            filtered_matrix[filtered_matrix < min_threshold] = 0

        filtered_counts[layer_pair] = filtered_matrix

    return filtered_counts


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
    figsize=(20, 12),
    title="Expert Routing Visualization",
    harmful_counts=None,
    harmless_counts=None,
    exclude_last_n_layers=3
):
    """
    Create a node-link diagram showing expert routing patterns.

    Args:
        transition_counts: Dictionary of transition matrices (used if harmful/harmless not provided)
        num_layers: Number of layers in the model
        num_experts: Number of experts per layer
        output_path: Path to save the figure (if None, displays instead)
        x_per_layer: Horizontal distance between layers
        max_line_width: Maximum line width for edges
        min_line_width: Minimum line width for edges
        edge_alpha: Transparency of edges
        layer_height_exponent: Controls vertical spacing of nodes
        figsize: Figure size as (width, height)
        title: Title for the plot
        harmful_counts: Separate transition matrix for harmful prompts
        harmless_counts: Separate transition matrix for harmless prompts
        exclude_last_n_layers: Number of non-expert layers to exclude from end
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Determine which layers to show
    effective_num_layers = num_layers - exclude_last_n_layers

    # Determine if we're using separate harmful/harmless visualization
    use_separate_colors = harmful_counts is not None and harmless_counts is not None

    if use_separate_colors:
        # Normalize each dataset independently so they each use the full color range
        max_harmful = max(
            matrix.max() for layer_pair, matrix in harmful_counts.items()
            if layer_pair[0] < effective_num_layers - 1 and matrix.max() > 0
        ) if harmful_counts else 0
        max_harmless = max(
            matrix.max() for layer_pair, matrix in harmless_counts.items()
            if layer_pair[0] < effective_num_layers - 1 and matrix.max() > 0
        ) if harmless_counts else 0
        # Don't combine them - use separate scales
        max_transitions = None  # Not used in separate color mode
    else:
        # Find maximum transition count for normalization
        max_transitions = max(
            matrix.max() for layer_pair, matrix in transition_counts.items()
            if layer_pair[0] < effective_num_layers - 1 and matrix.max() > 0
        )

    if not use_separate_colors and max_transitions == 0:
        print("Warning: No transitions found in the data")
        return

    def node_vertical_position(i_node, n_nodes):
        """Calculate vertical position for a node."""
        y_offset = i_node - 0.5 * (n_nodes - 1)
        if n_nodes > 1:
            return y_offset / (n_nodes - 1) ** layer_height_exponent
        else:
            return 0

    def edge_width(count, max_count):
        """Calculate edge width based on transition count."""
        if count <= 0 or max_count == 0:
            return 0
        normalized = count / max_count
        return min_line_width + (max_line_width - min_line_width) * normalized

    # Draw edges between layers
    for layer_i in range(effective_num_layers - 1):
        x_from = layer_i * x_per_layer
        x_to = (layer_i + 1) * x_per_layer

        if use_separate_colors:
            # Draw harmful edges in red (normalized independently)
            harmful_matrix = harmful_counts.get((layer_i, layer_i + 1))
            if harmful_matrix is not None and max_harmful > 0:
                for expert_from in range(num_experts):
                    for expert_to in range(num_experts):
                        count = harmful_matrix[expert_from, expert_to]
                        if count <= 0:
                            continue

                        width = edge_width(count, max_harmful)
                        y_from = node_vertical_position(expert_from, num_experts)
                        y_to = node_vertical_position(expert_to, num_experts)

                        # Red for harmful - normalize within harmful dataset only
                        intensity = min(1.0, count / max_harmful)
                        color = (intensity, 0, 0)  # Red

                        ax.plot(
                            [x_from, x_to],
                            [y_from, y_to],
                            lw=width,
                            color=color,
                            alpha=edge_alpha,
                            zorder=1
                        )

            # Draw harmless edges in blue (normalized independently)
            harmless_matrix = harmless_counts.get((layer_i, layer_i + 1))
            if harmless_matrix is not None and max_harmless > 0:
                for expert_from in range(num_experts):
                    for expert_to in range(num_experts):
                        count = harmless_matrix[expert_from, expert_to]
                        if count <= 0:
                            continue

                        width = edge_width(count, max_harmless)
                        y_from = node_vertical_position(expert_from, num_experts)
                        y_to = node_vertical_position(expert_to, num_experts)

                        # Blue for harmless - normalize within harmless dataset only
                        intensity = min(1.0, count / max_harmless)
                        color = (0, 0, intensity)  # Blue

                        ax.plot(
                            [x_from, x_to],
                            [y_from, y_to],
                            lw=width,
                            color=color,
                            alpha=edge_alpha,
                            zorder=1
                        )

        else:
            # Single color mode
            transition_matrix = transition_counts.get((layer_i, layer_i + 1))
            if transition_matrix is None:
                continue

            # Draw edges
            for expert_from in range(num_experts):
                for expert_to in range(num_experts):
                    count = transition_matrix[expert_from, expert_to]
                    if count <= 0:
                        continue

                    width = edge_width(count, max_transitions)
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
    for layer_i in range(effective_num_layers):
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
    for layer_i in range(effective_num_layers):
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

    # Add legend or colorbar depending on mode
    if use_separate_colors:
        # Create custom legend for harmful/harmless
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='Harmful prompts'),
            Line2D([0], [0], color='blue', lw=2, label='Harmless prompts')
        ]
        legend = ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        # Add note about independent normalization
        ax.text(
            0.98, 0.02,
            'Note: Each dataset processed and visualized independently\n(darker/thicker = more frequent within that dataset)',
            transform=ax.transAxes,
            ha='right',
            va='bottom',
            fontsize=9,
            style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3)
        )
    else:
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


def print_statistics(transition_counts, num_layers, num_experts, label="", exclude_last_n_layers=3):
    """Print statistics about the expert routing patterns."""
    effective_num_layers = num_layers - exclude_last_n_layers

    print("\n" + "="*60)
    print(f"Expert Routing Statistics{' - ' + label if label else ''}")
    print("="*60)

    for layer_i in range(effective_num_layers - 1):
        matrix = transition_counts.get((layer_i, layer_i + 1))
        if matrix is None:
            continue

        total_transitions = matrix.sum()
        if total_transitions == 0:
            continue

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
        type=lambda x: str(x).lower() == 'true',
        default=True,
        help='Use all top-k experts instead of just top-1 (default: True)'
    )
    parser.add_argument(
        '--separate-colors',
        action='store_true',
        help='Use different colors for harmful (red) and harmless (blue) paths'
    )
    parser.add_argument(
        '--min-threshold',
        type=int,
        default=0,
        help='Minimum transition count to display an edge'
    )
    parser.add_argument(
        '--top-k-edges',
        type=int,
        default=None,
        help='Keep only top K edges per layer pair'
    )
    parser.add_argument(
        '--top-percent-edges',
        type=float,
        default=None,
        help='Keep edges that carry this percentage of total traffic (0-100, e.g., 80 for top 80%%)'
    )
    parser.add_argument(
        '--max-line-width',
        type=float,
        default=3.0,
        help='Maximum line width for edges'
    )
    parser.add_argument(
        '--exclude-last-n-layers',
        type=int,
        default=3,
        help='Number of non-expert layers to exclude from end (default: 3)'
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

    print(f"Using top-k experts: {args.use_top_k}")
    print(f"Separate colors for harmful/harmless: {args.separate_colors}")
    print(f"Excluding last {args.exclude_last_n_layers} layers")

    # Build transition matrices
    print("Building transition matrices...")
    if args.separate_colors:
        harmful_counts, harmless_counts, num_layers, num_experts = build_transition_matrices(
            routing_data,
            use_top_k=args.use_top_k,
            separate_labels=True
        )

        # Apply filtering to each dataset independently
        if args.top_percent_edges:
            print(f"\nFiltering to keep edges carrying {args.top_percent_edges}% of traffic:")
            print("\nHarmful dataset:")
        harmful_counts = apply_edge_filtering(
            harmful_counts,
            top_k_edges=args.top_k_edges,
            top_percent_edges=args.top_percent_edges,
            min_threshold=args.min_threshold,
            verbose=bool(args.top_percent_edges)
        )
        if args.top_percent_edges:
            print("\nHarmless dataset:")
        harmless_counts = apply_edge_filtering(
            harmless_counts,
            top_k_edges=args.top_k_edges,
            top_percent_edges=args.top_percent_edges,
            min_threshold=args.min_threshold,
            verbose=bool(args.top_percent_edges)
        )

        # Print statistics
        print_statistics(harmful_counts, num_layers, num_experts, "Harmful", args.exclude_last_n_layers)
        print_statistics(harmless_counts, num_layers, num_experts, "Harmless", args.exclude_last_n_layers)

        # Create visualization
        print("\nCreating visualization...")
        visualize_expert_routing(
            None,
            num_layers,
            num_experts,
            output_path=args.output,
            max_line_width=args.max_line_width,
            figsize=tuple(args.figsize),
            title="Expert Routing: Harmful (Red) vs Harmless (Blue)",
            harmful_counts=harmful_counts,
            harmless_counts=harmless_counts,
            exclude_last_n_layers=args.exclude_last_n_layers
        )
    else:
        transition_counts, num_layers, num_experts = build_transition_matrices(
            routing_data,
            use_top_k=args.use_top_k,
            separate_labels=False
        )

        # Apply filtering
        transition_counts = apply_edge_filtering(
            transition_counts,
            top_k_edges=args.top_k_edges,
            top_percent_edges=args.top_percent_edges,
            min_threshold=args.min_threshold
        )

        # Print statistics
        print_statistics(transition_counts, num_layers, num_experts, "", args.exclude_last_n_layers)

        # Create visualization
        print("\nCreating visualization...")
        visualize_expert_routing(
            transition_counts,
            num_layers,
            num_experts,
            output_path=args.output,
            max_line_width=args.max_line_width,
            figsize=tuple(args.figsize),
            title="Expert Routing Visualization",
            exclude_last_n_layers=args.exclude_last_n_layers
        )

    print("\nDone!")


if __name__ == "__main__":
    main()
