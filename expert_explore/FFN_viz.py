"""Visualizing feedforward models."""

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.cluster import hierarchy


def nodelink_diagram_for_feedforward(
    model,
    cmap="PiYG",
    x_per_layer=0.2,
    dx_layer=0.0,
    max_line_width=3.0,
    min_line_width=0.5,
    with_spline_links=False,
    layer_height_exponent=0.5,
    neuron_plot_kwargs=None,
    ax=None,
    with_colorbar=True,
):
    """Make a node-link diagram of a feedforward neural network.

    Args:
        model: a PyTorch model
        cmap: a colormap for the weights
        x_per_layer: the horizontal distance between nodes in the same layer
        dx_layer: the horizontal distance between nodes in different layers
        max_line_width: the maximum line width
        min_line_width: the minimum line width
        with_spline_links: if True, draw links with a spline instead of a line
        layer_height_exponent: the exponent for the height of a layer, between 0.0 and 1.0
            if 0.0, layer height is proportional to the number of its neurons
            if 1.0, layer height is constant over all layers
        neuron_plot_kwargs: keyword arguments for the plot of each neuron
        ax: an optional matplotlib axis"""
    if ax is None:
        _, ax = plt.subplots()
    if neuron_plot_kwargs is None:
        neuron_plot_kwargs = {
            "edgecolor": "k",
            "marker": "o",
            "s": 1.0,
            "facecolor": "k",
        }

    max_abs_weight = max(
        get_layer_weight_matrix(layer).abs().max().max()
        for layer in model.children()
        if hasattr(layer, "weight")
    )

    def node_vertical_position(i_node, n_nodes):
        y_offset = i_node - 0.5 * (n_nodes - 1)
        if n_nodes > 1:
            return y_offset / (n_nodes - 1) ** layer_height_exponent
        else:
            return 0

    x_layer = 0

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    weight_norm = matplotlib.colors.Normalize(vmin=-max_abs_weight, vmax=max_abs_weight)

    for i_layer, (layer_name, layer) in enumerate(model.named_children()):

        layer_class = layer.__class__.__name__
        if layer_class == "Linear":

            weight_df = get_layer_weight_matrix(layer).T
            n_in = layer.in_features
            n_out = layer.out_features

            def edge_width(weight):
                return (
                    min_line_width
                    + (max_line_width - min_line_width) * abs(weight) / max_abs_weight
                )

            # edges
            for i_in in range(n_in):
                for i_out in range(n_out):
                    weight = weight_df.iloc[i_in, i_out]
                    weight_color = cmap(weight_norm(weight))

                    x_from_to = [x_layer + dx_layer, x_layer + x_per_layer - dx_layer]
                    y_from_to = [
                        node_vertical_position(i_in, n_in),
                        node_vertical_position(i_out, n_out),
                    ]
                    if with_spline_links:
                        x_line, y_line = cubic_spline(x_from_to, y_from_to)
                    else:
                        x_line, y_line = x_from_to, y_from_to
                    ax.plot(
                        x_line,
                        y_line,
                        lw=edge_width(weight),
                        color=weight_color,
                    )

            # nodes
            if i_layer == 0:
                for i_in in range(n_in):
                    y_in = node_vertical_position(i_in, n_in)
                    ax.scatter(x_layer, y_in, **neuron_plot_kwargs, zorder=10)
            for i_out in range(n_out):
                y_out = node_vertical_position(i_out, n_out)
                ax.scatter(
                    x_layer + x_per_layer, y_out, **neuron_plot_kwargs, zorder=10
                )

            x_layer += x_per_layer

    if with_colorbar:
        plt.colorbar(
            matplotlib.cm.ScalarMappable(
                cmap=cmap,
                norm=matplotlib.colors.Normalize(
                    vmin=-max_abs_weight, vmax=max_abs_weight
                ),
            ),
            ax=ax,
            fraction=0.01,
        )
    clean_axes(ax)


def draw_horizontal_arrow(ax, x_start, y_start, d_x, arrow_kwargs=None):
    """Draw an arrow from (x_start, y_start) to (x_start + d_x, y_start)"""
    if arrow_kwargs is None:
        arrow_kwargs = {
            "head_width": 0.03,
            "head_length": 0.03,
            "fc": "dodgerblue",
            "ec": "dodgerblue",
            "lw": 0.5,
        }
    ax.arrow(
        x_start,
        y_start,
        d_x,
        0,
        head_starts_at_zero=False,
        **arrow_kwargs,
    )


def get_layer_label(layer, layer_name=None, add_layer_name=False):
    """Get a label for a layer"""
    layer_class = layer.__class__.__name__
    layer_label = r"$\bf{" + layer_class + "}$\n"
    if layer_class == "Linear":
        layer_label += f"{layer.in_features} x {layer.out_features}\n"
    if add_layer_name and layer_name is not None:
        layer_label = layer_name + "\n" + layer_label
    return layer_label


def plot_sequential_list_of_layers(model, rectangle_kwargs=None, arrow_kwargs=None):
    """Illustrate a sequential model as a list of rectangular layers"""

    if rectangle_kwargs is None:
        rectangle_kwargs = {
            "linewidth": 1.0,
            "edgecolor": "dodgerblue",
            "facecolor": "none",
        }

    _, ax = plt.subplots(figsize=(5, 3))
    for i_layer, (layer_name, layer) in enumerate(model.named_children()):

        layer_class = layer.__class__.__name__
        layer_label = get_layer_label(layer, layer_name)

        if layer_class == "Linear":
            height = 2.0
            width = 0.8
        elif layer_class == "ReLU":
            height = 0.5
            width = 0.5
        else:
            raise NotImplementedError(f"Unknown layer: {layer_class}")

        ax.text(i_layer, height * 0.5, layer_label, ha="center", va="bottom")

        if i_layer == 0:
            draw_horizontal_arrow(
                ax, -0.5 * width - 0.3, 0, 0.2, arrow_kwargs=arrow_kwargs
            )
        draw_horizontal_arrow(
            ax, i_layer + 0.5 * width + 0.05, 0, 0.2, arrow_kwargs=arrow_kwargs
        )

        rectangle = matplotlib.patches.Rectangle(
            (i_layer - 0.5 * width, -0.5 * height), width, height, **rectangle_kwargs
        )
        ax.add_patch(rectangle)
        ax.set_xlim(-1.0, i_layer + 1.0)

    clean_axes(ax)


def plot_relu(ax, kwargs=None, delta=0.2):
    """Plot a symbol for the ReLU activation function."""
    if kwargs is None:
        kwargs = {"color": "black"}
    ax.plot([-1, 0, 1], [0, 0, 1], **kwargs)
    ax.set_xlim(-1 - delta, 1 + delta)
    ax.set_ylim(-delta, 1 + delta)
    ax.set_aspect("equal")


def diagram_with_inset_heatmaps(
    model, ax=None, length_mapping=None, x_space=0.2, cmap="PiYG", seriated=True
):
    """Illustrate a feedforward model with a diagram including heatmaps for weights and biases

    Args:
        model: a PyTorch model
        ax: an optional matplotlib axis
        length_mapping: a function that takes the number of rows/columns in the coefficient matrix
            and returns the length of the corresponding matrix side in inches
        x_space: the space between the layers in inches
        cmap: a colormap
        seriated: whether the heatmaps should be seriated
    """

    if length_mapping is None:

        def length_mapping(n_elements):
            return n_elements**0.9

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    max_n_in = max(
        layer.weight.shape[1] for layer in model.children() if hasattr(layer, "weight")
    )
    max_n_out = max(
        layer.weight.shape[0] for layer in model.children() if hasattr(layer, "weight")
    )
    max_n = max(max_n_in, max_n_out)
    max_abs_weight = get_max_absolute_weight(model)

    def normalized_length_mapping(n_elements):
        return length_mapping(n_elements) / length_mapping(max_n)

    heatmap_kwargs = {
        "cmap": cmap,
        "vmin": -max_abs_weight,
        "vmax": max_abs_weight,
        "with_colorbar": False,
        "write_values": False,
    }

    x_layer = 0
    for i_layer, (layer_name, layer) in enumerate(model.named_children()):

        layer_class = layer.__class__.__name__
        layer_label = get_layer_label(layer, layer_name)

        if layer_class == "Linear":
            height = 1.0
            width = 0.8
            weight_df = get_layer_weight_matrix(layer).T
            bias_df = get_layer_bias_vector(layer)
            n_in = layer.in_features
            n_out = layer.out_features
            height = normalized_length_mapping(n_in)
            width = normalized_length_mapping(n_out)

        elif layer_class == "ReLU":
            height = 0.5
            width = 0.2
        else:
            raise NotImplementedError(f"Unknown layer: {layer_class}")

        x_0 = x_layer
        x_mid = x_0 + width / 2
        y_0 = -0.5 * height

        ax.text(x_mid, height * 0.5 + 0.1, layer_label, ha="center", va="center")

        if layer_class == "Linear":
            # weights heatmap
            axin = ax.inset_axes([x_0, y_0, width, height], transform=ax.transData)
            if n_out > 1 and seriated:
                weight_df = seriate_matrix_columns(weight_df)
                if i_layer > 1 and seriated:
                    # order the rows by the order of the columns in the previous layer
                    weight_df = weight_df.loc[previous_weight_df.columns, :]

            plot_shaded_matrix(weight_df, ax=axin, **heatmap_kwargs)

            axin.set_xticks([])
            axin.set_yticks([])

            # bias heatmap under weights heatmap
            bias_height = normalized_length_mapping(1)
            axin = ax.inset_axes(
                [x_0, y_0 - bias_height * 0.5 - 0.2, width, bias_height],
                transform=ax.transData,
            )
            if seriated:
                bias_df = bias_df.loc[weight_df.columns, :]
            plot_shaded_matrix(bias_df.T, ax=axin, **heatmap_kwargs)
            axin.set_xticks([])
            axin.set_yticks([])

        elif layer_class == "ReLU":
            axin = ax.inset_axes([x_0, y_0, width, height], transform=ax.transData)
            plot_relu(axin)
            clean_axes(axin)
            axin.patch.set_alpha(0.0)

        rectangle = matplotlib.patches.Rectangle(
            (x_0, y_0),
            width,
            height,
            linewidth=1.5,
            edgecolor="dodgerblue",
            facecolor="none",
        )
        ax.add_patch(rectangle)

        if i_layer == 0:
            draw_horizontal_arrow(ax, -x_space * 0.8, y_0 + height / 2, x_space * 0.6)

        draw_horizontal_arrow(
            ax, x_mid + width / 2 + x_space * 0.1, y_0 + height / 2, x_space * 0.6
        )
        x_layer += width + x_space
        previous_weight_df = weight_df.copy()

    clean_axes(ax)

    ax.axis("equal")
    ax.set_xlim(-x_space, x_layer + x_space * 0.1)

    plt.colorbar(
        matplotlib.cm.ScalarMappable(
            cmap=cmap,
            norm=matplotlib.colors.Normalize(vmin=-max_abs_weight, vmax=max_abs_weight),
        ),
        ax=ax,
        fraction=0.01,
    )


def get_layer_weight_matrix(layer):
    """Extract the weight matrix from a PyTorch layer and return it as a pandas DataFrame"""
    weight_matrix = layer.weight.detach().numpy()
    weight_df = pd.DataFrame(weight_matrix)
    return weight_df


def get_layer_bias_vector(layer):
    """Extract the bias vector from a PyTorch layer and return it as a pandas DataFrame"""
    bias_vector = layer.bias.detach().numpy()
    bias_df = pd.DataFrame(bias_vector)
    return bias_df


def get_max_absolute_weight(model):
    max_abs_weight = max(
        get_layer_weight_matrix(layer).abs().max().max()
        for layer in model.children()
        if hasattr(layer, "weight")
    )
    return max_abs_weight


def cubic_spline(
    x_from_to,
    y_from_to,
    n_segments: int = 20,
):
    """Draw a (horizontally oriented) cubic spline with 0 slope from a point to another point

    Args:
        x_from_to: a numpy array with two x coordinates
        y_from_to: a numpy array with two y coordinates
        n_segments: number of straight segments with which to draw the spline

    Returns:
        two numpy arrays
    """
    x_t = np.linspace(0, 1, n_segments + 1)
    y_coord = y_from_to[0] + (y_from_to[1] - y_from_to[0]) * (x_t**2 * (3 - 2 * x_t))
    x_coord = x_from_to[0] + (x_from_to[1] - x_from_to[0]) * x_t

    return x_coord, y_coord


def plot_shaded_matrix(
    data,
    cmap="viridis",
    ax=None,
    write_values=False,
    format_value=None,
    text_data=None,
    vmin=None,
    vmax=None,
    with_colorbar=True,
    cbar_title="",
    cbar_ticks=None,
):
    """Plot data as a shaded matrix

    Args:
        data: a pandas DataFrame
        cmap: a colormap, either as a string or a matplotlib colormap
        ax: an optional matplotlib axis
        write_values: if True, write the values in the cells
        format_value: a function that takes a number and returns a string
            used in write_values is True
        text_data: a pandas DataFrame of the same shape as data
            used to write values if provided, otherwise the values of data are written
        vmin: minimum value for color mapping
        vmax: maximum value for color mapping
        with_colorbar: whether a colorbar should be drawn
        cbar_title: label for the colorbar
        cbar_ticks: values of ticks for the colorbar
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    if format_value is None:
        format_value = lambda x: f"{x:.1f}"
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    if vmin is None:
        vmin = data.min().min()
    if vmax is None:
        vmax = data.max().max()
    color_norm = plt.Normalize(vmin=vmin, vmax=vmax)
    img = ax.imshow(
        data, cmap=cmap, aspect="auto", norm=color_norm, interpolation="none"
    )

    if write_values:
        if text_data is None:
            text_data = data
        for i_x in range(text_data.shape[1]):
            for i_y in range(text_data.shape[0]):
                value = text_data.iloc[i_y, i_x]
                color_value = data.iloc[i_y, i_x]
                bg_color = cmap(color_norm(color_value))
                # if bg_color is light then choose a dark text color
                if lightness_from_rgb(bg_color) > 0.4:
                    txt_color = "black"
                else:
                    txt_color = "white"
                ax.text(
                    i_x,
                    i_y,
                    format_value(value),
                    ha="center",
                    va="center",
                    color=txt_color,
                )

    if with_colorbar:
        plt.colorbar(mappable=img, ax=ax, label=cbar_title, ticks=cbar_ticks)
    ax.set_xticks(np.arange(0, len(data.columns)))
    ax.set_yticks(np.arange(len(data.index)))
    ax.set_xticklabels(data.columns)
    ax.set_yticklabels(data.index)
    ax.set_xlabel(data.columns.name)
    ax.set_ylabel(data.index.name)
    ax.tick_params(axis="x", labelrotation=90)  # avoid overlapping x tick labels


def lightness_from_rgb(rgb):
    """Get an estimate of the lightness of an RGB color"""
    r, g, b, alpha = rgb
    return 0.2 * r + 0.7 * g + 0.1 * b


def seriate_matrix_rows(data):
    """Reorder the rows of a matrix by using hierarchical clustering and optimal leaf ordering

    Args:
        data: a pandas DataFrame
    """
    matrix = data.to_numpy()
    Z_rows = hierarchy.ward(matrix)
    row_order = hierarchy.leaves_list(hierarchy.optimal_leaf_ordering(Z_rows, matrix))
    return data.iloc[row_order, :]


def seriate_matrix_columns(data):
    """Reorder the columns of a matrix by using hierarchical clustering and optimal leaf ordering"""
    return seriate_matrix_rows(data.T).T


def clean_axes(ax):
    spines_to_hide = ["top", "right", "left", "bottom"]
    for side in spines_to_hide:
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="both", which="both", length=0)
    ax.set_xticks([])
    ax.set_yticks([])