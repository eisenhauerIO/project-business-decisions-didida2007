from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


def _plot_dag(include_z, save_path, show):
    """Render the DAG with or without the Maimonides rule node."""
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    positions = {
        "V": (0.14, 0.78),
        "D": (0.50, 0.50),
        "Y": (0.90, 0.50),
        "U": (0.14, 0.20),
    }
    if include_z:
        positions["Z"] = (0.50, 0.78)

    labels = {
        "V": "Enrollment (V)",
        "D": "Actual Class Size (D)",
        "Y": "Test Scores (Y)",
        "U": "Socioeconomic Status (U)",
    }
    if include_z:
        labels["Z"] = "Maimonides Rule (Z)"

    box_style = {
        "boxstyle": "round,pad=0.32",
        "fc": "#f7f7f7",
        "ec": "#333333",
        "lw": 1.6,
    }

    node_patches = {}
    for node, (x_pos, y_pos) in positions.items():
        facecolor = "#d9d9d9" if node == "U" else box_style["fc"]
        txt = ax.text(
            x_pos,
            y_pos,
            labels[node],
            ha="center",
            va="center",
            fontsize=17,
            color="#222222",
            bbox={**box_style, "fc": facecolor},
            zorder=5,
        )
        node_patches[node] = txt.get_bbox_patch()

    def draw_edge(
        start,
        end,
        linestyle="-",
        curve=0.0,
        zorder=2,
        start_shift=(0.0, 0.0),
        end_shift=(0.0, 0.0),
        gap=8,
    ):
        x1, y1 = positions[start]
        x2, y2 = positions[end]
        x1 += start_shift[0]
        y1 += start_shift[1]
        x2 += end_shift[0]
        y2 += end_shift[1]

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="->",
            mutation_scale=20,
            linewidth=2.0,
            linestyle=linestyle,
            color="#111111",
            connectionstyle=f"arc3,rad={curve}",
            patchA=node_patches[start],
            patchB=node_patches[end],
            shrinkA=gap,
            shrinkB=gap,
            zorder=zorder,
        )
        ax.add_patch(arrow)

    # Dashed arrows first (background), routed away from other node boxes.
    draw_edge(
        "V",
        "Y",
        linestyle="--",
        curve=-0.22,
        zorder=1,
        start_shift=(0.08, 0.03),
        end_shift=(-0.06, 0.11),
        gap=8,
    )
    draw_edge(
        "U",
        "Y",
        linestyle="--",
        curve=0.24,
        zorder=1,
        start_shift=(0.14, -0.02),
        end_shift=(-0.06, -0.07),
        gap=8,
    )

    # Solid arrows on top
    draw_edge("V", "D", linestyle="-", zorder=7, start_shift=(0.08, -0.03), end_shift=(-0.06, 0.06), gap=8)
    draw_edge("D", "Y", linestyle="-", zorder=7, start_shift=(0.12, 0.00), end_shift=(-0.12, 0.00), gap=8)
    draw_edge("U", "V", linestyle="--", zorder=7, start_shift=(0.03, 0.11), end_shift=(0.03, -0.11), gap=8)
    if include_z:
        draw_edge("V", "Z", linestyle="-", zorder=7, start_shift=(0.11, 0.00), end_shift=(-0.12, 0.00), gap=8)
        draw_edge("Z", "D", linestyle="-", zorder=7, start_shift=(0.00, -0.10), end_shift=(0.00, 0.10), gap=8)

    plt.tight_layout(pad=1.0)
    if save_path:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig, ax


def plot_identification_dag(save_path="pngs/angrist_lavy_dag.png", show=True):
    """Plot the full Angrist-Lavy identification DAG including the instrument node."""
    return _plot_dag(include_z=True, save_path=save_path, show=show)


def plot_identification_dag_without_z(save_path="pngs/angrist_lavy_dag_no_z.png", show=True):
    """Plot the identification DAG without the Maimonides rule node or its arrows."""
    return _plot_dag(include_z=False, save_path=save_path, show=show)
