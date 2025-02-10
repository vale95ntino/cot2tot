from typing import Optional, Any, Dict, List, Tuple

import networkx as nx
import matplotlib.pyplot as plt
import textwrap
import imageio
import io
import numpy as np
from cot2tot.models import CoT2ToTVisualConfig

from cot2tot.utils import get_tree_positions

class Visualizer:
    # Constants for visualization defaults
    def __init__(self, visual_config: CoT2ToTVisualConfig):
        self.visual_config = visual_config


    def plot_as_tree(self, G: nx.Graph, root: Optional[Any] = None) -> None:
        """
        Plot a given graph as a tree with the specified root at the top.

        The function computes a tree layout based on hierarchical positions and
        displays an interactive plot with hover annotations for node labels.

        Args:
            G (nx.Graph): The graph to plot.
            root (Optional[Any]): The root node id. If not provided, attempts to infer using "0" or "1".

        Raises:
            ValueError: If no root can be inferred or provided.
        """
        if root is None:
            if G.has_node("0"):
                root = "0"
            elif G.has_node("1"):
                root = "1"
            else:
                raise ValueError("Could not infer root id. Please provide one explicitly.")

        pos: Dict[Any, Tuple[float, float]] = get_tree_positions(G, root)
        fig, ax = plt.subplots(figsize=self.visual_config.DEFAULT_FIG_SIZE)

        # Draw the tree graph
        nx.draw(
            G,
            pos,
            with_labels=True,
            node_color="lightblue",
            edge_color="gray",
            node_size=self.visual_config.DEFAULT_NODE_SIZE,
            font_size=10,
            ax=ax,
        )

        # Prepare hover annotation
        node_labels: Dict[Any, str] = nx.get_node_attributes(G, "label")
        annotation = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            bbox=dict(boxstyle="round", fc="w"),
            arrowprops=dict(arrowstyle="->"),
        )
        annotation.set_visible(False)

        def on_hover(event) -> None:
            if event.inaxes == ax and event.xdata is not None and event.ydata is not None:
                # Loop through nodes and check proximity to the mouse pointer
                for node, (x, y) in pos.items():
                    if abs(event.xdata - x) < 0.1 and abs(event.ydata - y) < 0.1:
                        annotation.xy = (x, y)
                        annotation.set_text(node_labels.get(node, str(node)))
                        annotation.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            annotation.set_visible(False)
            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", on_hover)
        plt.show()


    def animate_as_tree(
        self,
        G: nx.Graph,
        root: Optional[Any] = None,
        speed: float = 0.4,
        show_reasoning: bool = False,
        save_file_name: Optional[str] = None,
    ) -> None:
        """
        Animate the process of building a tree from a graph, gradually revealing nodes and edges.

        The animation proceeds in a depth-first order (starting from the specified root) and
        optionally displays reasoning text for each node (if available as a label). The node sizes
        are dynamically scaled based on the number of visible nodes.

        Args:
            G (nx.Graph): The graph to animate.
            root (Optional[Any]): The starting/root node id. If not provided, infers using "0" or "1".
            speed (float): Time (in seconds) to pause per fade step.
            show_reasoning (bool): If True, displays node reasoning text in a separate subplot.
            save_file_name (Optional[str]): If provided, saves the animation as a GIF with this filename.

        Raises:
            ValueError: If no root can be inferred or provided.
        """
        if root is None:
            if G.has_node("0"):
                root = "0"
            elif G.has_node("1"):
                root = "1"
            else:
                raise ValueError("Could not infer root id. Please provide one explicitly.")

        pos: Dict[Any, Tuple[float, float]] = get_tree_positions(G, root)
        node_labels: Dict[Any, str] = nx.get_node_attributes(G, "label")
        num_nodes: int = G.number_of_nodes()

        # Determine the base node size with dynamic scaling for larger graphs.
        initial_node_size: float = max(
            self.visual_config.MIN_NODE_SIZE,
            min(
                self.visual_config.DEFAULT_NODE_SIZE * (10 / (num_nodes + 5)),
                self.visual_config.MAX_NODE_SIZE
                )
        )

        # Create figure with two subplots: one for the tree and one for reasoning text.
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [4, 1]}
        )
        ax1.set_title("Tree of Thoughts - Animation")
        ax1.axis("off")
        ax2.axis("off")

        # Get nodes in depth-first order
        nodes_in_order: List[Any] = list(nx.dfs_preorder_nodes(G, source=root))
        visible_nodes: set = set()
        previous_node: Optional[Any] = None

        # Prepare a list to capture frames if saving animation
        frames: Optional[List[np.ndarray]] = [] if save_file_name is not None else None

        for node in nodes_in_order:
            visible_nodes.add(node)
            # Determine dynamic scaling: shrink nodes by 10% for each threshold reached.
            thresholds_passed: int = sum(
                1 for t in self.visual_config.SHRINK_THRESHOLDS if len(visible_nodes) >= t
            )
            dynamic_factor: float = 0.7**thresholds_passed
            effective_node_size: float = initial_node_size * dynamic_factor

            # Determine node colors: default is lightblue, with variations based on connection.
            node_colors: Dict[Any, str] = {n: "lightblue" for n in visible_nodes}
            reasoning_color: str = "black"
            if previous_node is not None:
                # If the new node is not a neighbor of the previous, highlight it differently.
                if node not in G.neighbors(previous_node):
                    node_colors[node] = "purple"
                    reasoning_color = "purple"
                else:
                    node_colors[node] = "royalblue"

            # Animate the fade-in of the new node over FADE_STEPS subframes.
            for step in range(1, self.visual_config.FADE_STEPS + 1):
                fade_alpha: float = step / self.visual_config.FADE_STEPS

                ax1.clear()
                ax1.axis("off")

                # Draw already visible nodes fully opaque (excluding the new node).
                if len(visible_nodes) > 1:
                    other_nodes = list(visible_nodes - {node})
                    nx.draw_networkx_nodes(
                        G,
                        pos,
                        ax=ax1,
                        nodelist=other_nodes,
                        node_color=[node_colors[n] for n in other_nodes],
                        node_size=effective_node_size,
                        alpha=1.0,
                    )

                # Draw the new node with gradually increasing opacity.
                nx.draw_networkx_nodes(
                    G,
                    pos,
                    ax=ax1,
                    nodelist=[node],
                    node_color=[node_colors[node]],
                    node_size=effective_node_size,
                    alpha=fade_alpha,
                )

                # Draw edges that are already visible (excluding those involving the new node).
                other_edges = [
                    (u, v)
                    for u, v in G.edges
                    if u in visible_nodes and v in visible_nodes and node not in (u, v)
                ]
                nx.draw_networkx_edges(
                    G, pos, ax=ax1, edgelist=other_edges, edge_color="gray", alpha=1.0
                )

                # Draw new edges (those connected to the new node) with fade effect.
                new_edges = [
                    (u, v)
                    for u, v in G.edges
                    if node in (u, v) and u in visible_nodes and v in visible_nodes
                ]
                nx.draw_networkx_edges(
                    G, pos, ax=ax1, edgelist=new_edges, edge_color="gray", alpha=fade_alpha
                )

                # Update reasoning text if enabled.
                if show_reasoning:
                    ax2.clear()
                    ax2.axis("off")
                    reasoning_text: str = node_labels.get(node, "(No reasoning available)")
                    wrapped_text: str = "\n".join(textwrap.wrap(reasoning_text, width=70))
                    ax2.text(
                        0.5,
                        0.5,
                        wrapped_text,
                        ha="center",
                        va="center",
                        fontsize=12,
                        fontweight="bold",
                        color=reasoning_color,
                        alpha=fade_alpha,
                    )

                plt.pause(speed / self.visual_config.FADE_STEPS)

                # Capture frame for GIF if a filename is provided.
                if save_file_name is not None:
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png")
                    buf.seek(0)
                    # Use imageio.v2.imread to avoid the deprecation warning.
                    image = imageio.v2.imread(buf)
                    frames.append(image)
                    buf.close()

            previous_node = node

        plt.show()

        # Save the animation as a GIF if requested.
        if save_file_name is not None and frames:
            fps: float = 1 / speed if speed > 0 else 1
            imageio.mimsave(save_file_name, frames, fps=fps)
            print(f"Animation saved to {save_file_name}")
