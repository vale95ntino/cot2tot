from pydantic import BaseModel,  field_serializer, field_validator, ConfigDict
from typing import List, Optional, Dict
from networkx.drawing.nx_pydot import graphviz_layout
import networkx as nx
from networkx.readwrite import json_graph
import matplotlib.pyplot as plt
from cot2tot.utils import get_tree_positions
from matplotlib.widgets import Cursor
import matplotlib.animation as animation
import textwrap
import imageio
import io
import numpy as np

class Thought(BaseModel):
    id: str  # Unique identifier for the thought
    text: str  # The actual thought content
    metadata: Optional[Dict] = None  # Optional metadata (timestamps, confidence, etc.)

#
# Chain of Thoughts
#

class ChainOfThoughts(BaseModel):
    reasoning: List[Thought]
    solution: str

#
# Graph of Thoughts
#


class GraphNode(BaseModel):
    id: str
    thought: Thought
    is_abandoned: bool = False  # If this thought was abandoned during reasoning

class GraphEdge(BaseModel):
    from_node: str  # ID of the source thought
    to_node: str  # ID of the target thought
    comment: Optional[str] = None  # Explanation of the connection

class ListOfEdges(BaseModel):
    list: List[GraphEdge]

class GraphOfThoughts(BaseModel):
    nodes: Dict[str, GraphNode] = {}  # Stores thought nodes
    edges: List[GraphEdge] = []  # Stores connections
    nxGraph: nx.DiGraph = nx.DiGraph()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("nxGraph")
    def serialize_nx_graph(self, graph: nx.DiGraph, _info) -> dict:
        """Serialize networkx graph to a JSON-friendly format"""
        return json_graph.node_link_data(graph)

    @field_validator("nxGraph", mode="before")
    def deserialize_nx_graph(cls, value):
        """Deserialize networkx graph from JSON format"""
        if isinstance(value, dict):
            return json_graph.node_link_graph(value)
        return value  # If already a DiGraph, return as is

    def add_thought(self, thought: Thought):
        """Adds a new thought to the graph."""
        self.nodes[thought.id] = GraphNode(id=thought.id, thought=thought)
        # add to nx graph
        self.nxGraph.add_node(thought.id, label=thought.text)

    def add_edge(self, from_id: str, to_id: str, comment: Optional[str] = None):
        """Creates a directed edge between two thoughts with a comment."""
        if from_id in self.nodes and to_id in self.nodes:
            edge = GraphEdge(from_node=from_id, to_node=to_id, comment=comment)
            self.edges.append(edge)
        # add to nx graph
        self.nxGraph.add_edge(from_id, to_id, label=comment)

    def get_nx_graph(self):
        return self.nxGraph

    def force_nx_graph_update(self):
        G = nx.DiGraph()
        # Add nodes
        for node_id, node in self.nodes.items():
            G.add_node(node_id, label=node.thought.text)
        # Add edges
        for edge in self.edges:
            G.add_edge(edge.from_node, edge.to_node, label=edge.comment)
        # replace
        self.nxGraph = G

    def get_revisted_nodes(self):
        G = self.nxGraph
        nodes_with_multiple_incoming = [node for node, in_deg in G.in_degree() if in_deg > 1]
        return nodes_with_multiple_incoming

    def get_nodes_with_mult_children(self):
        G = self.nxGraph
        nodes_with_multiple_outgoing = [node for node, in_deg in G.out_degree() if in_deg > 1]
        return nodes_with_multiple_outgoing
    
    def plot_as_tree(self, root=None):
        if not root:
            if "0" in self.nodes:
                root = "0"
            elif "1" in self.nodes:
                root = "1"
            else:
                raise ValueError("Could not infer root id. Please provide it.")
        G = self.nxGraph
        """ Plots a graph as a tree with the root at the top. """
        pos = get_tree_positions(G, root)
        fig, ax = plt.subplots(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=2000, font_size=10, ax=ax)
        
        # Add interactive hover labels
        node_labels = nx.get_node_attributes(G, 'label')
        annotation = ax.annotate("", xy=(0, 0), xytext=(10, 10), textcoords="offset points",
                                bbox=dict(boxstyle="round", fc="w"), arrowprops=dict(arrowstyle="->"))
        annotation.set_visible(False)
        
        def on_hover(event):
            if event.inaxes == ax:
                for node, (x, y) in pos.items():
                    if abs(event.xdata - x) < 0.1 and abs(event.ydata - y) < 0.1:
                        annotation.xy = (x, y)
                        annotation.set_text(node_labels[node])
                        annotation.set_visible(True)
                        fig.canvas.draw_idle()
                        return
            annotation.set_visible(False)
            fig.canvas.draw_idle()
        
        fig.canvas.mpl_connect("motion_notify_event", on_hover)
        plt.show()

    def animate_as_tree(self, root=None, speed=0.4, show_reasoning=False, save_file_name:Optional[str]="animation.gif"):
        if not root:
            if "0" in self.nodes:
                root = "0"
            elif "1" in self.nodes:
                root = "1"
            else:
                raise ValueError("Could not infer root id. Please provide it.")
        
        G = self.nxGraph
        pos = get_tree_positions(G, root)  # Get hierarchical positions
        node_labels = nx.get_node_attributes(G, 'label')  # Node reasoning labels
        
        # Automatically adjust node size based on the number of nodes
        num_nodes = len(G.nodes)
        base_size = 2000  # Default size for small graphs
        min_size, max_size = 100, 2000  # Define min/max limits
        node_size = max(min_size, min(base_size * (10 / (num_nodes + 5)), max_size))

        # Create figure with two subplots: tree (ax1) and reasoning text (ax2)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={'height_ratios': [4, 1]})
        ax1.set_title("Tree of Thoughts - Animation")
        ax1.axis("off")  # Remove axis box around the tree plot
        ax2.axis("off")  # Hide axes for the reasoning box

        # Get nodes in depth-first order
        nodes_in_order = list(nx.dfs_preorder_nodes(G, source=root))
        
        visible_nodes = set()
        previous_node = None  # Track the last node added

        # If saving to file, prepare a list to capture frames.
        frames = [] if save_file_name is not None else None

        for node in nodes_in_order:
            visible_nodes.add(node)
            
            ax1.clear()
            ax1.axis("off")  # Ensure no axis box reappears
            
            # Initialize reasoning color as black by default
            reasoning_color = "black"

            # Determine node color dynamically
            node_colors = {n: 'lightblue' for n in visible_nodes}  # Default color
            if previous_node:
                # If the node is not a direct child of the previous node, make it purple
                if node not in G.neighbors(previous_node):
                    node_colors[node] = 'purple'
                    reasoning_color = "purple"
                else:
                    node_colors[node] = 'royalblue'  # Slightly darker blue for new node
            
            # Draw nodes with updated colors
            nx.draw_networkx_nodes(G, pos, ax=ax1, nodelist=list(visible_nodes), 
                                    node_color=[node_colors[n] for n in visible_nodes], node_size=node_size)
            
            # Draw edges only if both nodes are visible
            visible_edges = [(u, v) for u, v in G.edges if u in visible_nodes and v in visible_nodes]
            nx.draw_networkx_edges(G, pos, ax=ax1, edgelist=visible_edges, edge_color='gray')

            # Update reasoning text if enabled
            if show_reasoning:
                ax2.clear()
                ax2.axis("off")
                reasoning_text = node_labels.get(node, "(No reasoning available)")
                
                # Wrap text to fit within the same width as the tree plot
                wrapped_text = "\n".join(textwrap.wrap(reasoning_text, width=70))  # Adjust width as needed
                
                # Print in black unless it's a branch switch (then purple)
                ax2.text(0.5, 0.5, wrapped_text, ha="center", va="center", fontsize=12, fontweight="bold", 
                        color=reasoning_color)
            
            plt.pause(speed)

            # Capture frame if saving as gif
            if save_file_name is not None:
                # Draw the canvas to update it
                fig.canvas.draw()
                # Save the current figure to an in-memory buffer as PNG
                buf = io.BytesIO()
                fig.savefig(buf, format='png')
                buf.seek(0)
                # Read the PNG buffer into an image array
                image = imageio.imread(buf)
                frames.append(image)
                buf.close()

            # After the pause, reset the newly added node color to normal
            node_colors[node] = 'lightblue'
            previous_node = node  # Track last added node

        plt.show()

        # If a save filename was provided, write the frames as a gif.
        if save_file_name is not None and frames:
            # Calculate frames per second based on the pause duration
            fps = 1 / speed if speed > 0 else 1
            imageio.mimsave(save_file_name, frames, fps=fps)
            print(f"Animation saved to {save_file_name}")

