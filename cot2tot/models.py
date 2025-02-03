from pydantic import BaseModel, field_serializer, field_validator, ConfigDict
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
