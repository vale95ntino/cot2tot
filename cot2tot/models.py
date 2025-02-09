from typing import List, Optional, Dict, Any
import networkx as nx
from networkx.readwrite import json_graph
from pydantic import BaseModel, field_serializer, field_validator, ConfigDict


class CoT2ToTConfig(BaseModel):
    """
    Represent the config file used to setting up the CoT2ToT class.

    Attributes:
        llm_endpoint (str): Endpoint used for LLM in OpenAI SDK
        llm_key (str): Key for endpoint used for LLM in OpenAI SDK
        llm_model (str): Key for model used for LLM in OpenAI SDK
    """
    llm_endpoint: str
    llm_key: str
    llm_model: str

class Thought(BaseModel):
    """
    Represents an individual thought with associated text and optional metadata.

    Attributes:
        id (str): Unique identifier for the thought.
        text (str): The actual thought content.
        metadata (Optional[Dict[str, Any]]): Optional metadata such as timestamps or confidence values.
    """

    id: str
    text: str
    metadata: Optional[Dict[str, Any]] = None


class ChainOfThoughts(BaseModel):
    """
    Represents a chain of thoughts culminating in a final solution.

    Attributes:
        reasoning (List[Thought]): A list of Thought objects representing the reasoning process.
        solution (str): The final solution derived from the chain of thoughts.
    """

    reasoning: List[Thought]
    solution: str


class GraphNode(BaseModel):
    """
    Represents a node in the graph of thoughts.

    Attributes:
        id (str): Unique identifier for the graph node.
        thought (Thought): The thought associated with this node.
        is_abandoned (bool): Flag indicating if this thought was abandoned during reasoning.
    """

    id: str
    thought: Thought
    is_abandoned: bool = False


class GraphEdge(BaseModel):
    """
    Represents a directed edge between two thought nodes in the graph.

    Attributes:
        from_node (str): ID of the source thought.
        to_node (str): ID of the target thought.
        comment (Optional[str]): An optional explanation of the connection.
    """

    from_node: str
    to_node: str
    comment: Optional[str] = None


class ListOfEdges(BaseModel):
    """
    Represents a collection of graph edges.

    Attributes:
        list (List[GraphEdge]): List of GraphEdge objects.
    """

    list: List[GraphEdge]


class GraphOfThoughts(BaseModel):
    """
    Represents a graph of thoughts with nodes and edges.

    Attributes:
        nodes (Dict[str, GraphNode]): Dictionary mapping node IDs to GraphNode objects.
        edges (List[GraphEdge]): List of GraphEdge objects representing connections.
        nxGraph (nx.DiGraph): An internal NetworkX directed graph for visualization and analysis.
    """

    nodes: Dict[str, GraphNode] = {}
    edges: List[GraphEdge] = []
    nxGraph: nx.DiGraph = nx.DiGraph()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("nxGraph")
    def serialize_nx_graph(self, graph: nx.DiGraph, _info: Any) -> dict:
        """
        Serialize the NetworkX graph into a JSON-friendly format.

        Args:
            graph (nx.DiGraph): The NetworkX directed graph to serialize.
            _info (Any): Additional context provided by Pydantic.

        Returns:
            dict: The JSON-serializable representation of the graph.
        """
        # Explicitly set edges to "links" to preserve current behavior
        return json_graph.node_link_data(graph, edges="links")

    @field_validator("nxGraph", mode="before")
    def deserialize_nx_graph(cls, value: Any) -> nx.DiGraph:
        """
        Deserialize a JSON-friendly representation of a graph into a NetworkX DiGraph.

        Args:
            value (Any): The JSON-serializable representation of the graph or an existing DiGraph.

        Returns:
            nx.DiGraph: The deserialized NetworkX directed graph.
        """
        if isinstance(value, dict):
            # Explicitly set edges to "links" to preserve current behavior
            return json_graph.node_link_graph(value, edges="links")
        return value  # Already a DiGraph

    def add_thought(self, thought: Thought) -> None:
        """
        Add a new thought to the graph.

        This method creates a GraphNode from the given Thought and adds it to both
        the internal dictionary and the NetworkX graph.

        Args:
            thought (Thought): The thought to be added.
        """
        self.nodes[thought.id] = GraphNode(id=thought.id, thought=thought)
        self.nxGraph.add_node(thought.id, label=thought.text)

    def add_edge(self, from_id: str, to_id: str, comment: Optional[str] = None) -> None:
        """
        Create a directed edge between two thoughts with an optional comment.

        Args:
            from_id (str): The source thought ID.
            to_id (str): The target thought ID.
            comment (Optional[str]): An optional explanation of the connection.
        """
        if from_id in self.nodes and to_id in self.nodes:
            edge = GraphEdge(from_node=from_id, to_node=to_id, comment=comment)
            self.edges.append(edge)
        self.nxGraph.add_edge(from_id, to_id, label=comment)

    @property
    def nx_graph(self) -> nx.DiGraph:
        """
        Get the current NetworkX graph representing the graph of thoughts.

        Returns:
            nx.DiGraph: The internal NetworkX directed graph.
        """
        return self.nxGraph

    def force_nx_graph_update(self) -> None:
        """
        Rebuild the internal NetworkX graph from the stored nodes and edges.

        This can be useful if the graph's structure has been modified
        directly or deserialized from an external source.
        """
        updated_graph = nx.DiGraph()
        for node_id, node in self.nodes.items():
            updated_graph.add_node(node_id, label=node.thought.text)
        for edge in self.edges:
            updated_graph.add_edge(edge.from_node, edge.to_node, label=edge.comment)
        self.nxGraph = updated_graph
