import networkx as nx
import pytest
from pydantic import ValidationError

from cot2tot.models import Thought, ChainOfThoughts, GraphOfThoughts


def test_thought_creation():
    """Test that a Thought instance is created correctly."""
    thought = Thought(
        id="t1", text="This is a test thought.", metadata={"confidence": 0.9}
    )
    assert thought.id == "t1"
    assert thought.text == "This is a test thought."
    assert thought.metadata == {"confidence": 0.9}


def test_chain_of_thoughts_creation():
    """Test that a ChainOfThoughts instance is created with the correct reasoning and solution."""
    thought1 = Thought(id="t1", text="Step 1")
    thought2 = Thought(id="t2", text="Step 2")
    chain = ChainOfThoughts(reasoning=[thought1, thought2], solution="Final answer")
    assert len(chain.reasoning) == 2
    assert chain.solution == "Final answer"


def test_graph_of_thoughts_add_thought_and_edge():
    """Test adding thoughts and edges to GraphOfThoughts."""
    graph = GraphOfThoughts()

    # Create and add thoughts
    thought1 = Thought(id="t1", text="First thought")
    thought2 = Thought(id="t2", text="Second thought")
    graph.add_thought(thought1)
    graph.add_thought(thought2)

    # Check that nodes were added to the internal dictionary and NetworkX graph
    assert "t1" in graph.nodes
    assert "t2" in graph.nodes
    assert graph.nx_graph.has_node("t1")
    assert graph.nx_graph.has_node("t2")

    # Add an edge and check it is added
    graph.add_edge("t1", "t2", comment="connects first to second")
    # Check the custom edges list
    assert any(e.from_node == "t1" and e.to_node == "t2" for e in graph.edges)
    # Check the edge in the NetworkX graph
    assert graph.nx_graph.has_edge("t1", "t2")
    # Check that the edge attribute was added correctly
    assert graph.nx_graph.edges["t1", "t2"].get("label") == "connects first to second"


def test_nx_graph_property():
    """Test that the nx_graph property returns the correct NetworkX graph."""
    graph = GraphOfThoughts()
    thought = Thought(id="t1", text="Test thought")
    graph.add_thought(thought)
    # Access using the property
    nx_graph = graph.nx_graph
    assert isinstance(nx_graph, nx.DiGraph)
    assert nx_graph.has_node("t1")


def test_force_nx_graph_update():
    """Test that force_nx_graph_update rebuilds the NetworkX graph correctly."""
    graph = GraphOfThoughts()
    thought1 = Thought(id="t1", text="First thought")
    thought2 = Thought(id="t2", text="Second thought")
    graph.add_thought(thought1)
    graph.add_thought(thought2)
    graph.add_edge("t1", "t2", comment="edge comment")

    # Manually corrupt the nxGraph
    graph.nxGraph = nx.DiGraph()
    assert not graph.nx_graph.has_node("t1")

    # Force update
    graph.force_nx_graph_update()
    updated_graph = graph.nx_graph
    assert updated_graph.has_node("t1")
    assert updated_graph.has_node("t2")
    assert updated_graph.has_edge("t1", "t2")
    assert updated_graph.edges["t1", "t2"].get("label") == "edge comment"


def test_graph_serialization_deserialization():
    """Test that serialization and deserialization of nxGraph works correctly."""
    graph = GraphOfThoughts()
    thought1 = Thought(id="t1", text="First thought")
    thought2 = Thought(id="t2", text="Second thought")
    graph.add_thought(thought1)
    graph.add_thought(thought2)
    graph.add_edge("t1", "t2", comment="connects t1 to t2")

    # Serialize the graph to a JSON-friendly dict
    serialized = graph.model_dump_json()
    # Deserialize to create a new instance
    deserialized = GraphOfThoughts.model_validate_json(serialized)

    # Force update the NetworkX graph in case it's needed
    deserialized.force_nx_graph_update()

    # Check that the nodes and edges match
    assert "t1" in deserialized.nodes
    assert "t2" in deserialized.nodes
    assert deserialized.nx_graph.has_edge("t1", "t2")
    assert deserialized.nx_graph.edges["t1", "t2"].get("label") == "connects t1 to t2"


def test_invalid_thought_data():
    """Test that creating a Thought with invalid data raises an error."""
    with pytest.raises(ValidationError):
        # Missing required 'id' field
        Thought(text="Missing id")
