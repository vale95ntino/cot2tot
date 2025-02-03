from cot2tot.parser import parse_cot_manually
import json
import pytest
from pathlib import Path

from cot2tot.models import (
    ChainOfThoughts,
    Thought,
    ListOfEdges,
    GraphEdge,
)
from cot2tot.parser import parse_cot_with_llm, cot_to_graph


@pytest.fixture
def example_cot():
    """Load test data from JSON file."""
    file_path = Path(__file__).parent / "fixtures" / "llm_output.json"
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def test_json_import(example_cot):
    """Test using the imported JSON data."""
    assert "conversation" in example_cot
    assert len(example_cot["conversation"]) == 2


# --- Tests for parse_cot_manually --- #


def test_parse_cot_manually_valid():
    """
    Test that parse_cot_manually returns a valid ChainOfThoughts when provided with proper markers.
    """
    llm_output = (
        "Some preamble text...\n"
        "<|begin_of_thought|>Thought 1 text\n\nThought 2 text<|end_of_thought|>\n"
        "Some middle text...\n"
        "<|begin_of_solution|>The final solution<|end_of_solution|>\n"
        "Some postamble text..."
    )
    chain = parse_cot_manually(llm_output)
    assert chain.solution == "The final solution"
    assert len(chain.reasoning) == 2
    assert chain.reasoning[0].text == "Thought 1 text"
    assert chain.reasoning[1].text == "Thought 2 text"


def test_parse_cot_manually_missing_solution():
    """
    Test that parse_cot_manually raises a ValueError when the solution markers are missing.
    """
    llm_output = (
        "Some text...\n"
        "<|begin_of_thought|>Thought A\n\nThought B<|end_of_thought|>\n"
        "No solution markers here."
    )
    with pytest.raises(ValueError, match="Solution start and end keys not present"):
        parse_cot_manually(llm_output)


# --- Mocks for call_llm --- #


def fake_call_llm_for_chain(prompt: str, response_format):
    """
    Fake call_llm function that returns a predetermined ChainOfThoughts.
    """
    return ChainOfThoughts(
        reasoning=[
            Thought(id="1", text="Fake Thought 1"),
            Thought(id="2", text="Fake Thought 2"),
        ],
        solution="Fake Solution",
    )


def fake_call_llm_for_edges(prompt: str, response_format):
    """
    Fake call_llm function that returns a predetermined ListOfEdges.
    """
    return ListOfEdges(
        list=[GraphEdge(from_node="1", to_node="2", comment="connects 1 to 2")]
    )


# --- Tests for parse_cot_with_llm --- #


def test_parse_cot_with_llm(monkeypatch):
    """
    Test that parse_cot_with_llm returns the expected ChainOfThoughts when the LLM is mocked.
    """
    # Monkey-patch the call_llm function in the parser module.
    monkeypatch.setattr("cot2tot.parser.call_llm", fake_call_llm_for_chain)

    llm_output = "Dummy LLM output that will be ignored by the fake."
    chain = parse_cot_with_llm(llm_output)

    assert isinstance(chain, ChainOfThoughts)
    assert chain.solution == "Fake Solution"
    assert len(chain.reasoning) == 2
    assert chain.reasoning[0].text == "Fake Thought 1"
    assert chain.reasoning[1].text == "Fake Thought 2"


# --- Tests for cot_to_graph --- #


def test_cot_to_graph(monkeypatch):
    """
    Test that cot_to_graph properly builds a GraphOfThoughts using the provided chain and mocked edges.
    """
    # Create a simple ChainOfThoughts instance.
    chain = ChainOfThoughts(
        reasoning=[
            Thought(id="1", text="First Thought"),
            Thought(id="2", text="Second Thought"),
        ],
        solution="Test Solution",
    )

    # Monkey-patch call_llm in the parser module to return a fixed list of edges.
    monkeypatch.setattr("cot2tot.parser.call_llm", fake_call_llm_for_edges)

    graph = cot_to_graph(chain)

    # Check that the nodes exist.
    assert "1" in graph.nodes
    assert "2" in graph.nodes

    # Check that the edge was added both in the custom list and in the NetworkX graph.
    edge_found = any(
        edge.from_node == "1" and edge.to_node == "2" for edge in graph.edges
    )
    assert edge_found, "Expected edge from node 1 to 2 not found in custom edges list."

    # Verify the internal NetworkX graph.
    assert graph.nx_graph.has_edge("1", "2")
    edge_data = graph.nx_graph.edges["1", "2"]
    assert edge_data.get("label") == "connects 1 to 2"
