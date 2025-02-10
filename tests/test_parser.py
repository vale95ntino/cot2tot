import json
import pytest
from pathlib import Path
from pydantic import PrivateAttr
import networkx as nx

# Import from your package public API.
from cot2tot import CoT2ToT, CoT2ToTConfig
from cot2tot.models import (
    ChainOfThoughts,
    Thought,
    ListOfEdges,
    GraphOfThoughts,
    GraphEdge,
)

# --- Fake call_llm functions for testing --- #

def fake_call_llm_for_chain(prompt: str, response_format):
    """Fake response for generating a ChainOfThoughts."""
    return ChainOfThoughts(
        reasoning=[
            Thought(id="1", text="Fake Thought 1"),
            Thought(id="2", text="Fake Thought 2"),
        ],
        solution="Fake Solution",
    )

def fake_call_llm_for_edges(prompt: str, response_format):
    """Fake response for generating a ListOfEdges."""
    return ListOfEdges(
        list=[GraphEdge(from_node="1", to_node="2", comment="connects 1 to 2")]
    )

# --- Dummy GraphOfThoughts for testing update_graph --- #

class DummyGraphOfThoughts(GraphOfThoughts):
    _nxGraph_updated: bool = PrivateAttr(default=False)

    def __init__(self, **data):
        super().__init__(**data)
        # Create a simple NetworkX graph for visualization functions.
        self.nxGraph = nx.Graph()

    def force_nx_graph_update(self):
        self._nxGraph_updated = True

    @property
    def nxGraph_updated(self):
        return self._nxGraph_updated

# --- Fixture for sample JSON data --- #

@pytest.fixture
def example_cot():
    """Load test data from a JSON file."""
    file_path = Path(__file__).parent / "fixtures" / "llm_output.json"
    with file_path.open("r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture
def default_config():
    """Return a default CoT2ToTConfig instance."""
    return CoT2ToTConfig(
        llm_endpoint="https://dummy-endpoint",
        llm_key="dummy-key",
        llm_model="dummy-model"
    )

# --- Tests for parser methods --- #

def test_parse_cot_manually_valid(default_config):
    """
    Test that parse_cot_manually returns a valid ChainOfThoughts
    when provided with proper markers.
    """
    instance = CoT2ToT(default_config)
    llm_output = (
        "Some preamble text...\n"
        "<|begin_of_thought|>Thought 1 text\n\nThought 2 text<|end_of_thought|>\n"
        "Some middle text...\n"
        "<|begin_of_solution|>The final solution<|end_of_solution|>\n"
        "Some postamble text..."
    )
    chain = instance.parse_cot_manually(llm_output)
    assert chain.solution == "The final solution"
    assert len(chain.reasoning) == 2
    assert chain.reasoning[0].text == "Thought 1 text"
    assert chain.reasoning[1].text == "Thought 2 text"

def test_parse_cot_manually_missing_solution(default_config):
    """
    Test that parse_cot_manually raises a ValueError when solution markers are missing.
    """
    instance = CoT2ToT(default_config)
    llm_output = (
        "Some text...\n"
        "<|begin_of_thought|>Thought A\n\nThought B<|end_of_thought|>\n"
        "No solution markers here."
    )
    with pytest.raises(ValueError, match="Solution start and end keys not present"):
        instance.parse_cot_manually(llm_output)

def test_parse_cot_with_llm(default_config, monkeypatch):
    """
    Test that parse_cot_with_llm returns the expected ChainOfThoughts when the LLM is mocked.
    """
    instance = CoT2ToT(default_config)
    monkeypatch.setattr(instance, "call_llm", fake_call_llm_for_chain)
    llm_output = "Dummy LLM output that will be ignored by the fake."
    chain = instance.parse_cot_with_llm(llm_output)
    assert isinstance(chain, ChainOfThoughts)
    assert chain.solution == "Fake Solution"
    assert len(chain.reasoning) == 2
    assert chain.reasoning[0].text == "Fake Thought 1"
    assert chain.reasoning[1].text == "Fake Thought 2"

def test_cot_to_graph(default_config, monkeypatch):
    """
    Test that cot_to_graph properly builds a GraphOfThoughts from a given chain.
    """
    # Create a simple ChainOfThoughts instance.
    cot = ChainOfThoughts(
        reasoning=[
            Thought(id="1", text="First Thought"),
            Thought(id="2", text="Second Thought"),
        ],
        solution="Test Solution",
    )
    instance = CoT2ToT(default_config)
    monkeypatch.setattr(instance, "call_llm", fake_call_llm_for_edges)
    graph = instance.cot_to_graph(cot)
    # Verify that nodes "1" and "2" exist.
    assert "1" in graph.nodes
    assert "2" in graph.nodes
    # Check that the edge from "1" to "2" is present.
    edge_found = any(
        edge.from_node == "1" and edge.to_node == "2" for edge in graph.edges
    )
    assert edge_found, "Expected edge from node 1 to 2 not found."
    # Verify the NetworkX graph.
    assert graph.nxGraph.has_edge("1", "2")
    edge_data = graph.nxGraph.edges["1", "2"]
    assert edge_data.get("label") == "connects 1 to 2"

# --- Tests for CoT2ToT methods --- #

def test_update_graph(default_config):
    """
    Test that update_graph properly sets the GraphOfThoughts and calls its update method.
    """
    instance = CoT2ToT(default_config)
    dummy_graph = DummyGraphOfThoughts()
    dummy_graph.nxGraph.add_node("1")
    instance.update_graph(dummy_graph)
    assert instance.got == dummy_graph
    assert dummy_graph.nxGraph_updated is True

def test_plot_without_graph(default_config):
    """
    Test that plot raises an Exception if the graph has not been set.
    """
    instance = CoT2ToT(default_config)
    with pytest.raises(Exception, match="Graph of Thoughts need to be set first."):
        instance.plot()

def test_animate_without_graph(default_config):
    """
    Test that animate raises an Exception if the graph has not been set.
    """
    instance = CoT2ToT(default_config)
    with pytest.raises(Exception, match="Graph of Thoughts need to be set first."):
        instance.animate()

def test_run_pipeline(default_config, monkeypatch):
    """
    Test the full pipeline of run_pipeline by monkey-patching call_llm to return
    predetermined responses for both chain creation and edge identification.
    """
    instance = CoT2ToT(default_config)

    def fake_call_llm(prompt: str, response_format):
        if response_format == ChainOfThoughts:
            return ChainOfThoughts(
                reasoning=[
                    Thought(id="1", text="Fake Thought 1"),
                    Thought(id="2", text="Fake Thought 2"),
                ],
                solution="Fake Solution"
            )
        elif response_format == ListOfEdges:
            return ListOfEdges(
                list=[GraphEdge(from_node="1", to_node="2", comment="connects 1 to 2")]
            )

    monkeypatch.setattr(instance, "call_llm", fake_call_llm)
    input_text = "Dummy input for LLM"
    # Run the pipeline without plotting or animating.
    got = instance.run_pipeline(input_text, plot=False, animate=False, verbose=False)
    assert instance.got == got
    # Verify that the graph contains the expected nodes and edge.
    assert "1" in got.nodes
    assert "2" in got.nodes
    edge_found = any(edge.from_node == "1" and edge.to_node == "2" for edge in got.edges)
    assert edge_found

# --- Optional: Tests for configuration defaults --- #

def test_default_visual_config():
    """
    Test that a CoT2ToTConfig without explicit visual_settings uses the default values.
    """
    config = CoT2ToTConfig(
        llm_endpoint="https://dummy-endpoint",
        llm_key="dummy-key",
        llm_model="dummy-model"
    )
    visual = config.visual_settings
    assert visual.DEFAULT_FIG_SIZE == (8, 6)
    assert visual.DEFAULT_NODE_SIZE == 2000
    assert visual.MIN_NODE_SIZE == 100
    assert visual.FADE_STEPS == 2

def test_partial_visual_config_override():
    """
    Test that providing only some visual settings properly overrides defaults.
    """
    config = CoT2ToTConfig(
        llm_endpoint="https://dummy-endpoint",
        llm_key="dummy-key",
        llm_model="dummy-model",
        visual_settings={"DEFAULT_NODE_SIZE": 1500, "FADE_STEPS": 3}
    )
    visual = config.visual_settings
    # Overridden values
    assert visual.DEFAULT_NODE_SIZE == 1500
    assert visual.FADE_STEPS == 3
    # Unspecified values remain default
    assert visual.DEFAULT_FIG_SIZE == (8, 6)
    assert visual.MIN_NODE_SIZE == 100
