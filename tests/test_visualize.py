import os
import tempfile

# Use a non-interactive backend to avoid Tk issues.
import matplotlib

import networkx as nx
import matplotlib.pyplot as plt
import imageio
import pytest

from cot2tot.visualize import plot_as_tree, animate_as_tree

matplotlib.use("Agg")


# --- Helper class for capturing imageio.mimsave calls --- #
class DummySaver:
    """A dummy saver to capture calls to imageio.mimsave."""

    def __init__(self):
        self.called = False
        self.fps = None
        self.frames = None
        self.filename = None

    def save(self, filename, frames, fps):
        self.called = True
        self.filename = filename
        self.frames = frames
        self.fps = fps


# --- Tests for plot_as_tree --- #


def test_plot_as_tree_valid(monkeypatch):
    """
    Test that plot_as_tree displays a valid tree plot when a valid root is present.
    We monkey-patch plt.show to prevent an interactive window.
    """
    # Create a simple graph with a default root "0".
    G = nx.DiGraph()
    G.add_node("0", label="Root Node")
    G.add_node("1", label="Child Node")
    G.add_edge("0", "1")

    # Override plt.show and plt.pause to avoid blocking.
    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(plt, "pause", lambda x: None)

    # Should not raise an error.
    plot_as_tree(G)


def test_plot_as_tree_invalid_root(monkeypatch):
    """
    Test that plot_as_tree raises a ValueError when no default root ("0" or "1")
    is present and no root is provided.
    """
    # Create a graph with nodes that do not include "0" or "1".
    G = nx.DiGraph()
    G.add_node("A", label="Node A")
    G.add_node("B", label="Node B")
    G.add_edge("A", "B")

    with pytest.raises(ValueError, match="Could not infer root id"):
        plot_as_tree(G)


# --- Tests for animate_as_tree --- #


def test_animate_as_tree_valid(monkeypatch):
    """
    Test that animate_as_tree runs without error on a simple graph.
    Monkey-patch plt.show and plt.pause so the test runs quickly.
    """
    G = nx.DiGraph()
    # Create a simple tree with default root "0"
    G.add_node("0", label="Root")
    G.add_node("1", label="Child 1")
    G.add_node("2", label="Child 2")
    G.add_edge("0", "1")
    G.add_edge("0", "2")

    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(plt, "pause", lambda x: None)

    # Run animation without saving.
    animate_as_tree(G, speed=0.1)


def test_animate_as_tree_save(monkeypatch):
    """
    Test that animate_as_tree calls imageio.mimsave to save the animation when a filename is provided.
    Monkey-patch imageio.mimsave to capture the call.
    """
    G = nx.DiGraph()
    # Create a simple tree with default root "0"
    G.add_node("0", label="Root")
    G.add_node("1", label="Child")
    G.add_edge("0", "1")

    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(plt, "pause", lambda x: None)

    dummy_saver = DummySaver()
    monkeypatch.setattr(imageio, "mimsave", dummy_saver.save)

    tmp_dir = tempfile.gettempdir()
    tmp_file = os.path.join(tmp_dir, "test_animation.gif")

    animate_as_tree(G, speed=0.05, save_file_name=tmp_file)

    assert dummy_saver.called, (
        "Expected imageio.mimsave to be called for saving animation."
    )
    assert dummy_saver.filename == tmp_file, (
        "Filename passed to mimsave does not match."
    )
    assert dummy_saver.frames is not None and len(dummy_saver.frames) > 0
    assert dummy_saver.fps > 0


def test_animate_as_tree_show_reasoning(monkeypatch):
    """
    Test that animate_as_tree runs with show_reasoning enabled without errors.
    We simulate a more complex graph with multiple nodes and labels.
    """
    G = nx.DiGraph()
    # Build a small tree with reasoning labels.
    nodes = {
        "0": "Root Reasoning",
        "1": "Child One Reasoning",
        "2": "Child Two Reasoning",
        "3": "Grandchild Reasoning",
    }
    for node, label in nodes.items():
        G.add_node(node, label=label)
    G.add_edge("0", "1")
    G.add_edge("0", "2")
    G.add_edge("1", "3")

    monkeypatch.setattr(plt, "show", lambda: None)
    monkeypatch.setattr(plt, "pause", lambda x: None)

    # Running with show_reasoning=True should complete without error.
    animate_as_tree(G, speed=0.05, show_reasoning=True)


if __name__ == "__main__":
    pytest.main([__file__])
