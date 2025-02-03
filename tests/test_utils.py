
import networkx as nx
import pytest

from cot2tot.utils import (
    extract_between,
    sanitize_text,
    get_tree_layers,
    get_tree_positions,
)


# --- Tests for extract_between --- #


@pytest.mark.parametrize(
    "text, start, end, expected",
    [
        (
            "Hello <start>world</start>, and <start>universe</start>!",
            "<start>",
            "</start>",
            ["world", "universe"],
        ),
        ("No delimiters here", "<start>", "</start>", []),
        ("<a>first</a><a>second</a>", "<a>", "</a>", ["first", "second"]),
        (
            "<tag>Mixed <inner>content</inner> here</tag>",
            "<tag>",
            "</tag>",
            ["Mixed <inner>content</inner> here"],
        ),
        ("Edge case: <start></start>", "<start>", "</start>", [""]),
    ],
)
def test_extract_between(text, start, end, expected):
    """
    Verify that substrings between given delimiters are correctly extracted.
    """
    result = extract_between(text, start, end)
    assert result == expected


# --- Tests for sanitize_text --- #


@pytest.mark.parametrize(
    "text, expected_substr",
    [
        # After normalization, "Café" becomes "Cafe", "Münster" becomes "Munster", etc.
        ("Café Münster — naïve fiancé", "Cafe"),
        ("naïve", "naive"),
        ("ñandú", "nandu"),
        ("ASCII only text", "ASCII only text"),
        ("💖 emoji", " emoji"),
    ],
)
def test_sanitize_text(text, expected_substr):
    """
    Verify that sanitize_text replaces accented characters with their ASCII equivalents.
    """
    sanitized = sanitize_text(text)
    # Check that every character in the sanitized string is ASCII.
    assert all(ord(char) < 128 for char in sanitized)
    # Verify that the expected substring is present in the sanitized result.
    assert expected_substr in sanitized


# --- Tests for get_tree_layers --- #


def test_get_tree_layers_simple_tree():
    """
    Build a simple tree (A -> B -> C) and check that layers and visited nodes are correct.
    """
    G = nx.DiGraph()
    G.add_edge("A", "B")
    G.add_edge("B", "C")

    visited = set()
    layers = get_tree_layers(G, "A", visited)
    expected_layers = {"A": 0, "B": 1, "C": 2}

    assert layers == expected_layers
    assert visited == {"A", "B", "C"}


def test_get_tree_layers_with_cycle():
    """
    Create a graph with a cycle and verify that nodes are not revisited.
    The function should assign layers based on the first encounter.
    """
    G = nx.DiGraph()
    # Construct a cycle: A -> B -> C -> A
    G.add_edge("A", "B")
    G.add_edge("B", "C")
    G.add_edge("C", "A")

    visited = set()
    layers = get_tree_layers(G, "A", visited)
    # We expect A at 0, B at 1, C at 2 (or equivalent depending on order).
    assert layers.get("A") == 0
    assert "B" in layers and layers["B"] == 1
    assert "C" in layers and layers["C"] == 2
    # No extra nodes visited.
    assert visited == {"A", "B", "C"}


# --- Tests for get_tree_positions --- #


def test_get_tree_positions_single_component():
    """
    Create a small tree-like graph and verify that every node has an assigned (x, y) position.
    Also check that positions reflect the tree layering (y decreases with depth).
    """
    # Build a simple tree:
    #       A
    #      / \
    #     B   C
    #    / \
    #   D   E
    G = nx.DiGraph()
    G.add_edges_from(
        [
            ("A", "B"),
            ("A", "C"),
            ("B", "D"),
            ("B", "E"),
        ]
    )
    positions = get_tree_positions(G)
    # Verify that every node in G has an (x, y) position.
    for node in G.nodes():
        assert node in positions
        x, y = positions[node]
        assert isinstance(x, float)
        assert isinstance(y, float)

    # Check that A is at the highest level (least negative y) compared to its descendants.
    y_A = positions["A"][1]
    y_B = positions["B"][1]
    y_C = positions["C"][1]
    y_D = positions["D"][1]
    y_E = positions["E"][1]
    assert y_A > y_B and y_A > y_C
    assert y_B > y_D and y_B > y_E


def test_get_tree_positions_disconnected_components():
    """
    Verify that positions are computed for a graph with disconnected components,
    and that different components are offset vertically.
    """
    G = nx.DiGraph()
    # Component 1: 1 -> 2 -> 3
    G.add_edge("1", "2")
    G.add_edge("2", "3")
    # Component 2: A -> B
    G.add_edge("A", "B")

    positions = get_tree_positions(G)

    # Ensure positions exist for all nodes.
    for node in G.nodes():
        assert node in positions

    # Compare vertical positions: components should be offset sufficiently.
    # For example, the root of each component should not share the same y-coordinate.
    # Get the y-coordinate of the first node (smallest by default ordering) in each component.
    # Component 1's root is expected to be "1" and component 2's root "A" (since "1" < "A")
    y1 = positions["1"][1]
    yA = positions["A"][1]
    # They should differ by at least 2 (due to y_offset incrementation in the algorithm).
    assert abs(y1 - yA) >= 2

    # Also, all nodes in the same component should share a consistent layer difference.
    layers_component1 = [positions[n][1] for n in ["1", "2", "3"]]
    # The differences between successive layers should be consistent (or at least strictly decreasing).
    assert layers_component1[0] > layers_component1[1] > layers_component1[2]


def test_get_tree_positions_consistency_with_order():
    """
    Create two graphs with the same structure and verify that get_tree_positions
    returns a layout that preserves relative ordering for nodes in the same layer.
    """
    # Construct a balanced binary tree.
    G1 = nx.DiGraph()
    G1.add_edges_from(
        [
            ("root", "L"),
            ("root", "R"),
            ("L", "LL"),
            ("L", "LR"),
            ("R", "RL"),
            ("R", "RR"),
        ]
    )

    positions1 = get_tree_positions(G1)

    # Create another graph with the same nodes and edges.
    G2 = nx.DiGraph()
    G2.add_edges_from(
        [
            ("root", "L"),
            ("root", "R"),
            ("L", "LL"),
            ("L", "LR"),
            ("R", "RL"),
            ("R", "RR"),
        ]
    )
    positions2 = get_tree_positions(G2)

    # For each node, the relative horizontal ordering within the same layer should be similar.
    # We check that the x-coordinate orderings for nodes at the same y-level are identical.
    def group_by_y(positions):
        groups = {}
        for node, (x, y) in positions.items():
            groups.setdefault(y, []).append((node, x))
        # Sort each group by x
        for y in groups:
            groups[y].sort(key=lambda tup: tup[1])
        return groups

    groups1 = group_by_y(positions1)
    groups2 = group_by_y(positions2)

    # For every layer present in both, the node order should match.
    for y in groups1:
        if y in groups2:
            nodes1 = [node for node, _ in groups1[y]]
            nodes2 = [node for node, _ in groups2[y]]
            assert nodes1 == nodes2


if __name__ == "__main__":
    pytest.main([__file__])
