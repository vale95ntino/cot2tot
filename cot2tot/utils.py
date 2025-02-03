import re
import unicodedata
from collections import deque
from typing import Any, Dict, List, Set, Tuple, Optional

import networkx as nx


def extract_between(text: str, key_start: str, key_end: str) -> List[str]:
    """
    Extract all occurrences of substrings in `text` that are located between `key_start` and `key_end`.

    Args:
        text (str): The source string.
        key_start (str): The starting delimiter.
        key_end (str): The ending delimiter.

    Returns:
        List[str]: A list of all substrings found between the delimiters.
    """
    pattern = re.escape(key_start) + r"(.*?)" + re.escape(key_end)
    return re.findall(pattern, text, re.DOTALL)


def sanitize_text(text: str) -> str:
    """
    Sanitize the input text by replacing accented characters with their ASCII equivalents.

    This function uses Unicode normalization (NFKD) to decompose accented characters,
    then encodes to ASCII ignoring non-ASCII parts, effectively converting characters like
    'é' to 'e', 'ñ' to 'n', etc.

    Args:
        text (str): The text to sanitize.

    Returns:
        str: The sanitized, ASCII-only text.
    """
    normalized = unicodedata.normalize("NFKD", text)
    return normalized.encode("ascii", "ignore").decode("ascii")


def get_tree_layers(G: nx.Graph, root: Any, visited: Set[Any]) -> Dict[Any, int]:
    """
    Compute a mapping of each node in the connected component containing `root`
    to its depth (or layer) level, while avoiding cycles.

    Args:
        G (nx.Graph): The graph to process.
        root (Any): The starting node for the tree traversal.
        visited (Set[Any]): A set to track already visited nodes.

    Returns:
        Dict[Any, int]: A dictionary mapping nodes to their depth levels.
    """
    layers: Dict[Any, int] = {root: 0}  # Root is at level 0
    queue: deque[Any] = deque([root])
    visited.add(root)

    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                layers[neighbor] = layers[node] + 1
                visited.add(neighbor)
                queue.append(neighbor)

    return layers


def get_tree_positions(
    G: nx.Graph, root: Optional[Any] = None
) -> Dict[Any, Tuple[float, float]]:
    """
    Compute positions for nodes in a tree-like layout for the entire graph, handling disconnected components.
    Each connected component is laid out with its nodes spread horizontally based on their number
    and vertically offset to prevent overlap between components.

    **Note:** The `root` parameter is currently ignored; the layout is computed for the entire graph.

    Args:
        G (nx.Graph): The graph for which positions are to be computed.
        root (Optional[Any]): An optional starting root (currently not used).

    Returns:
        Dict[Any, Tuple[float, float]]: A dictionary mapping each node to its (x, y) position.
    """
    pos: Dict[Any, Tuple[float, float]] = {}
    visited: Set[Any] = set()
    remaining_nodes: Set[Any] = set(G.nodes())
    y_offset: float = 0.0  # Vertical offset for disconnected components

    while remaining_nodes:
        # Choose a new component's root as the smallest node in the remaining set.
        current_root = min(remaining_nodes)
        # Compute layers for the connected component starting at current_root.
        layers: Dict[Any, int] = get_tree_layers(G, current_root, visited)

        # Organize nodes by their layer (depth)
        layer_nodes: Dict[int, List[Any]] = {}
        for node, level in layers.items():
            layer_nodes.setdefault(level, []).append(node)
            remaining_nodes.discard(node)

        # Determine positions for nodes in each layer
        max_depth: int = max(layer_nodes.keys(), default=0)
        for level, nodes in layer_nodes.items():
            # Spread nodes horizontally in the range (0, 2)
            x_spacing: float = 2.0 / (len(nodes) + 1)
            y: float = -level - y_offset  # Negative y for downward layout
            for i, node in enumerate(nodes):
                pos[node] = ((i + 1) * x_spacing, y)

        # Increment y_offset for the next disconnected component
        y_offset += max_depth + 2

    return pos
