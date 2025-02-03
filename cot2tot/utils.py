import re


def extract_between(text: str, key_start: str, key_end: str) -> list:
    """Extracts all occurrences of text between key_start and key_end."""
    pattern = re.escape(key_start) + r"(.*?)" + re.escape(key_end)
    return re.findall(pattern, text, re.DOTALL)


def sanitize_text(text):
    return text.encode("ascii", "ignore").decode()  # Remove non-ASCII characters


import networkx as nx
from collections import deque


def get_tree_layers(G, root, visited):
    """Returns a dictionary mapping each node to its depth level while handling cycles."""
    layers = {root: 0}  # Root is at level 0
    queue = deque([root])
    visited.add(root)  # Prevent infinite loops due to cycles

    while queue:
        node = queue.popleft()
        for neighbor in G.neighbors(node):
            if neighbor not in visited:  # Avoid cycles
                layers[neighbor] = layers[node] + 1
                visited.add(neighbor)
                queue.append(neighbor)

    return layers


def get_tree_positions(G, root):
    """Compute positions for nodes in a tree-like layout while preventing cycles."""
    pos = {}
    visited = set()
    remaining_nodes = set(G.nodes())
    roots = []
    y_offset = 0  # Track the vertical offset for disconnected components

    while remaining_nodes:
        # Pick the smallest non-visited node as the new root
        root = min(remaining_nodes)
        roots.append(root)
        layers = get_tree_layers(G, root, visited)

        layer_nodes = {}
        for node, level in layers.items():
            if level not in layer_nodes:
                layer_nodes[level] = []
            layer_nodes[level].append(node)
            remaining_nodes.discard(node)

        # Assign x, y positions
        max_depth = max(layer_nodes.keys()) if layer_nodes else 0
        for level, nodes in layer_nodes.items():
            x_spacing = 2.0 / (len(nodes) + 1)  # Spread nodes in each layer
            y = -level - y_offset  # Ensure new trees are placed below previous ones
            for i, node in enumerate(nodes):
                pos[node] = ((i + 1) * x_spacing, y)

        y_offset += max_depth + 2  # Ensure enough space for the next component

    return pos
