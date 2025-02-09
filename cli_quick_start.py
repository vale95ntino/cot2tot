#!/usr/bin/env python
"""
Quick Start CLI for CoT2ToT

This script provides a command-line interface for:
  - Parsing chain-of-thought (CoT) outputs using LLM or manual extraction.
  - Converting a CoT to a graph.
  - Visualizing a graph (static plot and animated tree).

Usage examples:
    # Parse an LLM output file using the LLM extraction method and save the result
    $ python quick_start.py parse --input llm_output.json --method llm --output cot.json

    # Convert a CoT (in JSON) to a graph and save the graph
    $ python quick_start.py graph --input cot.json --output graph.json

    # Visualize a graph from a JSON file
    $ python quick_start.py visualize --input graph.json --animate --show_reasoning
"""

import argparse
import json
from pathlib import Path
import sys

from cot2tot.models import GraphOfThoughts, ChainOfThoughts
from cot2tot.parser import parse_cot_with_llm, parse_cot_manually, cot_to_graph
from cot2tot.visualize import plot_as_tree, animate_as_tree


def parse_cot(args: argparse.Namespace) -> None:
    """Parse LLM output into a ChainOfThoughts and optionally save the result."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        sys.exit(f"Input file {input_path} does not exist.")

    # Read the input file (assumed to be a JSON with key "conversation" or raw text)
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Here we assume the file contains a key "conversation" with a list of messages.
    # Adjust the logic as necessary.
    if "conversation" in data:
        # Use the second message's value as an example
        llm_output = data["conversation"][1]["value"]
    else:
        # Otherwise, assume the file contains plain text.
        llm_output = data

    # Choose parsing method
    if args.method.lower() == "manual":
        cot: ChainOfThoughts = parse_cot_manually(llm_output)
    else:
        cot: ChainOfThoughts = parse_cot_with_llm(llm_output)

    print(
        f"Parsed CoT with {len(cot.reasoning)} reasoning steps and solution: {cot.solution}"
    )

    if output_path:
        output_path.write_text(cot.model_dump_json(), encoding="utf-8")
        print(f"Saved ChainOfThoughts to {output_path}")


def convert_to_graph(args: argparse.Namespace) -> None:
    """Convert a ChainOfThoughts (from JSON) into a GraphOfThoughts and optionally save it."""
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else None

    if not input_path.exists():
        sys.exit(f"Input file {input_path} does not exist.")

    cot_data = json.loads(input_path.read_text(encoding="utf-8"))
    cot = ChainOfThoughts.model_validate(cot_data)
    graph = cot_to_graph(cot)

    print(
        f"Converted CoT to graph with {len(graph.nodes)} nodes and {len(graph.edges)} edges."
    )

    if output_path:
        output_path.write_text(graph.model_dump_json(), encoding="utf-8")
        print(f"Saved graph to {output_path}")


def visualize_graph(args: argparse.Namespace) -> None:
    """Load a GraphOfThoughts from JSON and visualize it."""
    input_path = Path(args.input)
    if not input_path.exists():
        sys.exit(f"Input file {input_path} does not exist.")

    graph_data = json.loads(input_path.read_text(encoding="utf-8"))
    graph = GraphOfThoughts.model_validate(graph_data)
    graph.force_nx_graph_update()

    # Visualize the graph as a static tree plot.
    print("Displaying static tree plot...")
    plot_as_tree(graph.nxGraph)

    # Optionally, run an animation.
    if args.animate:
        print("Starting tree animation...")
        animate_as_tree(
            graph.nxGraph,
            show_reasoning=args.show_reasoning,
            speed=args.speed,
            save_file_name=args.save if args.save else None,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CoT2ToT Quick Start CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Subparser for parsing LLM output
    parse_parser = subparsers.add_parser(
        "parse", help="Parse LLM output into a ChainOfThoughts"
    )
    parse_parser.add_argument(
        "--input", required=True, help="Path to the input file (JSON or text)"
    )
    parse_parser.add_argument(
        "--method",
        choices=["llm", "manual"],
        default="llm",
        help="Parsing method to use",
    )
    parse_parser.add_argument(
        "--output", help="Path to save the parsed ChainOfThoughts as JSON"
    )
    parse_parser.set_defaults(func=parse_cot)

    # Subparser for converting a CoT to a graph
    graph_parser = subparsers.add_parser(
        "graph", help="Convert a ChainOfThoughts JSON into a GraphOfThoughts"
    )
    graph_parser.add_argument(
        "--input", required=True, help="Path to the ChainOfThoughts JSON file"
    )
    graph_parser.add_argument(
        "--output", help="Path to save the GraphOfThoughts as JSON"
    )
    graph_parser.set_defaults(func=convert_to_graph)

    # Subparser for visualizing a graph
    viz_parser = subparsers.add_parser(
        "visualize", help="Visualize a GraphOfThoughts from JSON"
    )
    viz_parser.add_argument(
        "--input", required=True, help="Path to the GraphOfThoughts JSON file"
    )
    viz_parser.add_argument(
        "--animate", action="store_true", help="Run animated tree visualization"
    )
    viz_parser.add_argument(
        "--show_reasoning", action="store_true", help="Show reasoning text in animation"
    )
    viz_parser.add_argument(
        "--speed",
        type=float,
        default=0.3,
        help="Animation speed (pause time per frame)",
    )
    viz_parser.add_argument("--save", help="Filename to save the animation as GIF")
    viz_parser.set_defaults(func=visualize_graph)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
