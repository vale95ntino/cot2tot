![Logo](assets/logo.png)
# CoT2ToT: Chain-of-Thought to Tree-of-Thoughts

CoT2ToT is a Python package that processes and visualizes chain-of-thought (CoT) outputs from language models by converting them into structured graphs (trees). This tool helps you analyze and explore the reasoning steps generated by an LLM.

![Example GIF](assets/example.gif)

## Features

- **CoT Parsing:**
  Convert raw LLM outputs into structured `ChainOfThoughts` objects.
  - **LLM Extraction:** Automatically parse CoT outputs using the language model.
  - **Manual Extraction:** Use predefined delimiters to segment and extract reasoning steps.

- **Graph Construction:**
  Transform a `ChainOfThoughts` object into a `GraphOfThoughts` directed graph, representing individual thoughts as nodes and their relationships as edges.

- **Visualization:**
  Visualize graphs using:
  - **Static Tree Plot:** A hierarchical display of the reasoning graph.
  - **Animated Tree:** An animated visualization that reveals nodes and edges gradually, with optional reasoning text. Animations can be saved as GIFs.

- **Command-Line Interface (CLI):**
  A CLI tool (`quick_start.py`) that allows you to run the different steps—parsing, graph conversion, and visualization—from the command line.

## Repository Structure

- **cot2tot/**
  The core package directory containing:
  - `models.py`: Data models (e.g., `Thought`, `ChainOfThoughts`, `GraphOfThoughts`).
  - `parser.py`: Functions to parse LLM outputs into CoT objects and convert them into graphs.
  - `utils.py`: Utility functions for text processing and tree layout computations.
  - `visualize.py`: Functions for static and animated visualization of graphs.

- **tests/**
  Unit tests for the package:
  - `test_models.py`
  - `test_parser.py`
  - `test_utils.py`
  - `test_visualize.py`
  - **test/fixtures/**
    Sample data files used in tests and demonstrations.

- **quick_start.py**
  A command-line interface tool for running parsing, graph conversion, and visualization tasks.



- **pyproject.toml**
  Poetry configuration file that manages dependencies and package metadata.

## Installation with Poetry

CoT2ToT uses [Poetry](https://python-poetry.org/) for dependency management and packaging.

```bash
git clone https://github.com/your_username/cot2tot.git
cd cot2tot
poetry install
```

## Usage

You can create a CoT2ToT instance by using your own LLM enpoint and key which will be used with the OpenAI python library. Once a graph is created, you can plot it or animate it.

```python
from cot2tot import CoT2ToT, CoT2ToTConfig

config = CoT2ToTConfig(
    llm_endpoint="<YOUR ENDPOINT>",
    llm_key="<YOUR KEY>",
    llm_model="<CHOSEN LLM MODEL>"
)

cot2tot_instance = CoT2ToT(config)

example_reasoning = "<|begin_of_thought|>[....]<|end_of_solution|>"

cot2tot_instance.run_pipeline(example_reasoning, verbose=True, plot=False)

cot2tot_instance.plot()

cot2tot_instance.animate(save_file_name="example_video.gif")
```

## CLI

The package includes an CLI tool (`cli_quick_start.py`) for running different processing steps.

### 1. Parsing an LLM Output

Convert an LLM output file into a `ChainOfThoughts` object.
You can choose between LLM extraction (default) or manual extraction.

```bash
poetry run python quick_start.py parse --input path/to/llm_output.json --output parsed_cot.json
```

For manual extraction:

```bash
poetry run python quick_start.py parse --input path/to/llm_output.json --method manual --output parsed_cot.json
```

### 2. Converting a CoT to a Graph

Convert a parsed `ChainOfThoughts` JSON file into a `GraphOfThoughts`.

```bash
poetry run python quick_start.py graph --input parsed_cot.json --output graph.json
```

### 3. Visualizing a Graph

Visualize a graph from a JSON file either as a static plot or an animated visualization.

Static tree plot:

```bash
poetry run python quick_start.py visualize --input graph.json
```

Animated visualization (with reasoning text and GIF saving):

```bash
poetry run python quick_start.py visualize --input graph.json --animate --show_reasoning --speed 0.3 --save tree_animation.gif
```

## Running Tests

To run the tests with Poetry:

```bash
poetry run pytest
```

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements and bug fixes.

## License

This project is licensed under the Apache-2.0 License. See the [LICENSE](LICENSE) file for details.
