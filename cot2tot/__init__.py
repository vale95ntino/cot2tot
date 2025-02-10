from cot2tot.parser import Parser
from cot2tot.visualize import Visualizer
from typing import Optional

from cot2tot.models import GraphOfThoughts, CoT2ToTConfig


class CoT2ToT(Parser, Visualizer):
    """
    The CoT2ToT class integrates chain-of-thought (CoT) parsing and visualization functionalities.

    It inherits from both Parser and Visualizer, enabling the transformation of raw input text
    into a chain-of-thought, converting that into a graph representation (Graph of Thoughts),
    and providing methods to plot or animate the resulting graph.

    Attributes:
        state (dict): Internal state storage for the pipeline.
        got (GraphOfThoughts): The current Graph of Thoughts.
        config (CoT2ToTConfig): The configuration settings used to initialize the pipeline.
    """

    def __init__(self, config: CoT2ToTConfig):
        """
        Initialize a new CoT2ToT instance.

        This constructor initializes the parser and visualizer components using the provided
        configuration. It also sets up initial internal state.

        Args:
            config (CoT2ToTConfig): The configuration object containing parameters for the parser,
                                    visualizer, and other pipeline settings.
        """
        # Initialize sub-classes with the appropriate configuration.
        Parser.__init__(self, config)
        Visualizer.__init__(self, config.visual_settings)
        # Initialize any necessary components or attributes.
        self.state = {}
        self.got: GraphOfThoughts = None
        self.config = config

    def update_graph(self, got: GraphOfThoughts) -> None:
        """
        Update the internal Graph of Thoughts.

        This method sets the internal graph (got) and forces an update of its internal
        NetworkX graph representation.

        Args:
            got (GraphOfThoughts): The new Graph of Thoughts to be used.
        """
        self.got = got
        self.got.force_nx_graph_update()

    def animate(self,
                show_reasoning: bool = True,
                speed: float = 0.4,
                save_file_name: Optional[str] = None) -> None:
        """
        Animate the current Graph of Thoughts as a tree.

        This method animates the process of building the tree representation of the Graph of Thoughts.
        It requires that a Graph of Thoughts has been previously set using update_graph().
        Optionally, the animation can display reasoning text and be saved as a GIF.

        Args:
            show_reasoning (bool): If True, displays reasoning text for each node during the animation.
                                   Defaults to True.
            speed (float): The pause duration (in seconds) between animation steps. Defaults to 0.4.
            save_file_name (Optional[str]): If provided, the animation will be saved to the specified file
                                            (e.g., as a GIF). Defaults to None.

        Raises:
            Exception: If the Graph of Thoughts has not been set.
        """
        if not self.got:
            raise Exception("Graph of Thoughts need to be set first.")
        self.animate_as_tree(
            self.got.nxGraph,
            speed=speed,
            save_file_name=save_file_name,
            show_reasoning=show_reasoning
        )

    def plot(self) -> None:
        """
        Plot the current Graph of Thoughts as a tree.

        This method generates a static plot of the Graph of Thoughts in a tree layout.
        It requires that a Graph of Thoughts has been previously set using update_graph().

        Raises:
            Exception: If the Graph of Thoughts has not been set.
        """
        if not self.got:
            raise Exception("Graph of Thoughts need to be set first.")
        self.plot_as_tree(self.got.nxGraph)

    def run_pipeline(self,
                     input_text: str,
                     plot: bool = False,
                     animate: bool = False,
                     verbose: bool = False) -> GraphOfThoughts:
        """
        Run the complete CoT2ToT processing pipeline.

        This method performs the following steps:
          1. Parses the input text into a chain-of-thought using an LLM.
          2. Converts the chain-of-thought into a Graph of Thoughts.
          3. Updates the internal graph.
          4. Optionally, plots and/or animates the graph.

        Args:
            input_text (str): The raw text input to be processed.
            plot (bool): If True, generates a static plot of the graph.
            animate (bool): If True, animates the process of building the graph.
            verbose (bool): If True, prints progress messages during execution.

        Returns:
            GraphOfThoughts: The generated Graph of Thoughts.
        """
        if verbose:
            print("Creating chain of thoughts (nodes)...")
        cot = self.parse_cot_with_llm(input_text)
        if verbose:
            print(cot)
        if verbose:
            print("Converting to graph...")
        got = self.cot_to_graph(cot)
        if verbose:
            print(got)
        self.update_graph(got)
        if plot:
            if verbose:
                print("Plotting...")
            self.plot()
        if animate:
            if verbose:
                print("Animating...")
            self.animate()

        return got


# Define what is imported when using "from cot2tot import *"
__all__ = ["CoT2ToT", "CoT2ToTConfig"]
