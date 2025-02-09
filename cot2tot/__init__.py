from cot2tot.parser import parse_cot_with_llm, cot_to_graph
from cot2tot.visualize import plot_as_tree, animate_as_tree
from typing import Optional


from cot2tot.models import ChainOfThoughts, GraphOfThoughts,CoT2ToTConfig

class CoT2ToT:
    def __init__(self, config: CoT2ToTConfig):
        # Initialize any necessary components or attributes
        self.state = {}
        self.got: GraphOfThoughts = None
        self.config = config

    def parse_cot(self, input_text:str)->ChainOfThoughts:
        return parse_cot_with_llm(input_text)

    def cot_to_graph(self, cot: ChainOfThoughts) -> GraphOfThoughts:
        return cot_to_graph(cot)

    def update_graph(self, got: GraphOfThoughts) -> None:
        self.got = got
        self.got.force_nx_graph_update()

    def animate(self,
                show_reasoning: bool=True,
                speed: float = 0.4,
                save_file_name: Optional[str] = None)->None:
        if not self.got:
            raise Exception("Graph of Thoughts need to be set first.")
        animate_as_tree(
                    self.got.nxGraph,
                    speed=speed,
                    save_file_name=save_file_name,
                    show_reasoning=show_reasoning
                )

    def plot(self)->None:
        if not self.got:
            raise Exception("Graph of Thoughts need to be set first.")
        plot_as_tree(self.got.nxGraph)


    def run_pipeline(self, input_text: str, plot: bool = False, animate: bool = False, verbose=False)->GraphOfThoughts:
        """Example pipeline combining multiple functions."""
        if verbose: print("Creating chain of thoughts (nodes)...")
        cot = self.parse_cot(input_text)
        if verbose: print("Converting to graph...")
        got = self.cot_to_graph(cot)
        self.update_graph(got)
        # got.force_nx_graph_update()
        if plot:
            if verbose: print("Plotting...")
            self.plot()
        if animate:
            if verbose: print("Animating...")
            self.animate()

        return got

# Define what is imported when using "from cot2tot import *"
__all__ = ["CoT2ToT", "CoT2ToTConfig"]
