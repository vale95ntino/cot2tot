from .parser import parse_cot_with_llm, cot_to_graph
from .visualize import plot_as_tree, animate_as_tree


from cot2tot.models import ChainOfThoughts, GraphOfThoughts,CoT2ToTConfig

class CoT2ToT:
    def __init__(self, config: CoT2ToTConfig):
        # Initialize any necessary components or attributes
        self.state = {}
        self.cot: GraphOfThoughts = None
        self.config = config

    def parse_cot(self, input_text:str)->ChainOfThoughts:
        return parse_cot_with_llm(input_text)

    def cot_to_graph(self, cot: ChainOfThoughts) -> GraphOfThoughts:
        self.cot = cot
        return cot_to_graph(cot)

    def run_pipeline(self, input_text: str, plot: bool = False, animate: bool = False)->GraphOfThoughts:
        """Example pipeline combining multiple functions."""
        cot = self.parse_cot(input_text)
        self.cot_to_graph(cot) # saved in self.cot
        self.got.force_nx_graph_update()
        if plot:
            plot_as_tree(self.got.nxGraph)
        if animate:
            animate_as_tree(
                    self.got.nxGraph,
                    show_reasoning=True
                )

        return self.got

# Define what is imported when using "from cot2tot import *"
__all__ = ["CoT2ToT"]
