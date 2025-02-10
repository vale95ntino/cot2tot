import textwrap

from cot2tot.models import ChainOfThoughts, Thought, GraphOfThoughts, ListOfEdges
from cot2tot.utils import extract_between
from cot2tot.llm import LLMWrapper


class Parser(LLMWrapper):
    def __init__(self, config):
        super().__init__(config)

    def parse_cot_manually(
        self,
        llm_output: str,
        thinking_start_key: str = "<|begin_of_thought|>",
        thinking_end_key: str = "<|end_of_thought|>",
        solution_start_key: str = "<|begin_of_solution|>",
        solution_end_key: str = "<|end_of_solution|>",
        thought_split_key: str = "\n\n",
    ) -> ChainOfThoughts:
        """
        Parse a chain-of-thought (CoT) string manually into a ChainOfThoughts object.

        This function extracts the solution and reasoning parts using the provided
        start and end keys, splits the reasoning into individual thoughts, and then
        creates a ChainOfThoughts object.

        Args:
            llm_output (str): The full output from the LLM containing the chain-of-thought.
            thinking_start_key (str): Marker for the beginning of the reasoning.
            thinking_end_key (str): Marker for the end of the reasoning.
            solution_start_key (str): Marker for the beginning of the solution.
            solution_end_key (str): Marker for the end of the solution.
            thought_split_key (str): Delimiter used to split the reasoning into separate thoughts.

        Returns:
            ChainOfThoughts: The parsed chain-of-thought including reasoning steps and solution.

        Raises:
            ValueError: If the solution or thinking markers are missing or appear more than once.
        """
        # Extract solution(s)
        solution = extract_between(llm_output, solution_start_key, solution_end_key)
        if not solution:
            raise ValueError("Solution start and end keys not present in 'cot'.")
        if len(solution) > 1:
            raise ValueError(
                "Solution start and end keys are present in 'cot' more than once."
            )
        solution_text = solution[0]

        # Extract thinking (reasoning)
        thinking = extract_between(llm_output, thinking_start_key, thinking_end_key)
        if not thinking:
            raise ValueError("Thinking start and end keys not present in 'cot'.")
        if len(thinking) > 1:
            raise ValueError(
                "Thinking start and end keys are present in 'cot' more than once."
            )

        # Split reasoning text into individual thoughts
        thinking_str_list = thinking[0].split(thought_split_key)

        # Build the ChainOfThoughts object with a Thought for each reasoning step
        return ChainOfThoughts(
            reasoning=[
                Thought(id=str(i + 1), text=thought_str.strip())
                for i, thought_str in enumerate(thinking_str_list)
                if thought_str.strip()
            ],
            solution=solution_text.strip(),
        )


    def parse_cot_with_llm(self, llm_output: str) -> ChainOfThoughts:
        """
        Parse a chain-of-thought (CoT) string by calling the LLM to generate a formal Python object.

        The function constructs a prompt instructing the LLM to convert the provided output
        into a ChainOfThoughts object, ensuring that all text is preserved and steps are not skipped.

        Args:
            llm_output (str): The original output from the LLM containing the chain-of-thought.

        Returns:
            ChainOfThoughts: The parsed chain-of-thought object as returned by the LLM.
        """
        prompt = textwrap.dedent(f"""
            Given the following reasoning chain of thoughts from an LLM, split it into a formal Python object.
            Rules:
            1. ALL text from the LLM output needs to be added to the chain of thoughts.
            2. Don't hallucinate nor skip steps.
            3. Your job is to transcribe into the Python object, nothing else.
            4. If I concatenate all of the Thought.text values, I should get the original text again.

            LLM output to be formatted:
            {llm_output}
        """)
        response = self.call_llm(prompt, response_format=ChainOfThoughts)
        return response


    def cot_to_graph(self, cot: ChainOfThoughts) -> GraphOfThoughts:
        """
        Convert a ChainOfThoughts object into a GraphOfThoughts.

        This function first adds all thoughts as nodes in the graph. Then, it uses the LLM
        to determine the relationships between these thoughts by generating a tree-like structure.
        The LLM is prompted with a detailed explanation of how to form the edges, ensuring that
        every node is reachable from the root and that branches are created where appropriate.

        Args:
            cot (ChainOfThoughts): The chain-of-thought object to be converted.

        Returns:
            GraphOfThoughts: A graph representation of the chain-of-thought with nodes and edges.

        Raises:
            RuntimeError: If the LLM fails to return a valid list of edges.
        """
        graph = GraphOfThoughts()

        # Add nodes to the graph for each thought
        for thought in cot.reasoning:
            graph.add_thought(thought)

        # Construct a detailed prompt for the LLM to determine relationships (edges) between thoughts
        prompt_lines = [
            "Given the following extracted LLM chain of thoughts, construct a *tree of thoughts* that represents the reasoning.", # noqa: E501
            "Rules:",
            "1. Make sure to keep the ids intact.",
            "2. DON'T SKIP any reasoning step from the chain of thoughts.",
            "3. Allow for abandoning reasoning paths.",
            "4. From the list of edges I should be able to produce the whole reasoning graph.",
            "5. All nodes need to be reachable from the root.",
            "6. From the root I should be able to reach every node.",
            f"7. The root node has id = {cot.reasoning[0].id}.",
            "8. Important: Not all thoughts have to be linearly linked in a single chain. The goal is a nice *tree of thoughts*.", # noqa: E501
            (
                "9. You have some interpretative freedom to make the reasoning branches look like a tree. "
                "Connect them where it makes sense for you."
            ),
            (
                "10. If you have several steps N, N+1, N+2 --> they should be in a reasoning branch. "
                "You only create a new branch if you change a step or there are multiple options."
            ),
            "",
            "Thoughts to be processed:",
        ]
        # Append each thought's string representation to the prompt
        for thought in cot.reasoning:
            prompt_lines.append(str(thought))
        prompt = "\n".join(prompt_lines)

        # Get the identified edges from the LLM
        identified_edges: ListOfEdges = self.call_llm(prompt, response_format=ListOfEdges)
        if not identified_edges or not identified_edges.list:
            raise RuntimeError("Error: LLM did not return a valid list of edges.")

        # Add the identified edges to the graph
        for edge in identified_edges.list:
            graph.add_edge(from_id=edge.from_node, to_id=edge.to_node, comment=edge.comment)

        return graph
