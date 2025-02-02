from typing import List
from cot2tot.models import ChainOfThoughts, Thought, GraphOfThoughts, ListOfEdges
from cot2tot.utils import extract_between
from cot2tot.llm import call_llm

def parse_cot_manually(
        llm_output: str, 
        thinking_start_key:str="<|begin_of_thought|>",
        thinking_end_key:str="<|end_of_thought|>",
        solution_start_key:str="<|begin_of_solution|>",
        solution_end_key:str="<|end_of_solution|>",
        thought_split_key:str="\n\n"
        )-> ChainOfThoughts:
    """Splits a CoT string into individual Thought objects."""
    # get solution
    solution = extract_between(llm_output,solution_start_key, solution_end_key)
    if len(solution) == 0:
        raise TypeError("Solution start and end keys not present in 'cot'.")
    if len(solution) > 1:
        raise TypeError("Solution start and end keys are present in 'cot' more than once.")
    solution = solution[0]
    # get reasoning
    thinking = extract_between(llm_output,thinking_start_key, thinking_end_key)
    if len(thinking) == 0:
        raise TypeError("Thinking start and end keys not present in 'cot'.")
    if len(thinking) > 1:
        raise TypeError("Thinking start and end keys are present in 'cot' more than once.")
    thinking_str_list = thinking[0].split(thought_split_key)
    # construct reasoning response and return
    return ChainOfThoughts(
        reasoning=[Thought(id=str(i+1),text=thought_str) for i, thought_str in enumerate(thinking_str_list)],
        solution=solution
    )


def parse_cot_with_llm(llm_output: str)-> ChainOfThoughts:
    """Splits a CoT string into individual Thought objects."""
    prompt = f"""
    Given the following reasoning chain of thoughts from an LLM, split it into a formal Python object.
    Rules:
    1. ALL text from the LLM output needs to be added to the chain of thoughts.
    2. Don't hallucinate nor skip steps.
    3. Your job is to transcribe into the Python object, nothing else.
    4. If I concatenate all of the Thought.text values, I should get the original text again.
    
    LLM output to be formatted:
    {llm_output}
    """
    response = call_llm(prompt, response_format=ChainOfThoughts)
    return response
    

def cot_to_graph(cot: ChainOfThoughts)->GraphOfThoughts:
    graph = GraphOfThoughts()
    
    # Add nodes
    for thought in cot.reasoning:
        graph.add_thought(thought)
    
    # Use LLM to determine relationships
    prompt = f"""
    Given the following extracted LLM chain of thoughts, construct a *tree of thoughts* that represents the thoughts of the LLM.
    You need to construct the whole tree.

    Rules:
    1. Make sure to keep the ids intact.
    2. DON'T SKIP any reasoning step from the chain of thoughts.
    3. Allow for abandonding reasoning paths.
    4. From the list of edges I should be able to produce the whole reasoning graph.
    5. All nodes need to be reachable from the root.
    6. From the root I should be able to reach every node
    7. The root node has id = {cot.reasoning[0].id}.
    8. Important: Not all thoughts have to be linearly linked in a single chain. The goal is a nice *tree of thoughts*.
    9. You have some interpretative freedom to make the reasoning branches look like a tree. Connect them where it makes sense for you.
    10. If you have several steps N, N+1, N+2 --> they should be in a reasoning branch. You only create a new branch if you change a step or there are multiple options.

    For example:
    Thought 119 = The problem needs to be solved using ...
    Thought 120 = The algorithm could be built with two steps:
    Thought 121 = Step 1) I take ...
    Thought 122 = Step 2) I then ...
    Thought 123 = So the whole algorithm is ...
    Thought 124 = I need to run some tests
    Thought 125 = Example 1....
    Thought 126 = Example 2....
    Thought 127 = Oh wait, Example 2 doesn't work with this algorithm...
    Thought 128 = I need to change Step 2)
    Should result in:
    Edges = 
    (119, 120) connecting the problem with the first reasoning branch
    (120,121), (121, 122), (122,123), (123,124) linarly connecting the resasoning steps in one reasoning path as they build on top of each other
    (124,125) and (124,126) creating two reasoning branches based on the the two examples
    (126,127) Connecting the example with the reasoning step where a mistake was seen
    (121,128) Connecting the old reasoning branch with the new reasoning branch at an appropriate position
    



    Thoughts to be processed:
    """
    for thought in cot.reasoning:
        prompt += f"{thought}\n"
        
    identified_edges: ListOfEdges = call_llm(prompt, response_format=ListOfEdges)
    
    if not identified_edges:
        raise RuntimeError("Error: LLM did not return a valid response.")
    
    for edge in identified_edges.list:
        graph.add_edge(from_id=edge.from_node, to_id=edge.to_node, comment=edge.comment)
    
    return graph




import json
from pathlib import Path
if __name__ == "__main__":
    file_path = Path(__file__).parent.parent / "tests" / "fixtures" / "graph3.json"
    with file_path.open("r", encoding="utf-8") as f:
        # # get llm otput
        # output = json.load(f)
        # llm_output = output["conversation"][1]["value"]
        # # construct CoT
        # cot: ChainOfThoughts= parse_cot_with_llm(llm_output)
        # cot_v2 : ChainOfThoughts = parse_cot_manually(llm_output)
        # print("# reasoning steps:", len(cot.reasoning), "(llm extraction) vs", len(cot_v2.reasoning), "(manual extraction)")
        # save CoT
        # Convert to JSON string
        # json_data = cot.model_dump_json() 
        # with open("cot2.json", "w") as f2:
        #     f2.write(json_data)
        # construct graph
        # cot = json.load(f)
        # cot = ChainOfThoughts.model_validate(cot)
        # graph = cot_to_graph(cot)
        # # save graph
        # json_data = graph.model_dump_json() 
        # with open("graph3.json", "w") as f2:
        #     f2.write(json_data)
        # get graph
        graph = json.load(f)
        graph = GraphOfThoughts.model_validate(graph)
        # visualize
        graph.force_nx_graph_update()
        graph.animate_as_tree(show_reasoning=True, speed=0.5)
        
        






