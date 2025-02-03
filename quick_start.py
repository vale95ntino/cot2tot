import json
from pathlib import Path


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
    # plot_as_tree(graph.nxGraph)
    animate_as_tree(graph.nxGraph, show_reasoning=True, speed=0.3)
