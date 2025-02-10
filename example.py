from cot2tot import CoT2ToT, CoT2ToTConfig
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError(
        "Missing OPENAI_API_KEY. Set it as an environment variable in .env file."
    )
api_endpoint = os.getenv("OPENAI_API_ENDPOINT")
if not api_key:
    raise ValueError(
        "Missing OPENAI_API_ENDPOINT. Set it as an environment variable in .env file."
    )

config = CoT2ToTConfig(
    llm_endpoint=api_endpoint,
    llm_key=api_key,
    llm_model="gpt-4o-mini"
)

cot2tot_instance = CoT2ToT(config)

example_reasoning = "<|begin_of_thought|>To answer this question I need to think step by step. Step 1) 5+6=11, Step 2) 11-3=7. Ah wait, it is actually 11-3=8, Step 3) 8 is the final answer\n\n<|end_of_thought|>\n\n<|begin_of_solution|>The final answer is 8 years\n\n<|end_of_solution|>" # noqa: E501

cot2tot_instance.run_pipeline(example_reasoning, verbose=True, plot=False)

cot2tot_instance.plot()

cot2tot_instance.animate(save_file_name="example_video.gif")
