from openai import OpenAI
import os
from typing import Optional, Dict
from dotenv import load_dotenv
from pydantic import BaseModel

def call_llm(prompt: str, response_format: BaseModel, model: str = "gpt-4o-mini", temperature: float = 0, max_tokens: Optional[int] = None, ) -> Optional[BaseModel]:
    """
    Calls the OpenAI API with the given prompt and returns the response text.
    
    Args:
        prompt (str): The input prompt for the LLM.
        model (str): The model name (default: "gpt-4o-mini").
        temperature (float): Controls randomness (0 = quasi-deterministic, 1 = more creative).
        max_tokens (int): Limits the response length.po

    Returns:
        BaseModel: The response from the LLM or None if an error occurs.
    """
    try:
        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY. Set it as an environment variable in .env file.")
        
        client = OpenAI()
        completion = client.beta.chat.completions.parse(
            model=model,
            messages=[{"role": "system", "content": "You are a helpful AI that answers using the desired format to construct chains of thoughts and graphs of thoughts for LLM applications."},
                      {"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens if max_tokens is not None else None,
            response_format=response_format
        )
        return completion.choices[0].message.parsed
    
    except Exception as e:
        print(f"LLM Error: {e}")
        return None
