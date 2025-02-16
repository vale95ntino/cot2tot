from openai import OpenAI
from typing import Optional
from pydantic import BaseModel


class LLMWrapper:
    def call_llm(
        self,
        prompt: str,
        response_format: BaseModel,
        temperature: float = 0,
        max_tokens: Optional[int] = None,
    ) -> Optional[BaseModel]:
        """
        Calls the OpenAI API with the given prompt and returns the response text.

        Args:
            prompt (str): The input prompt for the LLM.
            response_format (BaseModel): BaseModel used for Pydantic structured output.
            temperature (float): Controls randomness (0 = quasi-deterministic, 1 = more creative).
            max_tokens (int): Limits the response length.po

        Returns:
            BaseModel: The response from the LLM or None if an error occurs.
        """
        try:
            client = OpenAI(api_key=self.config.llm_key, base_url=self.config.llm_endpoint)
            completion = client.beta.chat.completions.parse(
                model=self.config.llm_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful AI that answers using the desired format to construct chains of thoughts and graphs of thoughts for LLM applications.", # noqa: E501
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=temperature,
                max_tokens=max_tokens if max_tokens is not None else None,
                response_format=response_format,
            )
            return completion.choices[0].message.parsed

        except Exception as e:
            print(f"LLM Error: {e}")
            return None
