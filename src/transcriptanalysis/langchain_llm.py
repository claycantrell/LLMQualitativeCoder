# langchain_llm.py

from typing import Any, Type, Union, Dict
from langchain_openai import ChatOpenAI
from huggingface_hub import InferenceClient
from pydantic import BaseModel
from transcriptanalysis.config_schemas import LLMConfig, ProviderEnum

"""
CHANGES:
1) Added a `structured_generate` method that attempts to call `with_structured_output(...)`
   if using an OpenAI model that supports it. Otherwise, it falls back to a manual JSON approach.
"""

class LangChainLLM:
    """
    A simple wrapper around LangChain LLMs to abstract provider details (OpenAI / Hugging Face).
    """
    def __init__(self, config: LLMConfig):
        self.config = config
        self.provider = config.provider
        self.client = self._initialize_client(config)

    def _initialize_client(self, config: LLMConfig) -> Any:
        if self.provider == ProviderEnum.OPENAI:
            # ChatOpenAI from langchain_openai, passing in OpenAI API key
            return ChatOpenAI(
                model_name=config.model_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                openai_api_key=config.api_key
            )
        elif self.provider == ProviderEnum.HUGGINGFACE:
            # Hugging Face Serverless Inference API
            if not config.api_key:
                raise ValueError("Hugging Face API key is required for serverless inference.")
            return InferenceClient(
                model=config.model_name,
                token=config.api_key
            )
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def generate(self, prompt: str) -> str:
        """
        Generate text completion from the underlying client.
        """
        try:
            if self.provider == ProviderEnum.OPENAI:
                # Generate using ChatOpenAI
                response = self.client.generate([prompt])
                return response.generations[0][0].text
            elif self.provider == ProviderEnum.HUGGINGFACE:
                # Generate using Hugging Face InferenceClient
                response = self.client.chat.completions.create(prompt)
                return response.choices[0].message
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
        except Exception as e:
            # Handle exceptions and provide meaningful error messages
            raise RuntimeError(f"An error occurred during text generation: {e}")

    # ---------------------------------------------------------------
    # NEW: structured_generate method using with_structured_output
    # ---------------------------------------------------------------
    def structured_generate(self, prompt: str, schema: Type[BaseModel]) -> BaseModel:
        """
        Attempt to generate structured data from the model, adhering to the given Pydantic schema.
        - If using OpenAI with a model that supports tool/function calling or JSON Mode, 
          we leverage .with_structured_output().
        - Otherwise, we fallback to manual JSON parsing of the LLM output.
        """
        if self.provider in {ProviderEnum.OPENAI, ProviderEnum.HUGGINGFACE}:
            try:
                structured_llm = self.client.with_structured_output(schema)
                result = structured_llm.invoke(prompt)
                return result  # result is an instance of `schema`
            except AttributeError:
                # Fallback to manual
                raw_text = self.generate(prompt)
                return self._manual_parse(raw_text, schema)
        else:
            # prompt for JSON and parse
            raw_text = self.generate(prompt)
            return self._manual_parse(raw_text, schema)

    def _manual_parse(self, raw_text: str, schema: Type[BaseModel]) -> BaseModel:
        """
        Parse the raw text as JSON and validate against the given schema.
        If parsing fails, return an empty instance of the schema.
        """
        try:
            data = self._extract_first_json(raw_text)
            return schema.model_validate(data)
        except Exception:
            # In case of failure, return an empty instance
            return schema()  # type: ignore

    def _extract_first_json(self, text: str) -> Dict[str, Any]:
        """
        Very naive approach to extract the first valid JSON block from text.
        """
        import json, re
        pattern = r"\{(?:[^{}]|(?R))*\}"
        matches = re.findall(pattern, text)
        for match in matches:
            try:
                return json.loads(match)
            except json.JSONDecodeError:
                continue
        # If no match or all fail
        return {}
