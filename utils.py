# utils.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from config_schemas import ConfigModel, DataFormatConfig
from langchain_llm import LangChainLLM

logger = logging.getLogger(__name__)

class ParseResponse(BaseModel):
    source_id: str
    quote: str

# Instantiate a Pydantic parser for structured outputs
parser = PydanticOutputParser(pydantic_object=ParseResponse)

def load_environment_variables() -> Dict[str, str]:
    """
    Loads and validates required environment variables, returning them in a dictionary.
    For example, you could load your provider API keys here if not given in config.
    """
    openai_api_key = os.getenv('OPENAI_API_KEY', '')
    # You might also have HF API keys or other environment variables
    return {
        "OPENAI_API_KEY": openai_api_key
    }

def _load_json_file(file_path: str) -> Any:
    """
    Internal helper function to load JSON content from a file.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"File '{file_path}' not found.")
        raise FileNotFoundError(f"File '{file_path}' not found.")

    try:
        with path.open('r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{file_path}': {e}")
        raise
    except IOError as e:
        logger.error(f"I/O error loading file '{file_path}': {e}")
        raise

def _load_text_file(file_path: str, description: str = 'file') -> str:
    """
    Internal helper function to load raw text content from a file.
    """
    path = Path(file_path)
    if not path.exists():
        logger.error(f"{description.capitalize()} file '{file_path}' not found.")
        raise FileNotFoundError(f"{description.capitalize()} file '{file_path}' not found.")

    try:
        with path.open('r', encoding='utf-8') as file:
            content = file.read().strip()
    except OSError as e:
        logger.error(f"Error reading {description} file '{file_path}': {e}")
        raise

    if not content:
        logger.error(f"{description.capitalize()} file '{file_path}' is empty.")
        raise ValueError(f"{description.capitalize()} file '{file_path}' is empty.")

    return content

def load_prompt_file(prompts_folder: str, prompt_file: str, description: str = 'prompt') -> str:
    """
    Loads a prompt from a specified file.
    """
    prompt_path = str(Path(prompts_folder) / prompt_file)
    return _load_text_file(prompt_path, description=description)

def load_config(config_file_path: str) -> ConfigModel:
    """
    Loads and validates the main configuration from a JSON file using Pydantic.
    """
    raw_config = _load_json_file(config_file_path)
    return ConfigModel.model_validate(raw_config)

def load_data_format_config(config_file_path: str) -> DataFormatConfig:
    """
    Loads and validates the data format configuration from a JSON file using Pydantic.
    """
    raw_config = _load_json_file(config_file_path)
    return DataFormatConfig.model_validate(raw_config)

def generate_structured_response(llm: LangChainLLM, prompt: str) -> Dict[str, Any]:
    """
    Example function that uses an LLM to produce structured data validated by Pydantic.
    """
    raw_text = llm.generate(prompt)
    try:
        parsed_obj = parser.parse(raw_text)
        return parsed_obj.dict()
    except ValidationError as ve:
        logger.error(f"Failed to parse structured output: {ve}")
        return {}
