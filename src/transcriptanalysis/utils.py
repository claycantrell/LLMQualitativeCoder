# transcriptanalysis/utils.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any
from pydantic import ValidationError

from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

from .config_schemas import ConfigModel, DataFormatConfig
from .langchain_llm import LangChainLLM

from importlib import resources  # Import importlib.resources

logger = logging.getLogger(__name__)

class ParseResponse(BaseModel):
    source_id: str
    quote: str

# Instantiate a Pydantic parser for structured outputs
parser = PydanticOutputParser(pydantic_object=ParseResponse)

def load_environment_variables() -> Dict[str, str]:
    """
    Loads and validates required environment variables, returning them in a dictionary.
    First checks for a local_config.json file with API keys, then falls back to environment variables.
    Users can use either method or both.
    """
    # Initialize with empty values
    openai_api_key = ''
    huggingface_api_key = ''
    
    # First try to load from local_config.json
    try:
        local_config_path = Path(__file__).parent / 'configs' / 'local_config.json'
        if local_config_path.exists():
            with local_config_path.open('r', encoding='utf-8') as file:
                local_config = json.load(file)
                api_keys = local_config.get('api_keys', {})
                openai_api_key = api_keys.get('openai_api_key', '')
                huggingface_api_key = api_keys.get('huggingface_api_key', '')
                
                # Skip placeholder values
                if openai_api_key in ('your-openai-api-key', 'your-openai-api-key-here'):
                    openai_api_key = ''
                if huggingface_api_key in ('your-huggingface-api-key', 'your-huggingface-api-key-here'):
                    huggingface_api_key = ''
                
                if openai_api_key or huggingface_api_key:
                    logger.info("API keys loaded from local_config.json")
    except Exception as e:
        logger.warning(f"Error loading local config: {e}")
        # If local_config.json doesn't exist, check if we can create it from the template
        try:
            template_path = Path(__file__).parent / 'configs' / 'local_config.template.json'
            if template_path.exists() and not local_config_path.exists():
                logger.info("Creating local_config.json from template")
                import shutil
                shutil.copy(template_path, local_config_path)
        except Exception as copy_error:
            logger.warning(f"Error copying template config: {copy_error}")
    
    # If keys are not found in local config, try environment variables
    if not openai_api_key:
        openai_api_key = os.getenv('OPENAI_API_KEY', '')
        if openai_api_key:
            logger.info("OpenAI API key loaded from environment variable")
            
    if not huggingface_api_key:
        huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY', '')
        if huggingface_api_key:
            logger.info("HuggingFace API key loaded from environment variable")
    
    return {
        "OPENAI_API_KEY": openai_api_key,
        "HUGGINGFACE_API_KEY": huggingface_api_key
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

def load_prompt_file(package: str, filename: str, description: str = 'prompt') -> str:
    """
    Loads a prompt from a specified package and filename using importlib.resources.
    
    Args:
        package (str): The package path (e.g., 'transcriptanalysis.prompts').
        filename (str): The name of the prompt file (e.g., 'parse.txt').
        description (str): A description of the file for error messages.
    
    Returns:
        str: The contents of the prompt file.
    
    Raises:
        FileNotFoundError: If the prompt file is not found within the package.
        Exception: If any other error occurs while loading the prompt file.
    """
    try:
        return resources.read_text(package, filename)
    except FileNotFoundError:
        raise FileNotFoundError(f"{description.capitalize()} file '{package}/{filename}' not found.")
    except Exception as e:
        raise Exception(f"An error occurred while loading {description} file '{package}/{filename}': {e}")

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
