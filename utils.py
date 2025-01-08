# utils.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, TypedDict

from pydantic import ValidationError

logger = logging.getLogger(__name__)

# Import your updated Pydantic models
from config_schemas import ConfigModel, DataFormatConfig

class ProcessedCodeItem(TypedDict, total=False):
    """
    Example TypedDict for code items loaded from a codebase JSONL file.
    Customize fields based on your actual data structure.
    """
    code_name: str
    code_description: str
    # Add other known fields here, e.g. 'parent_code', 'examples', etc.


def load_environment_variables() -> Dict[str, str]:
    """
    Loads and validates required environment variables, returning them in a dictionary.
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise ValueError("Set the OPENAI_API_KEY environment variable.")

    # Return any additional environment variables here as needed
    return {
        "OPENAI_API_KEY": openai_api_key
    }


def _load_json_file(file_path: str) -> Any:
    """
    Internal helper function to load JSON content from a file.
    
    Args:
        file_path (str): Path to the JSON file.

    Returns:
        Any: The parsed JSON data.
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

    Args:
        file_path (str): Path to the text file.
        description (str): Description for logging purposes.

    Returns:
        str: The file content as a stripped string.
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


def load_config(config_file_path: str) -> ConfigModel:
    """
    Loads and validates the main configuration from a JSON file using Pydantic.
    
    Args:
        config_file_path (str): Path to the configuration file.
        
    Returns:
        ConfigModel: An instance of the validated ConfigModel.
    """
    try:
        raw_config = _load_json_file(config_file_path)
        logger.debug(f"Raw config loaded from '{config_file_path}'.")
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load config '{config_file_path}': {e}")
        raise

    # Validate using Pydantic
    try:
        validated_config = ConfigModel.model_validate(raw_config)
        logger.debug(f"Configuration validated successfully for '{config_file_path}'.")
        return validated_config
    except ValidationError as ve:
        logger.error(f"Pydantic validation error for '{config_file_path}': {ve}")
        raise


def load_data_format_config(config_file_path: str) -> DataFormatConfig:
    """
    Loads and validates the data format configuration from a JSON file using Pydantic.
    
    Args:
        config_file_path (str): Path to the data_format_config.json file.
        
    Returns:
        DataFormatConfig: An instance of the validated DataFormatConfig.
    """
    try:
        raw_config = _load_json_file(config_file_path)
        logger.debug(f"Raw data_format_config loaded from '{config_file_path}'.")
    except (FileNotFoundError, json.JSONDecodeError, OSError) as e:
        logger.error(f"Failed to load data format config '{config_file_path}': {e}")
        raise

    # Validate using Pydantic
    try:
        validated_config = DataFormatConfig.model_validate(raw_config)
        logger.debug(f"Data format configuration validated for '{config_file_path}'.")
        return validated_config
    except ValidationError as ve:
        logger.error(f"Pydantic validation error in data format config '{config_file_path}': {ve}")
        raise


def load_prompt_file(prompts_folder: str, prompt_file: str, description: str = 'prompt') -> str:
    """
    Loads a prompt from a specified file.

    Args:
        prompts_folder (str): Directory where prompt files are stored.
        prompt_file (str): Name of the prompt file.
        description (str): Description of the prompt (for logging purposes).

    Returns:
        str: The prompt content as a string.
    """
    prompt_path = str(Path(prompts_folder) / prompt_file)
    return _load_text_file(prompt_path, description=description)


def initialize_deductive_resources(
    codebase_folder: str,
    prompts_folder: str,
    selected_codebase: str,
    deductive_prompt_file: str
) -> Tuple[List[ProcessedCodeItem], str]:
    """
    Initializes resources needed for deductive coding: loads code instructions and the full codebase.

    Args:
        codebase_folder (str): Directory where the codebase files are stored.
        prompts_folder (str): Directory where prompt files are stored.
        selected_codebase (str): Specific codebase file to use.
        deductive_prompt_file (str): Name of the deductive coding prompt file.

    Returns:
        Tuple[List[ProcessedCodeItem], str]: A list of processed code items and the coding instructions.
    """
    # Load coding instructions for deductive coding
    try:
        coding_instructions = load_prompt_file(prompts_folder, deductive_prompt_file, description='deductive coding prompt')
        logger.debug("Coding instructions loaded for deductive coding.")
    except (FileNotFoundError, ValueError, OSError) as e:
        logger.error(f"Failed to load coding instructions: {e}")
        raise

    # Load processed codes from specified codebase file
    list_of_codes_file = Path(codebase_folder) / selected_codebase
    if not list_of_codes_file.exists():
        logger.error(f"List of codes file '{list_of_codes_file}' not found.")
        raise FileNotFoundError(f"List of codes file '{list_of_codes_file}' not found.")

    try:
        with list_of_codes_file.open('r', encoding='utf-8') as file:
            processed_codes: List[ProcessedCodeItem] = [
                json.loads(line) for line in file if line.strip()
            ]
        logger.debug(f"Loaded {len(processed_codes)} codes from '{list_of_codes_file}'.")
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"An error occurred while loading codes from '{list_of_codes_file}': {e}")
        raise

    return processed_codes, coding_instructions
