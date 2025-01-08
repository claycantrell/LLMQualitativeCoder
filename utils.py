# utils.py

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List, TypedDict, Optional

logger = logging.getLogger(__name__)

class ProcessedCodeItem(TypedDict, total=False):
    """
    Example TypedDict for code items loaded from a codebase JSONL file.
    Customize fields based on your actual data structure.
    """
    code_name: str
    code_description: str
    # Add other known fields here, e.g. 'parent_code', 'examples', etc.

def load_environment_variables() -> None:
    """
    Loads and validates required environment variables.
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise ValueError("Set the OPENAI_API_KEY environment variable.")

def load_config(config_file_path: str) -> Dict[str, Any]:
    """
    Loads the configuration from a JSON file.

    Args:
        config_file_path (str): Path to the configuration file.

    Returns:
        Dict[str, Any]: Configuration dictionary.
    """
    config_path = Path(config_file_path)
    if not config_path.exists():
        logger.error(f"Configuration file '{config_file_path}' not found.")
        raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")

    try:
        with config_path.open('r', encoding='utf-8') as file:
            config = json.load(file)
        logger.debug(f"Configuration loaded from '{config_file_path}'.")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{config_file_path}': {e}")
        raise
    except OSError as e:
        logger.error(f"OS error loading configuration from '{config_file_path}': {e}")
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
    prompt_path = Path(prompts_folder) / prompt_file
    if not prompt_path.exists():
        logger.error(f"{description.capitalize()} file '{prompt_path}' not found.")
        raise FileNotFoundError(f"{description.capitalize()} file '{prompt_path}' not found.")

    try:
        with prompt_path.open('r', encoding='utf-8') as file:
            prompt_content = file.read().strip()
    except OSError as e:
        logger.error(f"Error reading {description} file '{prompt_path}': {e}")
        raise

    if not prompt_content:
        logger.error(f"{description.capitalize()} file is empty.")
        raise ValueError(f"{description.capitalize()} file is empty.")

    return prompt_content

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
