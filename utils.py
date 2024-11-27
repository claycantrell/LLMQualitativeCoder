# utils.py

import os
import json
import logging
from typing import Dict, Any, Tuple, List, Optional

from qual_functions import (
    initialize_faiss_index_from_formatted_file
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs; adjust as needed

handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

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
    if not os.path.exists(config_file_path):
        logger.error(f"Configuration file '{config_file_path}' not found.")
        raise FileNotFoundError(f"Configuration file '{config_file_path}' not found.")

    try:
        with open(config_file_path, 'r', encoding='utf-8') as file:
            config = json.load(file)
        logger.debug(f"Configuration loaded from '{config_file_path}'.")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{config_file_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        raise

def load_data_format_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads data format configuration from the given JSON file path.

    Args:
        config_path (str): Path to the data format configuration file.

    Returns:
        Dict[str, Dict[str, Any]]: Data format configuration dictionary.
    """
    if not os.path.exists(config_path):
        logger.error(f"Data format configuration file '{config_path}' not found.")
        raise FileNotFoundError(f"Data format configuration file '{config_path}' not found.")

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
        logger.debug(f"Data format configuration loaded from '{config_path}'.")
        return config_data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{config_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data format configuration: {e}")
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
    prompt_path = os.path.join(prompts_folder, prompt_file)
    if not os.path.exists(prompt_path):
        logger.error(f"{description.capitalize()} file '{prompt_path}' not found.")
        raise FileNotFoundError(f"{description.capitalize()} file '{prompt_path}' not found.")

    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            prompt_content = file.read().strip()
    except Exception as e:
        logger.error(f"Error reading {description} file '{prompt_path}': {e}")
        raise

    if not prompt_content:
        logger.error(f"{description.capitalize()} file is empty.")
        raise ValueError(f"{description.capitalize()} file is empty.")

    return prompt_content

def initialize_deductive_resources(
    codebase_folder: str,
    prompts_folder: str,
    initialize_embedding_model: str,
    use_rag: bool,
    selected_codebase: str,
    deductive_prompt_file: str
) -> Tuple[List[Dict[str, Any]], Optional[Any], str]:
    """
    Initializes resources needed for deductive coding: loads code instructions, codebase, and builds a FAISS index if use_rag is True.
    Returns processed_codes, faiss_index (or None), and coding_instructions.

    Args:
        codebase_folder (str): Directory where the codebase files are stored.
        prompts_folder (str): Directory where prompt files are stored.
        initialize_embedding_model (str): Embedding model to use for FAISS.
        use_rag (bool): Whether to use Retrieval-Augmented Generation.
        selected_codebase (str): Specific codebase file to use.
        deductive_prompt_file (str): Name of the deductive coding prompt file.

    Returns:
        Tuple[List[Dict[str, Any]], Optional[Any], str]: Processed codes, FAISS index, and coding instructions.
    """
    # Load coding instructions for deductive coding
    try:
        coding_instructions = load_prompt_file(prompts_folder, deductive_prompt_file, description='deductive coding prompt')
        logger.debug("Coding instructions loaded for deductive coding.")
    except Exception as e:
        logger.error(f"Failed to load coding instructions: {e}")
        raise

    # Load processed codes from specified codebase file
    list_of_codes_file = os.path.join(codebase_folder, selected_codebase)
    if not os.path.exists(list_of_codes_file):
        logger.error(f"List of codes file '{list_of_codes_file}' not found.")
        raise FileNotFoundError(f"List of codes file '{list_of_codes_file}' not found.")

    try:
        with open(list_of_codes_file, 'r', encoding='utf-8') as file:
            processed_codes = [json.loads(line) for line in file if line.strip()]
        logger.debug(f"Loaded {len(processed_codes)} codes from '{list_of_codes_file}'.")
    except Exception as e:
        logger.error(f"An error occurred while loading codes from '{list_of_codes_file}': {e}")
        raise

    # Initialize FAISS index if use_rag is True
    if use_rag:
        try:
            faiss_index, _ = initialize_faiss_index_from_formatted_file(
                codes_list_file=list_of_codes_file,
                embedding_model=initialize_embedding_model
            )
            logger.debug("FAISS index initialized with processed codes.")
        except Exception as e:
            logger.error(f"Failed to initialize FAISS index: {e}")
            raise
    else:
        faiss_index = None  # FAISS index is not needed

    return processed_codes, faiss_index, coding_instructions
