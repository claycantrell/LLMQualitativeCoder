import os
import json
import logging
from typing import Dict, Any, Tuple, List, Type, Optional
from qual_functions import (
    initialize_faiss_index_from_formatted_file
)
from pydantic import create_model, BaseModel, ConfigDict

# Configure logging for utils module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)  # Ensure DEBUG logs are captured

def load_environment_variables() -> None:
    """
    Loads and validates required environment variables.
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise ValueError("Set the OPENAI_API_KEY environment variable.")

def load_coding_instructions(prompts_folder: str) -> str:
    """
    Loads coding instructions for deductive coding from a file.
    """
    coding_instructions_file = os.path.join(prompts_folder, 'coding_prompt.txt')
    if not os.path.exists(coding_instructions_file):
        logger.error(f"Coding instructions file '{coding_instructions_file}' not found.")
        raise FileNotFoundError(f"Coding instructions file '{coding_instructions_file}' not found.")

    try:
        with open(coding_instructions_file, 'r', encoding='utf-8') as file:
            coding_instructions = file.read().strip()
        if not coding_instructions:
            logger.error("Coding instructions file is empty.")
            raise ValueError("Coding instructions file is empty.")
    except Exception as e:
        logger.error(f"Error reading coding instructions file '{coding_instructions_file}': {e}")
        raise

    return coding_instructions

def load_parse_instructions(prompts_folder: str) -> str:
    """
    Loads parse instructions from a file for breaking down speaking turns into meaning units.
    """
    parse_prompt_file = os.path.join(prompts_folder, 'parse_prompt.txt')
    if not os.path.exists(parse_prompt_file):
        logger.error(f"Parse instructions file '{parse_prompt_file}' not found.")
        raise FileNotFoundError(f"Parse instructions file '{parse_prompt_file}' not found.")

    try:
        with open(parse_prompt_file, 'r', encoding='utf-8') as file:
            parse_instructions = file.read().strip()
    except Exception as e:
        logger.error(f"Error reading parse instructions file '{parse_prompt_file}': {e}")
        raise

    if not parse_instructions:
        logger.error("Parse instructions file is empty.")
        raise ValueError("Parse instructions file is empty.")

    return parse_instructions

def load_custom_coding_prompt(prompts_folder: str) -> str:
    """
    Loads a custom coding prompt for inductive coding from a file.
    """
    custom_coding_prompt_file = os.path.join(prompts_folder, 'custom_coding_prompt.txt')
    if not os.path.exists(custom_coding_prompt_file):
        logger.error(f"Custom coding prompt file '{custom_coding_prompt_file}' not found.")
        raise FileNotFoundError(f"Custom coding prompt file '{custom_coding_prompt_file}' not found.")

    try:
        with open(custom_coding_prompt_file, 'r', encoding='utf-8') as file:
            custom_coding_prompt = file.read().strip()
    except Exception as e:
        logger.error(f"Error reading custom coding prompt file '{custom_coding_prompt_file}': {e}")
        raise

    if not custom_coding_prompt:
        logger.error("Custom coding prompt file is empty.")
        raise ValueError("Custom coding prompt file is empty.")

    return custom_coding_prompt

def initialize_deductive_resources(
    codebase_folder: str,
    prompts_folder: str,
    initialize_embedding_model: str,
    use_rag: bool
) -> Tuple[List[Dict[str, Any]], Optional[Any], str]:
    """
    Initializes resources needed for deductive coding: loads code instructions, codebase, and builds a FAISS index if use_rag is True.
    Returns processed_codes, faiss_index (or None), and coding_instructions.
    """
    # Load coding instructions for deductive coding
    try:
        coding_instructions = load_coding_instructions(prompts_folder)
        logger.debug("Coding instructions loaded for deductive coding.")
    except Exception as e:
        logger.error(f"Failed to load coding instructions: {e}")
        raise

    # Load processed codes from .jsonl file
    list_of_codes_file = os.path.join(codebase_folder, 'new_schema.jsonl')  # Ensure the file exists
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

def load_schema_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads schema configuration from the given JSON file path.
    """
    if not os.path.exists(config_path):
        logger.error(f"Schema configuration file '{config_path}' not found.")
        raise FileNotFoundError(f"Schema configuration file '{config_path}' not found.")

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = json.load(file)
        logger.debug(f"Schema configuration loaded from '{config_path}'.")
        return config_data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{config_path}': {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading schema configuration: {e}")
        raise

def create_dynamic_model_for_format(data_format: str, schema_config: Dict[str, Dict[str, Any]]) -> Tuple[Type[BaseModel], str]:
    """
    Creates a dynamic Pydantic model for the given data format based on the provided schema configuration.
    """
    if data_format not in schema_config:
        raise ValueError(f"No schema configuration found for data format '{data_format}'")

    format_config = schema_config[data_format]
    if 'fields' not in format_config or 'content_field' not in format_config:
        raise ValueError(f"Configuration for data format '{data_format}' must include 'fields' and 'content_field'.")

    fields = format_config["fields"]
    content_field = format_config["content_field"]

    type_map = {
        "str": str,
        "int": int,
        "float": float,
        "list": list,
        "dict": dict,
        "bool": bool,
        # Additional mappings as needed
    }

    dynamic_fields = {}
    for field_name, field_type_str in fields.items():
        py_type = type_map.get(field_type_str, Any)  # Default to Any if not found
        dynamic_fields[field_name] = (py_type, ...)  # Required field by default

    # Use Pydantic v2's configuration for handling extra fields
    model_config = ConfigDict(extra='allow')

    try:
        dynamic_model = create_model(
            f"{data_format.capitalize()}DataModel",
            __base__=BaseModel,
            **dynamic_fields,
            model_config=model_config  # For Pydantic v2, to allow extra fields
        )
        logger.debug(f"Dynamic Pydantic model '{data_format.capitalize()}DataModel' created with content_field '{content_field}'.")
    except Exception as e:
        logger.error(f"Failed to create dynamic Pydantic model for '{data_format}': {type(e).__name__}: {e}")
        raise

    return dynamic_model, content_field

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
