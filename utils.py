# utils.py
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

def load_coding_instructions(prompts_folder: str, prompt_file: str) -> str:
    """
    Loads coding instructions from a specified prompt file.

    Args:
        prompts_folder (str): Directory where prompt files are stored.
        prompt_file (str): Name of the prompt file.

    Returns:
        str: Coding instructions as a string.
    """
    coding_instructions_file = os.path.join(prompts_folder, prompt_file)
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

def load_parse_instructions(prompts_folder: str, parse_prompt_file: str) -> str:
    """
    Loads parse instructions from a specified prompt file for breaking down speaking turns into meaning units.

    Args:
        prompts_folder (str): Directory where prompt files are stored.
        parse_prompt_file (str): Name of the parse prompt file.

    Returns:
        str: Parse instructions as a string.
    """
    parse_prompt_path = os.path.join(prompts_folder, parse_prompt_file)
    if not os.path.exists(parse_prompt_path):
        logger.error(f"Parse instructions file '{parse_prompt_path}' not found.")
        raise FileNotFoundError(f"Parse instructions file '{parse_prompt_path}' not found.")

    try:
        with open(parse_prompt_path, 'r', encoding='utf-8') as file:
            parse_instructions = file.read().strip()
    except Exception as e:
        logger.error(f"Error reading parse instructions file '{parse_prompt_path}': {e}")
        raise

    if not parse_instructions:
        logger.error("Parse instructions file is empty.")
        raise ValueError("Parse instructions file is empty.")

    return parse_instructions

def load_inductive_coding_prompt(prompts_folder: str, inductive_prompt_file: str) -> str:
    """
    Loads the inductive coding prompt from a specified file.

    Args:
        prompts_folder (str): Directory where prompt files are stored.
        inductive_prompt_file (str): Name of the inductive coding prompt file.

    Returns:
        str: Inductive coding prompt as a string.
    """
    inductive_coding_prompt_path = os.path.join(prompts_folder, inductive_prompt_file)
    if not os.path.exists(inductive_coding_prompt_path):
        logger.error(f"Inductive coding prompt file '{inductive_coding_prompt_path}' not found.")
        raise FileNotFoundError(f"Inductive coding prompt file '{inductive_coding_prompt_path}' not found.")

    try:
        with open(inductive_coding_prompt_path, 'r', encoding='utf-8') as file:
            inductive_coding_prompt = file.read().strip()
    except Exception as e:
        logger.error(f"Error reading inductive coding prompt file '{inductive_coding_prompt_path}': {e}")
        raise

    if not inductive_coding_prompt:
        logger.error("Inductive coding prompt file is empty.")
        raise ValueError("Inductive coding prompt file is empty.")

    return inductive_coding_prompt

def load_deductive_coding_prompt(prompts_folder: str, deductive_prompt_file: str) -> str:
    """
    Loads the deductive coding prompt from a specified file.

    Args:
        prompts_folder (str): Directory where prompt files are stored.
        deductive_prompt_file (str): Name of the deductive coding prompt file.

    Returns:
        str: Deductive coding prompt as a string.
    """
    deductive_coding_prompt_path = os.path.join(prompts_folder, deductive_prompt_file)
    if not os.path.exists(deductive_coding_prompt_path):
        logger.error(f"Deductive coding prompt file '{deductive_coding_prompt_path}' not found.")
        raise FileNotFoundError(f"Deductive coding prompt file '{deductive_coding_prompt_path}' not found.")

    try:
        with open(deductive_coding_prompt_path, 'r', encoding='utf-8') as file:
            deductive_coding_prompt = file.read().strip()
    except Exception as e:
        logger.error(f"Error reading deductive coding prompt file '{deductive_coding_prompt_path}': {e}")
        raise

    if not deductive_coding_prompt:
        logger.error("Deductive coding prompt file is empty.")
        raise ValueError("Deductive coding prompt file is empty.")

    return deductive_coding_prompt

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
        coding_instructions = load_coding_instructions(prompts_folder, deductive_prompt_file)
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

def load_schema_config(config_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Loads schema configuration from the given JSON file path.

    Args:
        config_path (str): Path to the schema configuration file.

    Returns:
        Dict[str, Dict[str, Any]]: Schema configuration dictionary.
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

def create_dynamic_models_for_format(data_format: str, schema_config: Dict[str, Dict[str, Any]]) -> Tuple[Type[BaseModel], Optional[Type[BaseModel]], str, Optional[str]]:
    """
    Creates dynamic Pydantic models for the given data format and item format based on the provided schema configuration.

    Args:
        data_format (str): The format of the data (e.g., "movie_script").
        schema_config (Dict[str, Dict[str, Any]]): Schema configuration for different data formats.

    Returns:
        Tuple[Type[BaseModel], Optional[Type[BaseModel]], str, Optional[str]]: The dynamic main data model, the item model, the content field name, and the speaker field name.
    """
    if data_format not in schema_config:
        raise ValueError(f"No schema configuration found for data format '{data_format}'")

    format_config = schema_config[data_format]
    if 'fields' not in format_config or 'content_field' not in format_config:
        raise ValueError(f"Configuration for data format '{data_format}' must include 'fields' and 'content_field'.")

    fields = format_config["fields"]
    content_field = format_config["content_field"]
    speaker_field = format_config.get("speaker_field")
    list_field = format_config.get("list_field")

    type_map = {
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "Any": Any,
        # Additional mappings as needed
    }

    def create_model_fields(fields_dict, model_name_prefix=""):
        dynamic_fields = {}
        for field_name, field_value in fields_dict.items():
            is_optional = False
            if isinstance(field_value, dict):
                is_optional = field_value.get("optional", False)
                field_type = field_value.get("type", "dict")
                if field_type == "dict":
                    # Nested dict
                    nested_fields = field_value.get("fields", {})
                    nested_model_name = f"{model_name_prefix}{field_name.capitalize()}Model"
                    nested_model_fields = create_model_fields(nested_fields, model_name_prefix=nested_model_name)
                    nested_model = create_model(
                        nested_model_name,
                        __base__=BaseModel,
                        **nested_model_fields
                    )
                    if is_optional:
                        dynamic_fields[field_name] = (Optional[nested_model], None)
                    else:
                        dynamic_fields[field_name] = (nested_model, ...)
                elif field_type == "list":
                    # List of items
                    item_type_value = field_value.get("items", "Any")
                    if isinstance(item_type_value, str) and item_type_value in schema_config:
                        # Reference to another format (e.g., 'script_entry')
                        item_format_config = schema_config[item_type_value]
                        item_fields = item_format_config["fields"]
                        item_model_name = f"{model_name_prefix}{field_name.capitalize()}ItemModel"
                        item_model_fields = create_model_fields(item_fields, model_name_prefix=item_model_name)
                        item_model = create_model(
                            item_model_name,
                            __base__=BaseModel,
                            **item_model_fields
                        )
                        list_type = List[item_model]
                        if is_optional:
                            dynamic_fields[field_name] = (Optional[list_type], None)
                        else:
                            dynamic_fields[field_name] = (list_type, ...)
                    else:
                        # List of simple types
                        item_py_type = type_map.get(item_type_value, Any)
                        list_type = List[item_py_type]
                        if is_optional:
                            dynamic_fields[field_name] = (Optional[list_type], None)
                        else:
                            dynamic_fields[field_name] = (list_type, ...)
                else:
                    # Simple type (e.g., 'str')
                    py_type = type_map.get(field_type, Any)
                    if is_optional:
                        dynamic_fields[field_name] = (Optional[py_type], None)
                    else:
                        dynamic_fields[field_name] = (py_type, ...)
            elif isinstance(field_value, str):
                # Simple type
                py_type = type_map.get(field_value, Any)
                dynamic_fields[field_name] = (py_type, ...)
            else:
                # Unrecognized field value
                py_type = Any
                dynamic_fields[field_name] = (py_type, ...)
        return dynamic_fields

    # Create the main data model
    dynamic_fields = create_model_fields(fields, model_name_prefix=f"{data_format.capitalize()}")

    # Use Pydantic v2's configuration for handling extra fields
    model_config = ConfigDict(extra='allow')

    try:
        dynamic_model = create_model(
            f"{data_format.capitalize()}DataModel",
            __base__=BaseModel,
            **dynamic_fields,
            model_config=model_config  # For Pydantic v2, to allow extra fields
        )
        logger.debug(f"Dynamic Pydantic model '{data_format.capitalize()}DataModel' created.")
    except Exception as e:
        logger.error(f"Failed to create dynamic Pydantic model for '{data_format}': {type(e).__name__}: {e}")
        raise

    # Create the item model if needed
    item_model = None
    if 'fields' in format_config and list_field:
        list_field_parts = list_field.split('.')
        current_fields = fields
        for part in list_field_parts:
            if part in current_fields:
                current_field = current_fields[part]
                if isinstance(current_field, dict):
                    field_type = current_field.get("type")
                    if field_type == "list":
                        item_type_value = current_field.get("items")
                        if isinstance(item_type_value, str) and item_type_value in schema_config:
                            # Create item model
                            item_format_config = schema_config[item_type_value]
                            item_fields = item_format_config['fields']
                            item_model_fields = create_model_fields(item_fields, model_name_prefix=item_type_value.capitalize())
                            try:
                                item_model = create_model(
                                    f"{item_type_value.capitalize()}Model",
                                    __base__=BaseModel,
                                    **item_model_fields,
                                    model_config=model_config
                                )
                                logger.debug(f"Dynamic Pydantic item model '{item_type_value.capitalize()}Model' created.")
                            except Exception as e:
                                logger.error(f"Failed to create dynamic Pydantic item model for '{item_type_value}': {type(e).__name__}: {e}")
                                raise
                        else:
                            item_model = None
                    elif field_type == "dict":
                        current_fields = current_field.get("fields", {})
                    else:
                        item_model = None
                else:
                    item_model = None

    return dynamic_model, item_model, content_field, speaker_field
