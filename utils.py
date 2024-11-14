# utils.py

import os
import json
import logging
from typing import List, Dict, Any, Tuple
import faiss
from qual_functions import (
    parse_transcript,
    initialize_faiss_index_from_formatted_file
)

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


def load_coding_instructions(prompts_folder: str) -> str:
    """
    Loads coding instructions for deductive coding from a file.
    """
    coding_instructions_file = os.path.join(prompts_folder, 'coding_prompt.txt')
    if not os.path.exists(coding_instructions_file):
        logger.error(f"Coding instructions file '{coding_instructions_file}' not found.")
        raise FileNotFoundError(f"Coding instructions file '{coding_instructions_file}' not found.")

    with open(coding_instructions_file, 'r', encoding='utf-8') as file:
        coding_instructions = file.read().strip()

    return coding_instructions


def load_parse_instructions(prompts_folder: str) -> str:
    """
    Loads parse instructions from a file for breaking down speaking turns into meaning units.
    """
    parse_prompt_file = os.path.join(prompts_folder, 'parse_prompt.txt')
    if not os.path.exists(parse_prompt_file):
        logger.error(f"Parse instructions file '{parse_prompt_file}' not found.")
        raise FileNotFoundError(f"Parse instructions file '{parse_prompt_file}' not found.")

    with open(parse_prompt_file, 'r', encoding='utf-8') as file:
        parse_instructions = file.read().strip()

    return parse_instructions


def load_custom_coding_prompt(prompts_folder: str) -> str:
    """
    Loads a custom coding prompt for inductive coding from a file.
    """
    custom_coding_prompt_file = os.path.join(prompts_folder, 'custom_coding_prompt.txt')
    if not os.path.exists(custom_coding_prompt_file):
        logger.error(f"Custom coding prompt file '{custom_coding_prompt_file}' not found.")
        raise FileNotFoundError(f"Custom coding prompt file '{custom_coding_prompt_file}' not found.")

    with open(custom_coding_prompt_file, 'r', encoding='utf-8') as file:
        custom_coding_prompt = file.read().strip()

    if not custom_coding_prompt:
        logger.error("Custom coding prompt file is empty.")
        raise ValueError("Custom coding prompt file is empty.")

    return custom_coding_prompt


def initialize_deductive_resources(
    codebase_folder: str,
    prompts_folder: str,
    initialize_embedding_model: str
) -> Tuple[List[Dict[str, Any]], faiss.IndexFlatL2, str]:
    """
    Initializes resources needed for deductive coding: loads code instructions, codebase, and builds a FAISS index.
    Returns processed_codes, faiss_index, and coding_instructions.
    """
    # Load coding instructions for deductive coding
    coding_instructions = load_coding_instructions(prompts_folder)

    # Initialize FAISS index and get processed codes
    list_of_codes_file = os.path.join(codebase_folder, 'new_schema.txt')
    if not os.path.exists(list_of_codes_file):
        logger.error(f"List of codes file '{list_of_codes_file}' not found.")
        raise FileNotFoundError(f"List of codes file '{list_of_codes_file}' not found.")

    faiss_index, processed_codes = initialize_faiss_index_from_formatted_file(
        list_of_codes_file,
        embedding_model=initialize_embedding_model
    )

    if not processed_codes:
        logger.warning(f"No codes found in '{list_of_codes_file}' or failed to process correctly.")
        processed_codes = []

    return processed_codes, faiss_index, coding_instructions
