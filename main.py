# main.py

import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple  # Added Any and Tuple
from qual_functions import (
    MeaningUnit,
    parse_transcript,
    assign_codes_to_meaning_units,
    initialize_faiss_index_from_formatted_file,
    CodeAssigned
)
import faiss  # Ensure faiss is imported if used

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_environment_variables() -> None:
    """
    Loads and validates required environment variables.
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise ValueError("Set the OPENAI_API_KEY environment variable.")

def load_transcript_data(json_folder: str) -> List[dict]:
    """
    Loads and returns JSON data from the transcript file.
    """
    json_transcript_file = os.path.join(json_folder, 'output_cues.json')
    if not os.path.exists(json_transcript_file):
        logger.error(f"JSON file '{json_transcript_file}' not found.")
        raise FileNotFoundError(f"JSON file '{json_transcript_file}' not found.")

    try:
        with open(json_transcript_file, 'r', encoding='utf-8') as file:
            json_data = json.load(file)
            logger.info(f"Loaded {len(json_data)} speaking turns from '{json_transcript_file}'.")
            return json_data
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from '{json_transcript_file}': {e}")
        raise

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

def parse_speaking_turns(
    json_data: List[dict],
    prompts_folder: str,
    coding_mode: str,
    use_parsing: bool,
    parse_model: str
) -> List[MeaningUnit]:
    """
    Parses speaking turns into meaning units based on the coding mode and parsing option.
    Returns a list of MeaningUnit objects.
    """
    meaning_unit_object_list = []

    # Load parse prompt instructions if needed
    parse_instructions = ""
    parse_prompt_file = os.path.join(prompts_folder, 'parse_prompt.txt')
    if coding_mode == "deductive" or (coding_mode == "inductive" and use_parsing):
        if not os.path.exists(parse_prompt_file):
            logger.error(f"Parse instructions file '{parse_prompt_file}' not found.")
            raise FileNotFoundError(f"Parse instructions file '{parse_prompt_file}' not found.")

        with open(parse_prompt_file, 'r', encoding='utf-8') as file:
            parse_instructions = file.read().strip()

    for idx, speaking_turn in enumerate(json_data, start=1):
        speaker_id = speaking_turn.get('speaker_name', 'Unknown')
        speaking_turn_string = speaking_turn.get('text_context', '')
        if not speaking_turn_string:
            logger.warning(f"No speaking turn text found for Speaking Turn {idx}. Skipping.")
            continue

        if coding_mode == "deductive":
            if use_parsing:
                # Parsing speaking turns for deductive coding
                logger.info(f"Processing Speaking Turn {idx} (Parsing for Deductive Coding): Speaker - {speaker_id}")
                formatted_prompt = parse_instructions.replace("{speaker_name}", speaker_id)
                meaning_unit_list = parse_transcript(
                    speaking_turn_string,
                    formatted_prompt,
                    completion_model=parse_model
                )
                if not meaning_unit_list:
                    logger.warning(f"No meaning units extracted from Speaking Turn {idx}. Skipping.")
                    continue

                for unit_idx, unit in enumerate(meaning_unit_list, start=1):
                    meaning_unit_object = MeaningUnit(
                        speaker_id=unit.get('speaker_id', speaker_id),
                        meaning_unit_string=unit.get('meaning_unit_string', '')
                    )
                    logger.debug(f"Added Meaning Unit {unit_idx} (Deductive/Parsed): Speaker - {meaning_unit_object.speaker_id}, "
                                 f"Quote - {meaning_unit_object.meaning_unit_string}")
                    meaning_unit_object_list.append(meaning_unit_object)
            else:
                # Using entire speaking turns as meaning units for deductive coding
                logger.info(f"Processing Speaking Turn {idx} (Deductive Coding without Parsing): Speaker - {speaker_id}")
                meaning_unit_object = MeaningUnit(
                    speaker_id=speaker_id,
                    meaning_unit_string=speaking_turn_string
                )
                logger.debug(f"Added Meaning Unit (Deductive/Unparsed): Speaker - {meaning_unit_object.speaker_id}, "
                             f"Quote - {meaning_unit_object.meaning_unit_string}")
                meaning_unit_object_list.append(meaning_unit_object)
        else:  # Inductive mode
            if use_parsing:
                # Parsing speaking turns for inductive coding
                logger.info(f"Processing Speaking Turn {idx} (Parsing for Inductive Coding): Speaker - {speaker_id}")
                formatted_prompt = parse_instructions.replace("{speaker_name}", speaker_id)
                meaning_unit_list = parse_transcript(
                    speaking_turn_string,
                    formatted_prompt,
                    completion_model=parse_model
                )
                if not meaning_unit_list:
                    logger.warning(f"No meaning units extracted from Speaking Turn {idx}. Skipping.")
                    continue

                for unit_idx, unit in enumerate(meaning_unit_list, start=1):
                    meaning_unit_object = MeaningUnit(
                        speaker_id=unit.get('speaker_id', speaker_id),
                        meaning_unit_string=unit.get('meaning_unit_string', '')
                    )
                    logger.debug(f"Added Meaning Unit {unit_idx} (Inductive/Parsed): Speaker - {meaning_unit_object.speaker_id}, "
                                 f"Quote - {meaning_unit_object.meaning_unit_string}")
                    meaning_unit_object_list.append(meaning_unit_object)
            else:
                # Using entire speaking turns as meaning units for inductive coding
                logger.info(f"Processing Speaking Turn {idx} (Inductive Coding without Parsing): Speaker - {speaker_id}")
                meaning_unit_object = MeaningUnit(
                    speaker_id=speaker_id,
                    meaning_unit_string=speaking_turn_string
                )
                logger.debug(f"Added Meaning Unit (Inductive/Unparsed): Speaker - {meaning_unit_object.speaker_id}, "
                             f"Quote - {meaning_unit_object.meaning_unit_string}")
                meaning_unit_object_list.append(meaning_unit_object)

    if not meaning_unit_object_list:
        logger.warning("No meaning units extracted from any speaking turns.")
    return meaning_unit_object_list

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

def main(
    coding_mode: str = "deductive",
    use_parsing: bool = True,
    use_rag: bool = True,
    parse_model: str = "gpt-4o-mini",
    assign_model: str = "gpt-4o-mini",
    initialize_embedding_model: str = "text-embedding-3-small",
    retrieve_embedding_model: str = "text-embedding-3-small"
):
    """
    Orchestrates the entire process of assigning qualitative codes to transcripts based on the provided modes and configurations.
    """
    # Validate coding mode and load environment variables
    load_environment_variables()
    logger.debug("Environment variables loaded and validated.")

    # Define paths
    prompts_folder = 'prompts'
    codebase_folder = 'qual_codebase'
    json_folder = 'json_transcripts'

    # Load transcript data
    json_data = load_transcript_data(json_folder)

    # Build meaning units
    meaning_unit_object_list = parse_speaking_turns(
        json_data=json_data,
        prompts_folder=prompts_folder,
        coding_mode=coding_mode,
        use_parsing=use_parsing,
        parse_model=parse_model
    )
    if not meaning_unit_object_list:
        logger.warning("No meaning units to process. Exiting.")
        return

    # Handle inductive or deductive coding mode
    if coding_mode == "deductive":
        processed_codes, faiss_index, coding_instructions = initialize_deductive_resources(
            codebase_folder=codebase_folder,
            prompts_folder=prompts_folder,
            initialize_embedding_model=initialize_embedding_model
        )

        if not processed_codes:
            logger.warning("No processed codes available for deductive coding. Exiting.")
            return

        # Assign codes to meaning units in deductive mode
        if use_rag:
            # Deductive coding with RAG
            coded_meaning_unit_list = assign_codes_to_meaning_units(
                meaning_unit_list=meaning_unit_object_list,
                coding_instructions=coding_instructions,
                processed_codes=processed_codes,
                index=faiss_index,
                top_k=5,
                context_size=5,
                use_rag=True,
                completion_model=assign_model,
                embedding_model=retrieve_embedding_model
            )
        else:
            # Deductive coding without RAG (using full codebase)
            coded_meaning_unit_list = assign_codes_to_meaning_units(
                meaning_unit_list=meaning_unit_object_list,
                coding_instructions=coding_instructions,
                processed_codes=processed_codes,
                index=faiss_index,
                top_k=5,
                context_size=5,
                use_rag=False,
                codebase=processed_codes,
                completion_model=assign_model,
                embedding_model=retrieve_embedding_model
            )

    else:  # Inductive coding
        # Load custom coding prompt
        coding_instructions = load_custom_coding_prompt(prompts_folder)

        # Assign codes to meaning units in inductive mode
        coded_meaning_unit_list = assign_codes_to_meaning_units(
            meaning_unit_list=meaning_unit_object_list,
            coding_instructions=coding_instructions,
            processed_codes=None,
            index=None,
            top_k=None,
            context_size=5,
            use_rag=False,
            codebase=None,
            completion_model=assign_model,
            embedding_model=None  # Embeddings not needed in inductive mode
        )

    # Output coded meaning units
    for unit in coded_meaning_unit_list:
        logger.info(f"\nID: {unit.unique_id}")
        logger.info(f"Speaker: {unit.speaker_id}")
        logger.info(f"Quote: {unit.meaning_unit_string}")
        if unit.assigned_code_list:
            for code in unit.assigned_code_list:
                logger.info(f"  Code: {code.code_name}")
                logger.info(f"  Justification: {code.code_justification}")
        else:
            logger.info("  No codes assigned.")

if __name__ == "__main__":
    # Example usage:

    # Deductive Coding with Parsing and RAG
    # main(
    #     coding_mode="deductive",
    #     use_parsing=True,
    #     use_rag=True,
    #     parse_model="gpt-4o-mini",
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",
    #     retrieve_embedding_model="text-embedding-3-small"
    # )

    # Deductive Coding without Parsing and without RAG
    main(
        coding_mode="deductive",
        use_parsing=False,
        use_rag=False,
        parse_model="gpt-4o-mini",
        assign_model="gpt-4o-mini",
        initialize_embedding_model="text-embedding-3-small",
        retrieve_embedding_model="text-embedding-3-small"
    )

    # Inductive Coding with Parsing
    # main(
    #     coding_mode="inductive",
    #     use_parsing=True,  # Enable parsing for inductive coding
    #     use_rag=False,     # RAG not applicable in inductive mode
    #     parse_model="gpt-4o-mini",  # Relevant only if use_parsing=True
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",  # Irrelevant in inductive mode
    #     retrieve_embedding_model="text-embedding-3-small"    # Irrelevant in inductive mode
    # )

    # Inductive Coding without Parsing
    # main(
    #     coding_mode="inductive",
    #     use_parsing=False,
    #     use_rag=False,
    #     parse_model="gpt-4o-mini",  # Irrelevant if use_parsing=False
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",  # Irrelevant in inductive mode
    #     retrieve_embedding_model="text-embedding-3-small"    # Irrelevant in inductive mode
    # )
