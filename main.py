# main.py

import os
import json
import logging
from qual_functions import (
    MeaningUnit,
    parse_transcript,
    assign_codes_to_meaning_units,
    initialize_faiss_index_from_formatted_file
)

# Configure logging
logging.basicConfig(level=logging.INFO,
                     format='%(message)s')  # Only show the message without any additional info)
logger = logging.getLogger(__name__)



def main(use_parsing: bool = True, use_rag: bool = True):
    """
    Processes the transcript data to assign qualitative codes.

    Args:
        use_parsing (bool, optional): Whether to parse speaking turns into meaning units using the parse_transcript function.
                                      If False, the entire speaking turn will be used as a single meaning unit.
                                      Defaults to True.
        use_rag (bool, optional): Whether to use Retrieval-Augmented Generation (RAG) for code assignment.
                                  If False, the entire codebase is included directly in the prompt.
                                  Defaults to True.
    """
    # Set API Key from Environment Variable
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise ValueError("Set the OPENAI_API_KEY environment variable.")
    
    # Define the path for the prompts folder
    prompts_folder = 'prompts'

    # JSON path (output from the VTT processing script)
    json_transcript_file = 'output_cues.json'
    if not os.path.exists(json_transcript_file):
        logger.error(f"JSON file '{json_transcript_file}' not found.")
        raise FileNotFoundError(f"JSON file '{json_transcript_file}' not found.")

    # Load JSON data
    with open(json_transcript_file, 'r', encoding='utf-8') as file:
        try:
            json_data = json.load(file)  # Read and parse JSON unit
            logger.info(f"Loaded {len(json_data)} speaking turns from '{json_transcript_file}'.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{json_transcript_file}': {e}")
            return

    # Parsing instructions path
    parse_prompt_file = os.path.join(prompts_folder, 'parse_prompt.txt')
    if not os.path.exists(parse_prompt_file):
        logger.error(f"Parse instructions file '{parse_prompt_file}' not found.")
        raise FileNotFoundError(f"Parse instructions file '{parse_prompt_file}' not found.")

    with open(parse_prompt_file, 'r', encoding='utf-8') as file:
        parse_instructions = file.read().strip()  # Read and strip any extra whitespace

    # Initialize a list to hold MeaningUnit objects (meaning units)
    meaning_unit_object_list = []

    if use_parsing:
        # Iterate over the JSON unit (speaking turns) and break into meaning units
        for idx, speaking_turn in enumerate(json_data, start=1):
            speaker_id = speaking_turn.get('speaker_name', 'Unknown')
            speaking_turn_string = speaking_turn.get('text_context', '')
            logger.info(f"Processing Speaking Turn {idx} (Parsing): Speaker - {speaker_id}")
            
            # Replace the placeholder with the actual speaker's name
            formatted_prompt = parse_instructions.replace("{speaker_name}", speaker_id)
            
            # Use parse_transcript to break the speaking turn into meaning units
            meaning_unit_list = parse_transcript(speaking_turn_string, formatted_prompt,)
            if not meaning_unit_list:
                logger.warning(f"No meaning units extracted from Speaking Turn {idx}. Skipping.")
                continue
            for unit_idx, unit in enumerate(meaning_unit_list, start=1):
                meaning_unit_object = MeaningUnit(
                    speaker_id=unit.get('speaker_id', speaker_id),
                    meaning_unit_string=unit.get('meaning_unit_string', '')
                )
                logger.debug(f"Added Meaning Unit {unit_idx}: Speaker - {meaning_unit_object.speaker_id}, "
                             f"Quote - {meaning_unit_object.meaning_unit_string}")
                meaning_unit_object_list.append(meaning_unit_object)
    else:
        # Use entire speaking turns as meaning units without parsing
        for idx, speaking_turn in enumerate(json_data, start=1):
            speaker_id = speaking_turn.get('speaker_name', 'Unknown')
            speaking_turn_string = speaking_turn.get('text_context', '')
            if not speaking_turn_string:
                logger.warning(f"No speaking turn text found for Speaking Turn {idx}. Skipping.")
                continue
            
            logger.info(f"Processing Speaking Turn {idx} (No Parsing): Speaker - {speaker_id}")
            meaning_unit_object = MeaningUnit(
                speaker_id=speaker_id,
                meaning_unit_string=speaking_turn_string
            )
            logger.debug(f"Added Meaning Unit: Speaker - {meaning_unit_object.speaker_id}, "
                         f"Quote - {meaning_unit_object.meaning_unit_string}")
            meaning_unit_object_list.append(meaning_unit_object)

    if not meaning_unit_object_list:
        logger.warning("No meaning units extracted (or speaking turns processed). Exiting.")
        return

    # Coding task prompt path
    coding_instructions_file = os.path.join(prompts_folder, 'coding_prompt.txt')
    if not os.path.exists(coding_instructions_file):
        logger.error(f"Coding instructions file '{coding_instructions_file}' not found.")
        raise FileNotFoundError(f"Coding instructions file '{coding_instructions_file}' not found.")

    with open(coding_instructions_file, 'r', encoding='utf-8') as file:
        coding_instructions = file.read().strip()

    # Codes list path
    list_of_codes_file = os.path.join(prompts_folder, 'new_schema.txt')
    if not os.path.exists(list_of_codes_file):
        logger.error(f"List of codes file '{list_of_codes_file}' not found.")
        raise FileNotFoundError(f"List of codes file '{list_of_codes_file}' not found.")

    # Initialize FAISS index and get processed codes
    try:
        faiss_index, processed_codes = initialize_faiss_index_from_formatted_file(list_of_codes_file)

        if not processed_codes:
            logger.warning(f"No codes found in '{list_of_codes_file}' or failed to process correctly. Exiting.")
            return

    except Exception as e:
        logger.error(f"An error occurred during FAISS index initialization: {e}")
        return

    if use_rag:
        # Assign codes to meaning units using the LLM and RAG
        coded_meaning_unit_list = assign_codes_to_meaning_units(
            meaning_unit_list=meaning_unit_object_list,
            coding_instructions=coding_instructions,
            processed_codes=processed_codes,
            index=faiss_index,
            top_k=5,
            context_size=5,
            use_rag=True
        )
    else:
        # Assign codes to meaning units directly by providing the codebase
        coded_meaning_unit_list = assign_codes_to_meaning_units(
            meaning_unit_list=meaning_unit_object_list,
            coding_instructions=coding_instructions,
            processed_codes=processed_codes,
            index=faiss_index,
            top_k=5,
            context_size=5,
            use_rag=False,  # Include the full codebase in the prompt
            codebase=processed_codes
        )

    # Output information about each coded meaning unit
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
    # 1. Parsing meaning units from speaking turns and then coding (use_parsing=True, use_rag=False)
    # 2. Directly coding speaking turns without parsing (use_parsing=False, use_rag=False)
    # You can also toggle use_rag to True if you want to use RAG for code retrieval.
    main(use_parsing=False, use_rag=False)
