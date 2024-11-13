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
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main(
    coding_mode: str = "deductive",
    use_parsing: bool = True,
    use_rag: bool = True,
    parse_model: str = "gpt-4o-mini",
    assign_model: str = "gpt-4o-mini",
    initialize_embedding_model: str = "text-embedding-3-small",
    retrieve_embedding_model: str = "text-embedding-3-small",
    custom_coding_prompt: str = None  # Only used in inductive mode
):
    """
    Processes the transcript data to assign qualitative codes.

    Args:
        coding_mode (str, optional): The coding mode to use. Options are "deductive" or "inductive".
                                     Defaults to "deductive".
        use_parsing (bool, optional): Whether to parse speaking turns into meaning units using the parse_transcript function.
                                      If False, the entire speaking turn will be used as a single meaning unit.
                                      Defaults to True.
        use_rag (bool, optional): Whether to use Retrieval-Augmented Generation (RAG) for code assignment.
                                  If False, the entire codebase is included directly in the prompt.
                                  Defaults to True.
        parse_model (str, optional): The OpenAI model to use for parsing transcripts into meaning units.
                                     Defaults to "gpt-4o-mini".
        assign_model (str, optional): The OpenAI model to use for assigning codes to meaning units.
                                      Defaults to "gpt-4o-mini".
        initialize_embedding_model (str, optional): The OpenAI embedding model to use for FAISS indexing.
                                                   Defaults to "text-embedding-3-small".
        retrieve_embedding_model (str, optional): The OpenAI embedding model to use for retrieving relevant codes.
                                                 Defaults to "text-embedding-3-small".
        custom_coding_prompt (str, optional): The custom GPT prompt guiding inductive coding.
                                              Required if `coding_mode` is "inductive".
                                              Defaults to None.
    """
    # Validate coding_mode
    if coding_mode not in ["deductive", "inductive"]:
        logger.error("Invalid coding_mode. Choose 'deductive' or 'inductive'.")
        raise ValueError("Invalid coding_mode. Choose 'deductive' or 'inductive'.")

    # In inductive mode, ensure custom_coding_prompt is provided
    if coding_mode == "inductive" and not custom_coding_prompt:
        logger.error("In 'inductive' mode, 'custom_coding_prompt' must be provided.")
        raise ValueError("In 'inductive' mode, 'custom_coding_prompt' must be provided.")

    # Set API Key from Environment Variable
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise ValueError("Set the OPENAI_API_KEY environment variable.")

    # Define the path for the prompts folder and other directories
    prompts_folder = 'prompts'
    codebase_folder = 'qual_codebase'
    json_folder = 'json_transcripts'

    # JSON path (output from the VTT processing script)
    json_transcript_file = os.path.join(json_folder, 'output_cues.json')
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

    # Initialize a list to hold MeaningUnit objects (meaning units)
    meaning_unit_object_list = []

    if coding_mode == "deductive" and use_parsing:
        # Parsing instructions path
        parse_prompt_file = os.path.join(prompts_folder, 'parse_prompt.txt')
        if not os.path.exists(parse_prompt_file):
            logger.error(f"Parse instructions file '{parse_prompt_file}' not found.")
            raise FileNotFoundError(f"Parse instructions file '{parse_prompt_file}' not found.")

        with open(parse_prompt_file, 'r', encoding='utf-8') as file:
            parse_instructions = file.read().strip()  # Read and strip any extra whitespace

        # Iterate over the JSON unit (speaking turns) and break into meaning units
        for idx, speaking_turn in enumerate(json_data, start=1):
            speaker_id = speaking_turn.get('speaker_name', 'Unknown')
            speaking_turn_string = speaking_turn.get('text_context', '')
            logger.info(f"Processing Speaking Turn {idx} (Parsing): Speaker - {speaker_id}")

            if not speaking_turn_string:
                logger.warning(f"No speaking turn text found for Speaking Turn {idx}. Skipping.")
                continue

            # Replace the placeholder with the actual speaker's name
            formatted_prompt = parse_instructions.replace("{speaker_name}", speaker_id)

            # Use parse_transcript to break the speaking turn into meaning units
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
                logger.debug(f"Added Meaning Unit {unit_idx}: Speaker - {meaning_unit_object.speaker_id}, "
                             f"Quote - {meaning_unit_object.meaning_unit_string}")
                meaning_unit_object_list.append(meaning_unit_object)
    elif coding_mode == "inductive":
        # In inductive mode, use the entire speaking turn as meaning units
        for idx, speaking_turn in enumerate(json_data, start=1):
            speaker_id = speaking_turn.get('speaker_name', 'Unknown')
            speaking_turn_string = speaking_turn.get('text_context', '')
            if not speaking_turn_string:
                logger.warning(f"No speaking turn text found for Speaking Turn {idx}. Skipping.")
                continue

            logger.info(f"Processing Speaking Turn {idx} (Inductive Coding): Speaker - {speaker_id}")
            meaning_unit_object = MeaningUnit(
                speaker_id=speaker_id,
                meaning_unit_string=speaking_turn_string
            )
            logger.debug(f"Added Meaning Unit: Speaker - {meaning_unit_object.speaker_id}, "
                         f"Quote - {meaning_unit_object.meaning_unit_string}")
            meaning_unit_object_list.append(meaning_unit_object)
    else:
        # Deductive coding without parsing (entire speaking turns as meaning units)
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

    # Coding task prompt path (only needed for deductive coding)
    if coding_mode == "deductive":
        coding_instructions_file = os.path.join(prompts_folder, 'coding_prompt.txt')
        if not os.path.exists(coding_instructions_file):
            logger.error(f"Coding instructions file '{coding_instructions_file}' not found.")
            raise FileNotFoundError(f"Coding instructions file '{coding_instructions_file}' not found.")

        with open(coding_instructions_file, 'r', encoding='utf-8') as file:
            coding_instructions = file.read().strip()
    else:
        # In inductive mode, use the custom coding prompt provided by the user
        coding_instructions = custom_coding_prompt

    # Codes list path (only needed for deductive coding)
    if coding_mode == "deductive":
        list_of_codes_file = os.path.join(codebase_folder, 'new_schema.txt')
        if not os.path.exists(list_of_codes_file):
            logger.error(f"List of codes file '{list_of_codes_file}' not found.")
            raise FileNotFoundError(f"List of codes file '{list_of_codes_file}' not found.")

    # Initialize FAISS index and get processed codes (only for deductive coding)
    if coding_mode == "deductive":
        try:
            faiss_index, processed_codes = initialize_faiss_index_from_formatted_file(
                list_of_codes_file,
                embedding_model=initialize_embedding_model
            )

            if not processed_codes:
                logger.warning(f"No codes found in '{list_of_codes_file}' or failed to process correctly. Exiting.")
                return

        except Exception as e:
            logger.error(f"An error occurred during FAISS index initialization: {e}")
            return
    else:
        # In inductive mode, FAISS index and processed_codes are not used
        faiss_index = None
        processed_codes = None

    if coding_mode == "deductive":
        if use_rag:
            # Deductive Coding: Assign codes using RAG
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
            # Deductive Coding: Assign codes by providing the full codebase in the prompt
            coded_meaning_unit_list = assign_codes_to_meaning_units(
                meaning_unit_list=meaning_unit_object_list,
                coding_instructions=coding_instructions,
                processed_codes=processed_codes,
                index=faiss_index,
                top_k=5,
                context_size=5,
                use_rag=False,  # Include the full codebase in the prompt
                codebase=processed_codes,
                completion_model=assign_model,
                embedding_model=retrieve_embedding_model
            )
    elif coding_mode == "inductive":
        # Inductive Coding: Assign codes by generating them based on the custom prompt
        coded_meaning_unit_list = assign_codes_to_meaning_units(
            meaning_unit_list=meaning_unit_object_list,
            coding_instructions=coding_instructions,  # Custom prompt for inductive coding
            processed_codes=None,  # No codebase
            index=None,  # No FAISS index
            top_k=None,  # Not applicable
            context_size=5,
            use_rag=False,  # RAG not applicable
            codebase=None,  # No codebase
            completion_model=assign_model,
            embedding_model=None  # Embedding not needed
        )
    else:
        # Should not reach here due to earlier validation
        logger.error(f"Unsupported coding_mode: {coding_mode}")
        return

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

    # 1. Deductive Coding with Parsing and RAG
    # main(
    #     coding_mode="deductive",
    #     use_parsing=True,
    #     use_rag=True,
    #     parse_model="gpt-4o-mini",
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",
    #     retrieve_embedding_model="text-embedding-3-small"
    # )

    # 2. Deductive Coding without Parsing and without RAG
    # main(
    #     coding_mode="deductive",
    #     use_parsing=False,
    #     use_rag=False,
    #     parse_model="gpt-4o-mini",
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",
    #     retrieve_embedding_model="text-embedding-3-small"
    # )

    # 3. Inductive Coding with Custom Prompt
    main(
        coding_mode="inductive",
        use_parsing=True,  # Irrelevant in inductive mode, can be set to either True or False
        use_rag=False,     # RAG not applicable in inductive mode
        parse_model="gpt-4o-mini",  # Irrelevant in inductive mode
        assign_model="gpt-4o-mini",
        initialize_embedding_model="text-embedding-3-small",  # Irrelevant in inductive mode
        retrieve_embedding_model="text-embedding-3-small",    # Irrelevant in inductive mode
        custom_coding_prompt=(
            "You are a qualitative research assistant. Identify and create codes based on the following guidelines:\n"
            "1. Be as sarcastic as possible\n"
            "2. Highlight areas of conversational focus.\n"
            "3. Generate clear and concise code names with justifications."
        )
    )
