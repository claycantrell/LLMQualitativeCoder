# main.py

import os
import json
from qual_functions import (
    MeaningUnit,
    parse_transcript,
    assign_codes_to_meaning_units,
    initialize_faiss_index_from_formatted_file
)

def main():
    # Set API Key from Environment Variable
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        raise ValueError("Set the OPENAI_API_KEY environment variable.")
    
    # Define the path for the prompts folder
    prompts_folder = 'prompts'

    # JSON path (output from the VTT processing script)
    json_transcript_file = 'output_cues.json'
    if not os.path.exists(json_transcript_file):
        raise FileNotFoundError(f"JSON file '{json_transcript_file}' not found.")

    # Load JSON unit
    with open(json_transcript_file, 'r', encoding='utf-8') as file:
        try:
            json_data = json.load(file)  # Read and parse JSON unit
            print(f"Loaded {len(json_data)} speaking turns from '{json_transcript_file}'.")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON from '{json_transcript_file}': {e}")
            return

    # Parsing instructions path
    parse_prompt_file = os.path.join(prompts_folder, 'parse_prompt.txt')
    if not os.path.exists(parse_prompt_file):
        raise FileNotFoundError(f"Parse instructions file '{parse_prompt_file}' not found.")

    with open(parse_prompt_file, 'r', encoding='utf-8') as file:
        parse_instructions = file.read().strip()  # Read and strip any extra whitespace

    # Initialize a list to hold MeaningUnit objects (meaning units)
    meaning_unit_object_list = []

    # Iterate over the JSON unit (speaking turns) and break into meaning units
    for idx, speaking_turn in enumerate(json_data, start=1):
        speaker_id = speaking_turn.get('speaker_name', 'Unknown')
        speaking_turn_string = speaking_turn.get('text_context', '')
        print(f"\nProcessing Speaking Turn {idx}: Speaker - {speaker_id}")
        
        # Replace the placeholder with the actual speaker's name
        formatted_prompt = parse_instructions.replace("{speaker_name}", speaker_id)
        
        # Use parse_transcript to break the speaking_turn into meaning units
        meaning_unit_list = parse_transcript(speaking_turn_string, formatted_prompt,)
        if not meaning_unit_list:
            print(f"No meaning units extracted from Speaking Turn {idx}. Skipping.")
            continue
        for unit_idx, unit in enumerate(meaning_unit_list, start=1):
            meaning_unit_object = MeaningUnit(
                speaker_id=unit.get('speaker_id', speaker_id),
                meaning_unit_string=unit.get('meaning_unit_string', '')
            )
            # Debug: Print each meaning unit being added
            print(f"  Added Meaning Unit {unit_idx}: Speaker - {meaning_unit_object.speaker_id}, Quote - {meaning_unit_object.meaning_unit_string}")
            meaning_unit_object_list.append(meaning_unit_object)

    if not meaning_unit_object_list:
        print("No meaning units extracted from any speaking turns. Exiting.")
        return

    # Coding task prompt path
    coding_instructions_file = os.path.join(prompts_folder, 'coding_prompt.txt')
    if not os.path.exists(coding_instructions_file):
        raise FileNotFoundError(f"Coding instructions file '{coding_instructions_file}' not found.")

    with open(coding_instructions_file, 'r', encoding='utf-8') as file:
        coding_instructions = file.read().strip()

    # Codes list path
    list_of_codes_file = os.path.join(prompts_folder, 'new_schema.txt')
    if not os.path.exists(list_of_codes_file):
        raise FileNotFoundError(f"List of codes file '{list_of_codes_file}' not found.")

    # Initialize FAISS index and get processed codes
    try:
        faiss_index, processed_codes = initialize_faiss_index_from_formatted_file(list_of_codes_file)

        if not processed_codes:
            print(f"No codes found in '{list_of_codes_file}' or failed to process correctly. Exiting.")
            return

    except Exception as e:
        print(f"An error occurred during FAISS index initialization: {e}")
        return

    # Assign codes to meaning units using the LLM and optionally using RAG
    # In this example, we are not using RAG and thus include the entire codebase in the prompt.
    coded_meaning_unit_list = assign_codes_to_meaning_units(
        meaning_unit_list=meaning_unit_object_list,
        coding_instructions=coding_instructions,
        processed_codes=processed_codes,
        index=faiss_index,
        top_k=5,
        context_size=5,
        use_rag=False,  # Set to False to include the full codebase in the prompt
        codebase=processed_codes  # Provide the full codebase here
    )

    # Example: Print all info from meaning unit object
    for unit in coded_meaning_unit_list:
        print(f"\nID: {unit.unique_id}")
        print(f"Speaker: {unit.speaker_id}")
        print(f"Quote: {unit.meaning_unit_string}")
        if unit.assigned_code_list:
            for code in unit.assigned_code_list:
                print(f"  Code: {code.code_name}")
                print(f"  Justification: {code.code_justification}\n")
        else:
            print("  No codes assigned.\n")

if __name__ == "__main__":
    main()
