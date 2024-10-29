# main.py
import os
from parse_task import (
    OpenAI, 
    PassageData, 
    parse_transcript, 
    assign_codes_to_passages, 
    initialize_faiss_index_from_formatted_file
)

def main():
    # Set API 
    OpenAI.api_key = os.getenv('OPENAI_API_KEY')
    if not OpenAI.api_key:
        raise ValueError("Set the OPENAI_API_KEY environment variable.")

    # Define the path for the prompts folder
    prompts_folder = 'prompts'

    # transcript path
    transcript_file = os.path.join(prompts_folder, 'transcript.txt')
    if not os.path.exists(transcript_file):
        raise FileNotFoundError(f"Transcript file '{transcript_file}' not found.")

    with open(transcript_file, 'r', encoding='utf-8') as file:
        transcript = file.read().strip()  # Read and strip any extra whitespace

    # parsing instructions path
    instructions_file = os.path.join(prompts_folder, 'parse_prompt.txt')
    if not os.path.exists(instructions_file):
        raise FileNotFoundError(f"Instructions file '{instructions_file}' not found.")

    with open(instructions_file, 'r', encoding='utf-8') as file:
        instructions = file.read().strip()  # Read and strip any extra whitespace

    # Get parsed data from LLM
    parsed_data = parse_transcript(transcript, instructions)

    if not parsed_data:
        print("No data parsed. Exiting.")
        return

    # Initialize a list to hold PassageData objects
    passage_data_list = []

    # Iterate over the parsed data and create PassageData instances
    for item in parsed_data:
        passage_data = PassageData(
            speaker_id=item.get('speaker_id', 'Unknown'),
            quote=item.get('quote', '')
        )
        passage_data_list.append(passage_data)

    # coding task prompt path
    coding_instructions_file = os.path.join(prompts_folder, 'coding_prompt.txt')
    if not os.path.exists(coding_instructions_file):
        raise FileNotFoundError(f"Coding instructions file '{coding_instructions_file}' not found.")

    with open(coding_instructions_file, 'r', encoding='utf-8') as file:
        coding_instructions = file.read().strip()

    # codes list path
    list_of_codes_file = os.path.join(prompts_folder, 'codes_list.txt')
    if not os.path.exists(list_of_codes_file):
        raise FileNotFoundError(f"List of codes file '{list_of_codes_file}' not found.")

    # Initialize FAISS index and get processed codes
    try:
        faiss_index, processed_codes = initialize_faiss_index_from_formatted_file(list_of_codes_file)

        if not processed_codes:
            print("No codes found in codes_list.txt or failed to process correctly. Exiting.")
            return

    except Exception as e:
        print(f"An error occurred during FAISS index initialization: {e}")
        return

    # Assign codes to passages using the LLM and RAG
    coded_passages = assign_codes_to_passages(passage_data_list, coding_instructions, processed_codes, faiss_index, top_k=5)

    # Example: Print the coding data for each passage
    for data in coded_passages:
        print(f"\nID: {data.unique_id}")
        print(f"Speaker: {data.speaker_id}")
        print(f"Quote: {data.quote}")
        for code in data.codes:
            print(f"  Code: {code.code}")
            print(f"  Justification: {code.justification}\n")

if __name__ == "__main__":
    main()
