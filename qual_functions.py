# qual_functions.py

from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, ClassVar, Dict, Tuple
import faiss
import numpy as np
from pydantic import BaseModel

client = OpenAI()

#qualitative code_name name with code_justification for assignment
@dataclass
class CodeAssigned:
    code_name: str
    code_justification: str

#text unit with assigned codes and speaker info
@dataclass
class MeaningUnit:
    unique_id: int = field(init=False)
    speaker_id: str
    meaning_unit_string: str
    assigned_code_list: List[CodeAssigned] = field(default_factory=list) 

#Defines the expected output when parsing the transcript.
class ParseFormat(BaseModel):
    speaker_id: str
    meaning_unit_string_list: list[str]

#Defines the expected output when assigning codes to meaning unit.
class CodeFormat(BaseModel):
    codeList: list[CodeAssigned]

def parse_transcript(meaning_unit_string: str, prompt: str) -> List[dict]:
    """
    Breaks up a speaking turn into smaller meaning units based on criteria in the LLM prompt.
    
    Args:
        meaning_unit_string (str): The dialogue text from a speaker.
        prompt (str): The complete prompt with speaker's name inserted.

    Returns:
        List[dict]: A list of meaning units with 'speaker_id' and 'meaning_unit_string'.
    """
    try:
        response = client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a qualitative research assistant that breaks down speaking turns into smaller meaning units based on given instructions."},
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        f"Speaking Turn:\n{meaning_unit_string}\n\n"
                    )
                }
            ],
            response_format=ParseFormat,
            temperature=0.2,
            max_tokens=1500,
        )
        
        parsed_output = response.choices[0].message.parsed

        # Debug: Print the raw response from LLM
        #print("Raw LLM Output for parse_transcript:")
        #print(parsed_output)

        # Extract 'speaker_id' and individual 'meaning_unit_string' entries from the parsed model
        speaker_id = parsed_output.speaker_id
        meaningunit_stringlist_parsed = parsed_output.meaning_unit_string_list

        # Create a list of meaning units
        meaning_units = [{"speaker_id": speaker_id, "meaning_unit_string": single_quote} for single_quote in meaningunit_stringlist_parsed]
        
        # Debug: Print the parsed meaning units
        #print("Parsed Meaning Units:")
        #print(meaning_units)
        
        return meaning_units

    except Exception as e:
        print(f"An error occurred while parsing transcript into meaning units: {e}")
        return []


def initialize_faiss_index_from_formatted_file(
    codes_list_file: str, 
    embedding_model: str = "text-embedding-ada-002", 
    batch_size: int = 32
) -> Tuple[faiss.IndexFlatL2, List[Dict[str, List[str]]]]:
    """
    Reads codes_list.txt file and initializes a FAISS index directly using batch embedding.
    Returns the FAISS index and the processed codes as dictionaries.
    """
    embeddings = []
    processed_codes = []
    combined_texts = []  # To store combined texts for batch processing
    code_batch = []      # To temporarily store processed codes for each batch

    try:
        with open(codes_list_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines
                
                # Parse the line into code_name and examples
                if ':' in line:
                    parts = line.split(':', 2)
                    if len(parts) < 3:
                        raise ValueError("Line format is incorrect. Expected format: 'code_name:definition_name:examples'")
                    code_name, definition_name, examples_str = parts
                    examples = [ex.strip() for ex in examples_str.split(';') if ex.strip()]
                else:
                    raise ValueError("Line format is incorrect. Expected format: 'code_name:definition_name:examples'")

                processed_code = {
                    'code_name': code_name.strip(),
                    'definition': definition_name.strip(),
                    'examples': examples
                }
                processed_codes.append(processed_code)
                code_batch.append(processed_code)

                # Combine code_name, definition, and examples for embedding
                combined_text = f"{code_name} - {definition_name}: {'; '.join(examples)}"
                combined_texts.append(combined_text)

                # If batch size is reached, process the batch
                if len(combined_texts) == batch_size:
                    response = client.embeddings.create(
                        input=combined_texts,
                        model=embedding_model
                    )
                    batch_embeddings = [res.embedding for res in response.data]
                    embeddings.extend(batch_embeddings)

                    # Reset for the next batch
                    combined_texts = []
                    code_batch = []

            # Process any remaining texts in the last batch
            if combined_texts:
                response = client.embeddings.create(
                    input=combined_texts,
                    model=embedding_model
                )
                batch_embeddings = [res.embedding for res in response.data]
                embeddings.extend(batch_embeddings)

        # Convert embeddings to numpy array
        embeddings = np.array(embeddings).astype('float32')

        # Initialize FAISS index
        if embeddings.size > 0:
            dimension = len(embeddings[0])
            index = faiss.IndexFlatL2(dimension)
            index.add(embeddings)
        else:
            raise ValueError("No valid embeddings found. Check the content of your codes_list.txt file.")

        return index, processed_codes

    except Exception as e:
        print(f"An error occurred while processing the file and initializing FAISS index: {e}")
        raise e

def retrieve_relevant_codes(
    speaker_id: str, 
    meaning_unit_string: str, 
    index: faiss.IndexFlatL2, 
    processed_codes: List[Dict[str, List[str]]], 
    top_k: int = 5, 
    embedding_model: str = "text-embedding-ada-002"
) -> List[Dict[str, List[str]]]:
    """
    Retrieves the top_k most relevant codes for a given meaning_unit_string using FAISS.
    Returns a list of code_name dictionaries with code_name names and examples.
    """
    try:
        meaning_unit_string_with_speaker = f"{speaker_id}\nUnit: {meaning_unit_string}"

        response = client.embeddings.create(
            input=meaning_unit_string_with_speaker,
            model=embedding_model
        )
        # Updated access using dot notation
        meaning_unit_embedding = np.array([response.data[0].embedding]).astype('float32')

        distances, indices = index.search(meaning_unit_embedding, top_k)
        relevant_codes = [processed_codes[idx] for idx in indices[0]]

        print(f"\nSpeaker: {meaning_unit_string_with_speaker}")
        for i, item in enumerate(relevant_codes):
            print(f"Retrieved Code {i+1}: {item['code_name']}")

        return relevant_codes

    except Exception as e:
        print(f"An error occurred while retrieving relevant codes: {e}")
        return []


def assign_codes_to_meaning_units(
    meaning_unit_list: List[MeaningUnit], 
    coding_instructions: str, 
    processed_codes: List[Dict[str, List[str]]], 
    index: faiss.IndexFlatL2, 
    top_k: int = 5,
    context_size: int = 5 
) -> List[MeaningUnit]:
    """
    Assigns codes to each MeaningUnit object, including contextual information from surrounding units.

    Args:
        meaning_unit_list (List[MeaningUnit]): List of MeaningUnit objects to be coded.
        coding_instructions (str): Instructions for the coding task.
        processed_codes (List[Dict[str, List[str]]]): List of processed codes with definitions and examples.
        index (faiss.IndexFlatL2): FAISS index for retrieving relevant codes.
        top_k (int, optional): Number of top relevant codes to retrieve. Defaults to 5.
        context_size (int, optional): Number of preceding and following meaning units to include as context. Defaults to 5.

    Returns:
        List[MeaningUnit]: List of MeaningUnit objects with assigned codes.
    """
    try:
        total_units = len(meaning_unit_list)
        for idx, meaning_unit_object in enumerate(meaning_unit_list):
            unique_id = idx + 1  # Start unique_id at 1
            meaning_unit_object.unique_id = unique_id  # Assign the unique ID to the meaning_unit_string

            # Retrieve relevant codes using FAISS
            relevant_codes = retrieve_relevant_codes(
                meaning_unit_object.speaker_id, 
                meaning_unit_object.meaning_unit_string, 
                index, 
                processed_codes, 
                top_k=top_k
            )

            # Format the relevant codes with examples as a structured string
            relevant_codes_str = "\n".join([
                f"{code_name['code_name']}: " + "; ".join([f"Example {i+1}: {ex}" for i, ex in enumerate(code_name['examples'])])
                for code_name in relevant_codes
            ])

            # Retrieve previous and next meaning units for context
            previous_excerpt = ""
            next_excerpt = ""

            # Collect previous context
            if context_size > 0 and idx > 0:
                prev_units = meaning_unit_list[max(0, idx - context_size):idx]
                previous_excerpt = "\n".join([
                    f"Previous Excerpt {i+1}:\nSpeaker: {unit.speaker_id}\nQuote: {unit.meaning_unit_string}\n"
                    for i, unit in enumerate(prev_units, start=1)
                ]) + "\n" if prev_units else ""

            # Collect next context
            if context_size > 0 and idx < total_units - 1:
                next_units = meaning_unit_list[idx + 1: idx + 1 + context_size]
                next_excerpt = "\n".join([
                    f"Following Excerpt {i+1}:\nSpeaker: {unit.speaker_id}\nQuote: {unit.meaning_unit_string}\n"
                    for i, unit in enumerate(next_units, start=1)
                ]) + "\n" if next_units else ""

            # Construct the full prompt with context
            full_prompt = (
                f"{coding_instructions}\n\n"
                f"Relevant Codes:\n{relevant_codes_str}\n\n"
                f"{previous_excerpt}"
                f"Current Excerpt:\nSpeaker: {meaning_unit_object.speaker_id}\nQuote: {meaning_unit_object.meaning_unit_string}\n\n"
                f"{next_excerpt}"
                f"**Important:** Please use the previous and following excerpts **only** as context to understand the current excerpt better. **Apply codes exclusively to the current excerpt provided above. Do not assign codes to the previous or following excerpts.**"
            )

            print("full prompt")
            print(full_prompt)

            response = client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": (
                            "You are tasked with applying qualitative codes to excerpts from a transcript between a teacher and a coach in a teacher coaching meeting. The purpose of this task is to identify all codes that best describe each excerpt based on the provided list of codes and their examples."
                        )
                    },
                    {
                        "role": "user",
                        "content": full_prompt
                    }
                ],
                response_format=CodeFormat,  # Parse output using the CodeFormat model
                temperature=0.2,
                max_tokens=1500,
            )

            # Retrieve the parsed response as a structured list of CodeAssigned
            code_output = response.choices[0].message.parsed
            print("LLM Code Assignment Output:")
            print(code_output.codeList)

            # Append each code_name and code_justification to the meaning_unit_object
            for code_item in code_output.codeList:
                meaning_unit_object.assigned_code_list.append(CodeAssigned(code_name=code_item.code_name, code_justification=code_item.code_justification))

    except Exception as e:
        print(f"An error occurred while assigning codes: {e}")

    return meaning_unit_list