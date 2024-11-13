# qual_functions.py

import logging
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Any, Optional
import faiss
import json
import numpy as np
from pydantic import BaseModel

client = OpenAI()

# Configure module-level logger
logger = logging.getLogger(__name__)

# Qualitative code with code_name and code_justification for assignment
@dataclass
class CodeAssigned:
    code_name: str
    code_justification: str

# Text unit with assigned codes and speaker info
@dataclass
class MeaningUnit:
    unique_id: int = field(init=False)
    speaker_id: str
    meaning_unit_string: str
    assigned_code_list: List[CodeAssigned] = field(default_factory=list) 

# Defines the expected output when parsing the transcript.
class ParseFormat(BaseModel):
    speaker_id: str
    meaning_unit_string_list: List[str]

# Defines the expected output when assigning codes to meaning unit.
class CodeFormat(BaseModel):
    codeList: List[CodeAssigned]

def parse_transcript(speaking_turn_string: str, prompt: str, completion_model: str) -> List[dict]:
    """
    Breaks up a speaking turn into smaller meaning units based on criteria in the LLM prompt.
    
    Args:
        speaking_turn_string (str): The dialogue text from a speaker.
        prompt (str): The complete prompt with speaker's name inserted.
        completion_model (str): The OpenAI model to use for parsing the transcript.

    Returns:
        List[dict]: A list of meaning units with 'speaker_id' and 'meaning_unit_string'.
    """
    try:
        response = client.beta.chat.completions.parse(
            model=completion_model,
            messages=[
                {
                    "role": "system", 
                    "content": "You are a qualitative research assistant that breaks down speaking turns into smaller meaning units based on given instructions."
                },
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        f"Speaking Turn:\n{speaking_turn_string}\n\n"
                    )
                }
            ],
            response_format=ParseFormat,
            temperature=0.2,
            max_tokens=1500,
        )
        
        parsed_output = response.choices[0].message.parsed

        # Extract 'speaker_id' and individual 'meaning_unit_string' entries from the parsed model
        speaker_id = parsed_output.speaker_id
        meaningunit_stringlist_parsed = parsed_output.meaning_unit_string_list

        # Create a list of meaning units
        meaning_units = [{"speaker_id": speaker_id, "meaning_unit_string": single_quote} for single_quote in meaningunit_stringlist_parsed]
        
        logger.debug(f"Parsed transcript for speaker '{speaker_id}'. Extracted {len(meaning_units)} meaning units.")
        return meaning_units

    except Exception as e:
        logger.error(f"An error occurred while parsing transcript into meaning units: {e}")
        return []

def initialize_faiss_index_from_formatted_file(
    codes_list_file: str, 
    embedding_model: str = "text-embedding-3-small", 
    batch_size: int = 32
) -> Tuple[faiss.IndexFlatL2, List[Dict[str, Any]]]:
    """
    Reads a JSONL-formatted file and initializes a FAISS index directly using batch embedding.
    Returns the FAISS index and the processed codes as dictionaries.
    """
    embeddings = []
    processed_codes = []
    combined_texts = []  # To store combined texts for batch processing

    try:
        with open(codes_list_file, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue  # Skip empty lines

                # Parse each JSONL line
                data = json.loads(line)
                text = data.get("text", "")
                metadata = data.get("metadata", {})
                
                processed_code = {
                    'text': text,
                    'metadata': metadata
                }
                processed_codes.append(processed_code)

                # Combine `text` and metadata elements for embedding
                combined_text = f"{text} Metadata: {metadata}"
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
            raise ValueError("No valid embeddings found. Check the content of your JSONL file.")

        logger.info(f"Initialized FAISS index with {len(processed_codes)} codes from file '{codes_list_file}'.")
        return index, processed_codes

    except Exception as e:
        logger.error(f"An error occurred while processing the file '{codes_list_file}' and initializing FAISS index: {e}")
        raise e

def retrieve_relevant_codes(
    speaker_id: str, 
    meaning_unit_string: str, 
    index: faiss.IndexFlatL2, 
    processed_codes: List[Dict[str, Any]], 
    top_k: int = 5, 
    embedding_model: str = "text-embedding-3-small"
) -> List[Dict[str, Any]]:
    """
    Retrieves the top_k most relevant codes for a given meaning_unit_string using FAISS.
    Returns a list of code dictionaries with relevant information.
    """
    try:
        meaning_unit_string_with_speaker = f"{speaker_id}\nUnit: {meaning_unit_string}"

        response = client.embeddings.create(
            input=meaning_unit_string_with_speaker,
            model=embedding_model
        )
        meaning_unit_embedding = np.array([response.data[0].embedding]).astype('float32')

        distances, indices = index.search(meaning_unit_embedding, top_k)
        relevant_codes = [processed_codes[idx] for idx in indices[0]]

        logger.debug(f"Retrieved top {top_k} relevant codes for speaker '{speaker_id}': {[code.get('text', 'Unnamed Code') for code in relevant_codes]}")
        return relevant_codes

    except Exception as e:
        logger.error(f"An error occurred while retrieving relevant codes: {e}")
        return []

def assign_codes_to_meaning_units(
    meaning_unit_list: List[MeaningUnit], 
    coding_instructions: str, 
    processed_codes: Optional[List[Dict[str, Any]]] = None, 
    index: Optional[faiss.IndexFlatL2] = None, 
    top_k: Optional[int] = 5,
    context_size: int = 5,
    use_rag: bool = True,
    codebase: Optional[List[Dict[str, Any]]] = None,
    completion_model: Optional[str] = "gpt-4o-mini",
    embedding_model: Optional[str] = "text-embedding-3-small"
) -> List[MeaningUnit]:
    """
    Assigns codes to each MeaningUnit object, including contextual information from surrounding units.

    Args:
        meaning_unit_list (List[MeaningUnit]): List of MeaningUnit objects to be coded.
        coding_instructions (str): Instructions for the coding task or custom coding prompt.
        processed_codes (List[Dict[str, Any]], optional): List of processed codes with definitions and examples.
        index (faiss.IndexFlatL2, optional): FAISS index for retrieving relevant codes.
        top_k (Optional[int], optional): Number of top relevant codes to retrieve. Not applicable in inductive mode.
                                         Defaults to 5.
        context_size (int, optional): Number of preceding and following meaning units to include as context. Defaults to 5.
        use_rag (bool, optional): Flag to determine whether to use RAG or include the entire codebase. Defaults to True.
        codebase (List[Dict[str, Any]], optional): The entire codebase to include in the prompt when not using RAG.
        completion_model (Optional[str], optional): The OpenAI model to use for generative tasks (assigning codes).
                                                   Defaults to "gpt-4o-mini".
        embedding_model (Optional[str], optional): The OpenAI embedding model used for retrieval. Not applicable in inductive mode.
                                                   Defaults to "text-embedding-3-small".

    Returns:
        List[MeaningUnit]: List of MeaningUnit objects with assigned codes.
    """
    try:
        total_units = len(meaning_unit_list)
        for idx, meaning_unit_object in enumerate(meaning_unit_list):
            unique_id = idx + 1  # Start unique_id at 1
            meaning_unit_object.unique_id = unique_id  # Assign the unique ID to the meaning_unit_string

            # Determine if deductive or inductive based on presence of processed_codes and codebase
            is_deductive = processed_codes is not None and index is not None

            if is_deductive and use_rag:
                # Retrieve relevant codes using FAISS and the specified embedding model
                relevant_codes = retrieve_relevant_codes(
                    meaning_unit_object.speaker_id, 
                    meaning_unit_object.meaning_unit_string, 
                    index, 
                    processed_codes, 
                    top_k=top_k,
                    embedding_model=embedding_model
                )

                # Format the relevant codes as their entire JSONL lines
                codes_to_include = relevant_codes
            elif is_deductive and not use_rag and codebase:
                # Include the entire codebase
                codes_to_include = codebase
            else:
                # Inductive coding: No predefined codes
                codes_to_include = None

            if codes_to_include is not None:
                # Deductive coding: format codes as a string
                codes_str = "\n\n".join([
                    json.dumps(code, indent=2)
                    for code in codes_to_include
                ])
            else:
                # Inductive coding: No predefined codes
                codes_str = "No predefined codes. Please generate codes based on the following guidelines."

            # Retrieve previous and next meaning units for context
            context_excerpt = ""

            # Collect previous context
            if context_size > 0 and idx > 0:
                prev_units = meaning_unit_list[max(0, idx - context_size):idx]
                context_excerpt += "\n".join([
                    f"{unit.speaker_id}: {unit.meaning_unit_string}"
                    for unit in prev_units
                ]) + "\n"

            # Add current excerpt embedded into context
            context_excerpt += f"{meaning_unit_object.speaker_id}: {meaning_unit_object.meaning_unit_string}\n"

            # Collect next context
            if context_size > 0 and idx < total_units - 1:
                next_units = meaning_unit_list[idx + 1: idx + 1 + context_size]
                context_excerpt += "\n".join([
                    f"{unit.speaker_id}: {unit.meaning_unit_string}"
                    for unit in next_units
                ]) + "\n"

            # Separately label the current excerpt for coding
            current_excerpt_labeled = (
                f"Current Excerpt for Coding:\n"
                f"Speaker: {meaning_unit_object.speaker_id}\n"
                f"Quote: {meaning_unit_object.meaning_unit_string}\n"
            )

            # Construct the full prompt with context and clearly labeled current excerpt
            full_prompt = (
                f"{coding_instructions}\n\n"
                f"{'Relevant Codes (full details):' if codes_to_include else 'Guidelines for Inductive Coding:'}\n{codes_str}\n\n"
                f"Contextual Excerpts:\n{context_excerpt}\n"
                f"{current_excerpt_labeled}"
                f"**Important:** Please use the provided contextual excerpts **only** as background information to understand the current excerpt better. "
                f"{'**Apply codes exclusively to the current excerpt provided above. Do not assign codes to the contextual excerpts.**' if codes_to_include else '**Generate codes based on the current excerpt provided above using the guidelines.**'}\n\n"
                f"Please provide the assigned codes in the following JSON format:\n"
                f"{{\n  \"codeList\": [\n    {{\"code_name\": \"<Name of the code>\", \"code_justification\": \"<Justification for the code>\"}},\n    ...\n  ]\n}}"
            )

            logger.debug(f"Full Prompt for Unique ID {unique_id}:\n{full_prompt}")

            try:
                response = client.beta.chat.completions.parse(
                    model=completion_model,
                    messages=[
                        {
                            "role": "system", 
                            "content": (
                                "You are tasked with applying qualitative codes to excerpts from a transcript between a teacher and a coach in a teacher coaching meeting. "
                                "The purpose of this task is to identify all codes that best describe each excerpt based on the provided instructions."
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
                logger.debug(f"LLM Code Assignment Output for ID {unique_id}:\n{code_output.codeList}")

                # Append each code_name and code_justification to the meaning_unit_object
                for code_item in code_output.codeList:
                    # Access the fields with fallbacks to handle unexpected structures
                    code_name = getattr(code_item, 'code_name', getattr(code_item, 'name', 'Unknown Code'))
                    code_justification = getattr(code_item, 'code_justification', 'No justification provided')
                    
                    meaning_unit_object.assigned_code_list.append(
                        CodeAssigned(code_name=code_name, code_justification=code_justification)
                    )

            except Exception as e:
                logger.error(f"An error occurred while retrieving code assignments for Unique ID {unique_id}: {e}")

    except Exception as e:
        logger.error(f"An error occurred while assigning codes: {e}")

    return meaning_unit_list
