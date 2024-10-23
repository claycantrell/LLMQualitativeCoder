# parse_task.py
import os
from openai import OpenAI
from dataclasses import dataclass, field
from typing import List, ClassVar, Dict, Tuple
import csv
import io
import re
import faiss
import numpy as np


client = OpenAI()

@dataclass
class Code:
    code: str
    justification: str

@dataclass
class PassageData:
    unique_id: int = field(init=False)
    speaker_id: str
    quote: str
    codes: List[Code] = field(default_factory=list) 

    # Class variable to keep track of the last assigned ID
    _id_counter: ClassVar[int] = 0

    def __post_init__(self):
        # Increment the counter and assign it to unique_id
        type(self)._id_counter += 1
        self.unique_id = self._id_counter

def parse_transcript(transcript: str, instructions: str) -> List[dict]:
    """
    Sends the transcript and instructions to the LLM and returns the parsed data in CSV format.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a qualitative research assistant that parses transcripts based on given instructions."},
                {
                    "role": "user",
                    "content": (
                        f"{instructions}\n\n"
                        f"Transcript:\n{transcript}\n\n"
                        "Please provide the output in CSV format"
                        "Do not include any additional text, explanations, or code blocks."
                    )
                }
            ],
            temperature=0.2,
            max_tokens=1500,
        )

        #print("Parse Raw Output:")
        #print(response)

        parsed_output = response.choices[0].message.content.strip()

        #print("Parse Message Content:")
        #print(parsed_output)

        # Attempt to extract CSV content
        csv_content = ""

        # Check if the output is within a code block
        csv_pattern = re.compile(r"```csv\s*\n(.*?)\n```", re.DOTALL)
        match = csv_pattern.search(parsed_output)
        if match:
            csv_content = match.group(1).strip()
            print("Extracted CSV from Code Block:")
            print(csv_content)
        else:
            # Check if output starts with 'speaker_id,quote' indicating CSV
            if parsed_output.lower().startswith("speaker_id,"):
                csv_content = parsed_output
                print("CSV detected without code block:")
                print(csv_content)
            else:
                print("LLM did not return CSV format as expected.")
                return []

        # Use StringIO to treat the string as a file for csv.DictReader
        csv_file = io.StringIO(csv_content)
        reader = csv.DictReader(csv_file)

        parsed_data = []
        for row in reader:
            # Normalize column names by stripping whitespace
            row = {k.strip(): v.strip() for k, v in row.items()}

            # Ensure that the required fields are present
            if 'speaker_id' in row and 'quote' in row:
                parsed_data.append({
                    'speaker_id': row['speaker_id'],
                    'quote': row['quote']
                })
            else:
                print("CSV row missing required fields:", row)

        print(f"Parsed Data: {parsed_data}")

        return parsed_data

    except Exception as e:
        print(f"An error occurred while parsing CSV: {e}")
        return []

def initialize_faiss_index_from_formatted_file(codes_list_file: str, embedding_model: str = "text-embedding-ada-002", batch_size: int = 10) -> Tuple[faiss.IndexFlatL2, List[Dict[str, List[str]]]]:
    """
    Reads a pre-formatted codes_list.txt file and initializes a FAISS index directly using batch embedding.
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
                
                # Parse the line into code and examples
                if ':' in line:
                    code_name, examples_str = line.split(':', 1)
                    examples = [ex.strip() for ex in examples_str.split(';') if ex.strip()]
                else:
                    code_name = line
                    examples = []

                processed_code = {
                    'code': code_name.strip(),
                    'examples': examples
                }
                processed_codes.append(processed_code)
                code_batch.append(processed_code)

                # Combine code and examples for embedding
                combined_text = f"{code_name}: {'; '.join(examples)}"
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


def retrieve_relevant_codes(passage: str, index: faiss.IndexFlatL2, processed_codes: List[Dict[str, List[str]]], top_k: int = 5, embedding_model: str = "text-embedding-ada-002") -> List[Dict[str, List[str]]]:
    """
    Retrieves the top_k most relevant codes for a given passage using FAISS.
    Returns a list of code dictionaries with code names and examples.
    """
    try:
        response = client.embeddings.create(
            input=passage,
            model=embedding_model
        )
        # Updated access using dot notation
        passage_embedding = np.array([response.data[0].embedding]).astype('float32')

        distances, indices = index.search(passage_embedding, top_k)
        relevant_codes = [processed_codes[idx] for idx in indices[0]]

        return relevant_codes

    except Exception as e:
        print(f"An error occurred while retrieving relevant codes: {e}")
        return []

def assign_codes_to_passages(passages: List[PassageData], coding_instructions: str, processed_codes: List[Dict[str, List[str]]], index: faiss.IndexFlatL2, top_k: int = 5) -> List[PassageData]:
    """
    Assigns codes to each PassageData object using the LLM based on the coding instructions and RAG.
    """
    try:
        for passage in passages:
            # Retrieve relevant codes using RAG
            relevant_codes = retrieve_relevant_codes(passage.quote, index, processed_codes, top_k=top_k)

            # Format the relevant codes with examples as a structured string
            relevant_codes_str = "\n".join([
                f"{code['code']}: " + " ".join([f"Example {i+1}: {ex}" for i, ex in enumerate(code['examples'])])
                for code in relevant_codes
            ])

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a qualitative coding assistant that assigns codes based on given instructions and relevant code list with examples."},
                    {
                        "role": "user",
                        "content": (
                            f"{coding_instructions}\n\n"
                            f"Relevant Codes:\n{relevant_codes_str}\n\n"
                            f"Passage:\nSpeaker: {passage.speaker_id}\nQuote: {passage.quote}\n\n"
                            "Please provide the code and justification for this passage in CSV format with the following columns:\n\n"
                            "code, justification\n\n"
                            "Where:\n"
                            "- `code` is the primary code you have selected for the passage.\n"
                            "- `justification` is a brief explanation (2-3 sentences) for why this code was applied."
                        )
                    }
                ],
                temperature=0.2,
                max_tokens=1000,
            )

            code_output = response.choices[0].message.content.strip()

            print("LLM Code Assignment Output:")
            print(code_output)

            # Attempt to extract CSV content
            csv_content = ""

            # Check if the output is within a code block
            csv_pattern = re.compile(r"```csv\s*\n(.*?)\n```", re.DOTALL)
            match = csv_pattern.search(code_output)
            if match:
                csv_content = match.group(1).strip()
                print("Extracted CSV from Code Block:")
                print(csv_content)
            else:
                # Check if output starts with 'code, justification' indicating CSV
                if code_output.lower().startswith("code,"):
                    csv_content = code_output
                    print("CSV detected without code block:")
                    print(csv_content)
                else:
                    print("LLM did not return CSV format as expected.")
                    continue  # Skip to the next passage

            # Use StringIO to treat the string as a file for csv.DictReader
            csv_file = io.StringIO(csv_content)
            reader = csv.DictReader(csv_file)

            for row in reader:
                row = {k.strip(): v.strip() for k, v in row.items()}
                if 'code' in row and 'justification' in row:
                    passage.codes.append(Code(code=row['code'], justification=row['justification']))
                else:
                    print("CSV row missing required fields:", row)

    except Exception as e:
        print(f"An error occurred while assigning codes: {e}")

    return passages
