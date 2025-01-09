# qual_functions.py

import logging
import os
import json
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, ValidationError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import openai
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

def get_openai_client() -> OpenAI:
    """
    Retrieves an instance of the OpenAI client.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=openai_api_key)

@dataclass
class CodeAssigned:
    code_name: str
    code_justification: str

    def is_valid(self) -> bool:
        return bool(self.code_name and self.code_justification)

@dataclass
class SpeakingTurn:
    source_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        metadata = self.metadata.copy()
        metadata.pop('source_id', None)
        return {
            "source_id": self.source_id,
            "content": self.content,
            "metadata": metadata
        }

@dataclass
class MeaningUnit:
    meaning_unit_id: int
    meaning_unit_string: str
    assigned_code_list: List[CodeAssigned] = field(default_factory=list)
    speaking_turn: Optional[SpeakingTurn] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "meaning_unit_id": self.meaning_unit_id,
            "meaning_unit_string": self.meaning_unit_string,
            "assigned_code_list": [code.__dict__ for code in self.assigned_code_list],
            "speaking_turn": self.speaking_turn.to_dict() if self.speaking_turn else None
        }

class ParseFormat(BaseModel):
    source_id: str
    quote: str

class ParseResponse(BaseModel):
    parse_list: List[ParseFormat]

class CodeAssignedModel(BaseModel):
    code_name: str
    code_justification: str

class CodeAssignmentResponse(BaseModel):
    meaning_unit_id: int
    codeList: List[CodeAssignedModel]

class CodeResponse(BaseModel):
    assignments: List[CodeAssignmentResponse]

openai_client = get_openai_client()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(client: OpenAI = openai_client, **kwargs):
    """
    Performs a chat completion request with exponential backoff.
    """
    if client is None:
        client = get_openai_client()

    try:
        return client.beta.chat.completions.parse(**kwargs)
    except openai.error.RateLimitError as e:
        logger.error("OpenAI API rate limit exceeded: %s", e)
        raise
    except openai.error.APIError as e:
        logger.error("OpenAI API error: %s", e)
        raise
    except openai.error.Timeout as e:
        logger.error("OpenAI API request timed out: %s", e)
        raise
    except openai.error.InvalidRequestError as e:
        logger.error("OpenAI API invalid request: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error from OpenAI API: %s", e)
        raise

def parse_transcript(
    speaking_turns: List[Dict[str, Any]],
    prompt: str,
    completion_model: str,
    openai_client: Optional[OpenAI] = None
) -> List[Tuple[str, str]]:
    """
    Breaks up multiple speaking turns into smaller meaning units based on 
    criteria in the LLM prompt.
    """
    try:
        response = completion_with_backoff(
            client=openai_client,
            model=completion_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a qualitative research assistant that breaks down multiple speaking "
                        "turns into smaller meaning units based on given instructions."
                    )
                },
                {
                    "role": "user",
                    "content": (
                        f"{prompt}\n\n"
                        f"Speaking Turns:\n{json.dumps(speaking_turns, indent=2)}\n\n"
                    )
                }
            ],
            response_format=ParseResponse,
            temperature=0.2,
            max_tokens=16000,
        )

        if not response.choices:
            logger.error("No choices returned in the response from OpenAI API.")
            return []

        parsed_output = response.choices[0].message.parsed
        if not parsed_output:
            logger.error("Parsed output is empty.")
            return []

        # Validate with pydantic
        parse_response = ParseResponse.parse_obj(parsed_output.dict())

        meaning_units = []
        for unit in parse_response.parse_list:
            meaning_units.append((unit.source_id, unit.quote))

        logger.debug(f"Parsed Meaning Units: {meaning_units}")
        return meaning_units

    except ValidationError as ve:
        logger.error(f"Validation error while parsing transcript: {ve}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while parsing transcript into meaning units: {e}")
        return []

def assign_codes_to_meaning_units(
    meaning_unit_list: List[MeaningUnit],
    coding_instructions: str,
    processed_codes: Optional[List[Dict[str, Any]]] = None,
    codebase: Optional[List[Dict[str, Any]]] = None,
    completion_model: Optional[str] = "gpt-4o-mini",
    context_size: int = 2,
    meaning_units_per_assignment_prompt: int = 1,
    speaker_field: Optional[str] = None,
    content_field: str = 'content',
    full_speaking_turns: Optional[List[Dict[str, Any]]] = None,
    openai_client: Optional[OpenAI] = None,
    # NEW ARG: concurrency
    thread_count: int = 1
) -> List[MeaningUnit]:
    """
    Assigns codes to each MeaningUnit, including context. Now uses a ThreadPoolExecutor
    to process each batch in parallel according to thread_count.
    """
    try:
        if full_speaking_turns is None:
            logger.error("full_speaking_turns must be provided for context.")
            raise ValueError("full_speaking_turns must be provided for context.")

        # Create a mapping from source_id to index
        source_id_to_index = {str(d.get('source_id')): idx for idx, d in enumerate(full_speaking_turns)}

        # Prepare the codebase block
        if processed_codes:
            codes_to_include = codebase if codebase else processed_codes
            unique_codes_strs = set(json.dumps(code, indent=2, sort_keys=True) for code in codes_to_include)
            code_heading = "Full Codebase (all codes with details):"
            codes_block = f"{code_heading}\n{chr(10).join(unique_codes_strs)}\n\n"
        else:
            code_heading = "Guidelines for Inductive Coding:"
            codes_block = f"{code_heading}\nNo predefined codes. Please generate codes based on the following guidelines.\n\n"

        # We store results from each batch in a dict to avoid thread collisions
        # key: meaning_unit_id, value: List[CodeAssigned]
        batch_results_map: Dict[int, List[CodeAssigned]] = {}

        def process_batch(start_idx: int) -> Dict[int, List[CodeAssigned]]:
            """
            Process a batch of meaning units, returning a dict so we can safely
            merge results after all threads finish.
            """
            local_result: Dict[int, List[CodeAssigned]] = {}
            batch = meaning_unit_list[start_idx:start_idx + meaning_units_per_assignment_prompt]
            if not batch:
                return local_result

            # Build a combined prompt for this batch
            full_prompt = f"{coding_instructions}\n\n{codes_block}"

            for unit in batch:
                unit_context = ""
                source_id = unit.speaking_turn.source_id
                unit_idx = source_id_to_index.get(source_id)

                if unit_idx is not None:
                    start_context_idx = max(0, unit_idx - (context_size - 1))
                    end_context_idx = unit_idx + 1
                    context_speaking_turns = full_speaking_turns[start_context_idx:end_context_idx]
                    for st in context_speaking_turns:
                        st_source_id = str(st.get('source_id'))
                        speaker = st.get(speaker_field, "Unknown Speaker") if speaker_field else "Unknown Speaker"
                        content = st.get(content_field, "")
                        unit_context += f"ID: {st_source_id}\nSpeaker: {speaker}\n{content}\n\n"

                    if context_size > 0:
                        full_prompt += (
                            f"Contextual Excerpts for Meaning Unit ID {unit.meaning_unit_id}:\n{unit_context}\n"
                            f"**Important:** Please use the provided contextual excerpts only as background.\n\n"
                        )
                else:
                    logger.warning(f"Source ID {source_id} not found in full speaking turns.")

                speaker = (
                    unit.speaking_turn.metadata.get(speaker_field, "Unknown Speaker")
                    if speaker_field else "Unknown Speaker"
                )
                current_unit_excerpt = f"Quote: {unit.meaning_unit_string}\n\n"
                full_prompt += (
                    f"Current Excerpt For Coding (Meaning Unit ID {unit.meaning_unit_id}) Speaker: {speaker}:\n"
                    f"{current_unit_excerpt}"
                )

            if processed_codes:
                full_prompt += (
                    "**Apply codes exclusively to the current excerpt(s) provided above.**\n\n"
                )
            else:
                full_prompt += (
                    "**Generate codes based on the current excerpt(s) provided above using the guidelines.**\n\n"
                )

            try:
                response = completion_with_backoff(
                    client=openai_client,
                    model=completion_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are tasked with applying qualitative codes to segments of text. "
                                "Use the codebase or guidelines as provided."
                            )
                        },
                        {
                            "role": "user",
                            "content": full_prompt
                        }
                    ],
                    store=True,
                    response_format=CodeResponse,
                    temperature=0.2,
                    max_tokens=16000,
                )

                if not response.choices:
                    logger.error(f"No choices returned for batch starting at index {start_idx}.")
                    return local_result

                code_output = response.choices[0].message.parsed
                code_response = CodeResponse.parse_obj(code_output.dict())

                for assignment in code_response.assignments:
                    # Store the assigned codes in local_result
                    assigned_codes_list: List[CodeAssigned] = []
                    for code_item in assignment.codeList:
                        assigned_codes_list.append(CodeAssigned(
                            code_name=code_item.code_name,
                            code_justification=code_item.code_justification
                        ))
                    local_result[assignment.meaning_unit_id] = assigned_codes_list

            except ValidationError as ve:
                logger.error(f"Validation error while assigning codes for batch {start_idx}: {ve}")
            except Exception as e:
                logger.error(f"Error in code assignment for batch {start_idx}: {e}")

            return local_result

        # Distribute the workload across threads
        from math import ceil
        total_batches = ceil(len(meaning_unit_list) / meaning_units_per_assignment_prompt)

        with ThreadPoolExecutor(max_workers=thread_count) as executor:
            futures = []
            for i in range(0, len(meaning_unit_list), meaning_units_per_assignment_prompt):
                futures.append(executor.submit(process_batch, i))

            for future in as_completed(futures):
                batch_dict = future.result()
                # Merge results
                for mu_id, code_list in batch_dict.items():
                    batch_results_map[mu_id] = code_list

        # Now update our meaning_unit_list with assigned codes
        for mu in meaning_unit_list:
            if mu.meaning_unit_id in batch_results_map:
                mu.assigned_code_list = batch_results_map[mu.meaning_unit_id]

    except Exception as e:
        logger.error(f"An error occurred while assigning codes: {e}")

    return meaning_unit_list
