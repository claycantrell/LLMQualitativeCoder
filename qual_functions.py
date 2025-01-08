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

logger = logging.getLogger(__name__)

# ---------------------------------------------
# 14. Improve OpenAI Client Initialization
# ---------------------------------------------

def get_openai_client() -> OpenAI:
    """
    Retrieves an instance of the OpenAI client. This function can be
    called multiple times to create fresh or different client instances
    based on dynamic configurations.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return OpenAI(api_key=openai_api_key)

# ---------------------------------------------
# 15. Utilize More Specific Data Classes
#    (All fields typed; optional methods added)
# ---------------------------------------------

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
        # Ensure source_id is not duplicated in metadata
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

# ---------------------------------------------
# 18. Use Pydantic for Response Validation
#    (Pydantic models to ensure response schema)
# ---------------------------------------------

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

# ---------------------------------------------
# 16. Enhance completion_with_backoff Function
#    (Accepts client as parameter, handles API errors)
# ---------------------------------------------
openai_client = get_openai_client()

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(
    client: OpenAI = openai_client,
    **kwargs
):
    """
    Performs a chat completion request with exponential backoff and optional
    custom OpenAI client. Also handles specific API exceptions to improve
    robustness.
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

# ---------------------------------------------
# Core Functions
# ---------------------------------------------

def parse_transcript(
    speaking_turns: List[Dict[str, Any]],
    prompt: str,
    completion_model: str,
    openai_client: Optional[OpenAI] = None
) -> List[Tuple[str, str]]:
    """
    Breaks up multiple speaking turns into smaller meaning units based on 
    criteria in the LLM prompt. Returns a list of tuples containing (source_id, meaning unit string).
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
    except (OSError, json.JSONDecodeError) as e:
        logger.error(f"File or JSON error while parsing transcript: {e}")
        return []
    except Exception as e:
        logger.error(f"An unexpected error occurred while parsing transcript into meaning units: {e}")
        return []

# ---------------------------------------------
# 17. Optimize assign_codes_to_meaning_units
#    (Batching, precompute prompt parts)
# ---------------------------------------------

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
    openai_client: Optional[OpenAI] = None
) -> List[MeaningUnit]:
    """
    Assigns codes to each MeaningUnit object, including contextual information
    from speaking turns. Processes multiple meaning units per prompt 
    based on configuration.

    Args:
        meaning_unit_list (List[MeaningUnit]]): List of meaning units to process.
        coding_instructions (str): Coding instructions prompt.
        processed_codes (Optional[List[Dict[str, Any]]]): Full codebase for deductive coding.
        codebase (Optional[List[Dict[str, Any]]]): Same as processed_codes, included for clarity.
        completion_model (str): Language model to use for code assignment.
        context_size (int): How many speaking turns to include as context (backward in the transcript).
        meaning_units_per_assignment_prompt (int): Number of meaning units to process per assignment prompt.
        speaker_field (Optional[str]): The field name for speaker information.
        content_field (str): The field name for the text content in speaking turns.
        full_speaking_turns (Optional[List[Dict[str, Any]]]): List of the original speaking turns for context.
        openai_client (Optional[OpenAI]): Pass a custom initialized OpenAI client.

    Returns:
        List[MeaningUnit]: Updated list with assigned codes.
    """
    try:
        if full_speaking_turns is None:
            logger.error("full_speaking_turns must be provided for context.")
            raise ValueError("full_speaking_turns must be provided for context.")

        # Create a mapping from source_id to index in full_speaking_turns
        source_id_to_index = {str(d.get('source_id')): idx for idx, d in enumerate(full_speaking_turns)}

        # Precompute constant parts of the prompt that do not change per meaning unit/batch
        # For deductive coding with a known codebase
        if processed_codes:
            codes_to_include = codebase if codebase else processed_codes
            # Collect unique codes in a stable JSON representation
            unique_codes_strs = set(json.dumps(code, indent=2, sort_keys=True) for code in codes_to_include)
            code_heading = "Full Codebase (all codes with details):"
            codes_block = f"{code_heading}\n{chr(10).join(unique_codes_strs)}\n\n"
        else:
            code_heading = "Guidelines for Inductive Coding:"
            codes_block = f"{code_heading}\nNo predefined codes. Please generate codes based on the following guidelines.\n\n"

        def process_batch(start_idx: int) -> None:
            batch = meaning_unit_list[start_idx:start_idx + meaning_units_per_assignment_prompt]
            if not batch:
                return

            # Build the prompt incrementally
            full_prompt = f"{coding_instructions}\n\n{codes_block}"

            # For each meaning unit in the batch, add context and excerpt
            for unit in batch:
                unit_context = ""
                source_id = unit.speaking_turn.source_id
                unit_idx = source_id_to_index.get(source_id)

                if unit_idx is not None:
                    start_context_idx = max(0, unit_idx - (context_size - 1))
                    end_context_idx = unit_idx + 1  # Include the current speaking turn

                    context_speaking_turns = full_speaking_turns[start_context_idx:end_context_idx]
                    for st in context_speaking_turns:
                        st_source_id = str(st.get('source_id'))
                        speaker = st.get(speaker_field, "Unknown Speaker") if speaker_field else "Unknown Speaker"
                        content = st.get(content_field, "")
                        unit_context += f"ID: {st_source_id}\nSpeaker: {speaker}\n{content}\n\n"

                    if context_size > 0:
                        full_prompt += (
                            f"Contextual Excerpts for Meaning Unit ID {unit.meaning_unit_id}:\n{unit_context}\n"
                            f"**Important:** Please use the provided contextual excerpts **only** as background information "
                            f"to understand the current excerpt better.\n\n"
                        )
                else:
                    logger.warning(f"Source ID {source_id} not found in full speaking turns.")

                speaker = unit.speaking_turn.metadata.get(speaker_field, "Unknown Speaker") if speaker_field else "Unknown Speaker"
                current_unit_excerpt = f"Quote: {unit.meaning_unit_string}\n\n"
                full_prompt += (
                    f"Current Excerpt For Coding (Meaning Unit ID {unit.meaning_unit_id}) Speaker: {speaker}:\n"
                    f"{current_unit_excerpt}"
                )

            if processed_codes:
                full_prompt += (
                    "**Apply codes exclusively to the current excerpt(s) provided above. "
                    "Do not assign codes to the contextual excerpts.**\n\n"
                )
            else:
                full_prompt += (
                    "**Generate codes based on the current excerpt(s) provided above using the guidelines.**\n\n"
                )

            logger.debug(f"Full Prompt for Batch starting at index {start_idx}:\n{full_prompt}")

            try:
                response = completion_with_backoff(
                    client=openai_client,
                    model=completion_model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are tasked with applying qualitative codes to segments of text. "
                                "The purpose of this task is to identify all codes that best describe each text segment "
                                "based on the provided instructions."
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
                    return

                code_output = response.choices[0].message.parsed

                # Validate with pydantic
                code_response = CodeResponse.parse_obj(code_output.dict())

                for assignment in code_response.assignments:
                    matching_units = [mu for mu in batch if mu.meaning_unit_id == assignment.meaning_unit_id]
                    if not matching_units:
                        logger.warning(f"No matching meaning unit found for meaning_unit_id {assignment.meaning_unit_id}.")
                        continue

                    meaning_unit = matching_units[0]
                    # Convert CodeAssignedModel => CodeAssigned
                    for code_item in assignment.codeList:
                        meaning_unit.assigned_code_list.append(
                            CodeAssigned(
                                code_name=code_item.code_name,
                                code_justification=code_item.code_justification
                            )
                        )

            except ValidationError as ve:
                logger.error(f"Validation error while assigning codes for batch starting at index {start_idx}: {ve}")
            except (OSError, json.JSONDecodeError) as e:
                logger.error(f"File or JSON error while assigning codes for batch starting at index {start_idx}: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while assigning codes for batch starting at index {start_idx}: {e}")

        # Process batches sequentially
        for i in range(0, len(meaning_unit_list), meaning_units_per_assignment_prompt):
            process_batch(i)

    except ValueError as ve:
        logger.error(f"Validation or value error while assigning codes: {ve}")
    except Exception as e:
        logger.error(f"An error occurred while assigning codes: {e}")

    return meaning_unit_list
