# qual_functions.py

import logging
import json
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from pydantic import BaseModel, ValidationError

from langchain_llm import LangChainLLM
from config_schemas import LLMConfig

logger = logging.getLogger(__name__)

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

# -----------------------
# MODELS FOR STRUCTURED OUTPUT
# -----------------------
class CodeAssignedModel(BaseModel):
    code_name: str
    code_justification: str

class CodeAssignmentResponse(BaseModel):
    meaning_unit_id: int
    codeList: List[CodeAssignedModel]

class CodeResponse(BaseModel):
    assignments: List[CodeAssignmentResponse]


def assign_codes_to_meaning_units(
    meaning_unit_list: List[MeaningUnit],
    coding_instructions: str,
    processed_codes: Optional[List[Dict[str, Any]]] = None,
    codebase: Optional[List[Dict[str, Any]]] = None,
    completion_model: str = "gpt-4",
    context_size: int = 2,
    meaning_units_per_assignment_prompt: int = 1,
    speaker_field: Optional[str] = None,
    content_field: str = 'content',
    full_speaking_turns: Optional[List[Dict[str, Any]]] = None,
    # Instead of openai_client, we pass an LLM config or instance
    thread_count: int = 1,
    llm_config: Optional[LLMConfig] = None
) -> List[MeaningUnit]:
    """
    Assigns codes to each MeaningUnit, including context. 
    Uses concurrency to parallelize LLM calls.
    """

    if not llm_config:
        logger.error("No LLMConfig provided.")
        return meaning_unit_list

    # Build a single instance of LangChainLLM to reuse across concurrency
    llm = LangChainLLM(llm_config)

    if full_speaking_turns is None:
        logger.error("full_speaking_turns must be provided for context.")
        return meaning_unit_list

    # Create a mapping from source_id to index for context retrieval
    source_id_to_index = {str(d.get('source_id')): idx for idx, d in enumerate(full_speaking_turns)}

    # Prepare the codebase block
    if processed_codes:
        codes_to_include = codebase if codebase else processed_codes
        unique_codes_strs = set(json.dumps(code, indent=2, sort_keys=True) for code in codes_to_include)
        code_heading = "Full Codebase (all codes with details):"
        codes_block = f"{code_heading}\n{chr(10).join(unique_codes_strs)}\n\n"
    else:
        code_heading = "Guidelines for Inductive Coding:"
        codes_block = (
            f"{code_heading}\nNo predefined codes. Please generate codes based on the following guidelines.\n\n"
        )

    # We store results from each batch in a dict to avoid collisions
    batch_results_map: Dict[int, List[CodeAssigned]] = {}

    def process_batch(start_idx: int) -> Dict[int, List[CodeAssigned]]:
        """
        Process a batch of meaning units, returning a dict so we can safely
        merge results after threads finish.
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

            speaker = (
                unit.speaking_turn.metadata.get(speaker_field, "Unknown Speaker") 
                if speaker_field else "Unknown Speaker"
            )
            current_unit_excerpt = f"Quote: {unit.meaning_unit_string}\n\n"

            full_prompt += (
                f"Contextual Excerpts for Meaning Unit ID {unit.meaning_unit_id}:\n{unit_context}\n"
                f"Current Excerpt For Coding (Meaning Unit ID {unit.meaning_unit_id}) Speaker: {speaker}:\n"
                f"{current_unit_excerpt}"
            )

        if processed_codes:
            full_prompt += "**Apply codes exclusively to the excerpt(s) provided above.**\n\n"
        else:
            full_prompt += "**Generate codes based on the excerpt(s) provided above using the guidelines.**\n\n"

        # -----------------------------
        # NEW: Use structured_generate
        # -----------------------------
        try:
            code_response = llm.structured_generate(full_prompt, CodeResponse)
        except Exception as e:
            # If something goes wrong with structured output, fallback to empty
            logger.warning(f"Structured generation failed for batch {start_idx}, using fallback empty assignment. Error: {e}")
            code_response = CodeResponse(assignments=[])

        # Merge results
        for assignment in code_response.assignments:
            assigned_codes_list: List[CodeAssigned] = []
            for code_item in assignment.codeList:
                assigned_codes_list.append(CodeAssigned(
                    code_name=code_item.code_name,
                    code_justification=code_item.code_justification
                ))
            local_result[assignment.meaning_unit_id] = assigned_codes_list

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

    # Update meaning_unit_list with assigned codes
    for mu in meaning_unit_list:
        if mu.meaning_unit_id in batch_results_map:
            mu.assigned_code_list = batch_results_map[mu.meaning_unit_id]

    return meaning_unit_list
