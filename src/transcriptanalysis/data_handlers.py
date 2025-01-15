# data_handlers.py

import json
import logging
import os
import uuid  # Import UUID module
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from transcriptanalysis.qual_functions import MeaningUnit, PreliminarySegment
from transcriptanalysis.config_schemas import ProviderEnum, LLMConfig
from transcriptanalysis.langchain_llm import LangChainLLM

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# NEW: Pydantic models for structured parse responses
# ------------------------------------------------------------------
from pydantic import BaseModel, ValidationError
from typing import List


class ParseUnit(BaseModel):
    """
    Represents a single meaning unit (smaller chunk) after parsing.
    """
    source_id: str
    quote: str


class FullParseResponse(BaseModel):
    """
    Represents the entire structured output from the parsing step.
    """
    parse_list: List[ParseUnit]


class FlexibleDataHandler:
    def __init__(
        self,
        file_path: str,
        parse_instructions: str,
        completion_model: str,
        content_field: str,
        context_fields: Optional[List[str]] = None,  # CHANGED
        list_field: Optional[str] = None,
        source_id_field: Optional[str] = None,
        filter_rules: Optional[List[Dict[str, Any]]] = None,
        use_parsing: bool = True,
        preliminary_segments_per_prompt: int = 1,  # Renamed
        thread_count: int = 1
    ):
        self.file_path = file_path
        self.parse_instructions = parse_instructions
        self.completion_model = completion_model
        self.content_field = content_field
        self.context_fields = context_fields
        self.list_field = list_field
        self.source_id_field = source_id_field
        self.filter_rules = filter_rules
        self.use_parsing = use_parsing
        self.preliminary_segments_per_prompt = preliminary_segments_per_prompt
        self.thread_count = thread_count
        self.document_metadata = {}  # Store document-level metadata
        self.full_data = None
        self.filtered_out_source_ids: Set[str] = set()

        # Initialize counters
        self.meaning_unit_counter = 1  # Independent counter for meaning_unit_id

        # ------------------------------------------------------------------
        # NEW: Initialize LangChainLLM for parsing (if needed)
        # ------------------------------------------------------------------
        self.llm = None
        if self.use_parsing:
            try:
                # Build an LLMConfig (example: defaulting to OpenAI with a 0.2 temperature)
                self.llm_config = LLMConfig(
                    provider=ProviderEnum.OPENAI,
                    model_name=self.completion_model,
                    temperature=0.2,
                    max_tokens=16000,
                    api_key=os.getenv('OPENAI_API_KEY', '')
                )
                self.llm = LangChainLLM(self.llm_config)
            except Exception as e:
                logger.warning(
                    f"Could not initialize LangChainLLM with model={self.completion_model}: {e}"
                )
                self.llm = None

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the JSON file into a DataFrame and applies filter rules if any.
        """
        try:
            with Path(self.file_path).open('r', encoding='utf-8') as file:
                raw_data = json.load(file)
            logger.debug(f"Loaded data from '{self.file_path}'.")
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load or parse data from '{self.file_path}': {e}")
            raise

        # Determine how the data is structured
        if isinstance(raw_data, list):
            self.document_metadata = {}
            content_list = raw_data
        elif isinstance(raw_data, dict):
            if self.list_field:
                self.document_metadata = {
                    k: v for k, v in raw_data.items() if k != self.list_field
                }
                content_list = raw_data.get(self.list_field, [])
                if not content_list:
                    logger.error(f"No content found under the list_field '{self.list_field}'.")
                    raise ValueError(f"No content found under the list_field '{self.list_field}'.")
            else:
                self.document_metadata = {
                    k: v for k, v in raw_data.items() if not isinstance(v, list)
                }
                content_list = [v for v in raw_data.values() if isinstance(v, list)]
                if content_list:
                    content_list = content_list[0]
                else:
                    logger.error("No list of content items found in the data.")
                    raise ValueError("No list of content items found in the data.")
        else:
            logger.error(f"Unexpected data format in '{self.file_path}'. Expected dict or list.")
            raise ValueError(f"Unexpected data format in '{self.file_path}'.")

        data = pd.DataFrame(content_list)

        # Ensure we have a source_id
        if self.source_id_field and self.source_id_field in data.columns:
            data['source_id'] = data[self.source_id_field].astype(str)
            # Optionally, ensure uniqueness if necessary
        else:
            # Generate unique UUIDs for source_id
            data['source_id'] = [str(uuid.uuid4()) for _ in range(len(data))]

        self.full_data = data.copy()
        all_source_ids = set(data['source_id'])

        # Apply filter rules if any
        if self.filter_rules:
            mask = pd.Series(True, index=data.index)
            for rule in self.filter_rules:
                field = rule.get('field')
                operator = rule.get('operator', 'equals')
                value = rule.get('value')

                if field not in data.columns:
                    logger.warning(f"Field '{field}' not found in data. Skipping filter rule.")
                    continue

                if operator == 'equals':
                    mask &= (data[field] == value)
                elif operator == 'not_equals':
                    mask &= (data[field] != value)
                elif operator == 'contains':
                    mask &= data[field].astype(str).str.contains(str(value), na=False, regex=False)
                else:
                    logger.warning(f"Operator '{operator}' is not supported. Skipping this filter rule.")

            data = data[mask]
            logger.debug(f"Data shape after applying filter rules: {data.shape}")

        # Identify filtered out source_ids
        filtered_source_ids = all_source_ids - set(data['source_id'])
        self.filtered_out_source_ids = filtered_source_ids
        logger.debug(f"Data shape after loading: {data.shape}")

        return data

    def _run_langchain_parse_chunk(
        self,
        preliminary_segments: List[Dict[str, Any]],
        prompt: str
    ) -> List[Dict[str, str]]:
        """
        Uses LangChainLLM.structured_generate() to parse a chunk of preliminary_segments into smaller meaning units.
        Returns a list of dicts with keys "source_id" and "parsed_text".
        """

        if not self.llm:
            logger.error("LLM was not initialized; cannot parse.")
            return []

        # Build the combined prompt (system role + user instructions + data)
        system_content = (
            "You are a qualitative research assistant that breaks down multiple preliminary segments "
            "into smaller meaning units based on given instructions."
        )

        combined_prompt = f"""
System instructions:
{system_content}

User instructions:
{prompt}

Preliminary Segments (JSON):
{json.dumps(preliminary_segments, indent=2)}
        """

        try:
            parsed_response: FullParseResponse = self.llm.structured_generate(
                combined_prompt,
                FullParseResponse
            )

            if not parsed_response or not isinstance(parsed_response, FullParseResponse):
                logger.error("Parsed output is not an instance of FullParseResponse.")
                return []

            if not parsed_response.parse_list:
                logger.error("Parsed output is empty or missing 'parse_list'.")
                return []

            results = []
            for unit in parsed_response.parse_list:
                results.append({
                    "source_id": unit.source_id,
                    "parsed_text": unit.quote
                })

            return results

        except ValidationError as ve:
            logger.error(f"Validation error while parsing chunk: {ve}")
            return []
        except Exception as e:
            logger.error(f"An error occurred while parsing chunk: {e}")
            return []

    def _parse_chunk_of_data(
        self,
        chunk_data: pd.DataFrame,
        parse_instructions: str,
        batch_index: int
    ) -> Tuple[int, List[Dict[str, str]]]:
        """
        Helper method for parsing a chunk of data.
        Returns a tuple of (batch_index, parsed_units).
        """
        preliminary_segments_dicts = []
        for _, record in chunk_data.iterrows():
            preliminary_segments_dicts.append({
                "source_id": str(record['source_id']),
                "content": record.get(self.content_field, ""),
                "metadata": record.drop(labels=[self.content_field], errors='ignore').to_dict()
            })

        parsed_units = self._run_langchain_parse_chunk(
            preliminary_segments=preliminary_segments_dicts,
            prompt=parse_instructions
        )

        return (batch_index, parsed_units)

    def transform_data(self, data: pd.DataFrame) -> List[MeaningUnit]:
        """
        Transforms data into MeaningUnit objects, optionally using LLM-based parsing.
        If parsing is off, treat each row as a single meaning unit.
        Otherwise, parse in batches using concurrency.
        """
        meaning_units: List[MeaningUnit] = []

        if not self.use_parsing:
            # No parsing needed
            for _, record in data.iterrows():
                content = record.get(self.content_field, "")
                metadata = record.drop(labels=[self.content_field], errors='ignore').to_dict()
                source_id = str(record['source_id'])

                # Generate UUID for meaning_unit_uuid
                meaning_unit_uuid = str(uuid.uuid4())

                preliminary_segment = PreliminarySegment(
                    source_id=source_id,
                    content=content,
                    metadata=metadata
                )
                mu = MeaningUnit(
                    meaning_unit_id=self.meaning_unit_counter,
                    meaning_unit_uuid=meaning_unit_uuid,
                    source_id=source_id,  # NEW: Link meaning unit to the same source_id
                    meaning_unit_string=content,
                    assigned_code_list=[],
                    preliminary_segment=preliminary_segment
                )
                meaning_units.append(mu)
                self.meaning_unit_counter += 1

            logger.debug(f"Transformed data (no parsing) into {len(meaning_units)} meaning units.")
            return meaning_units

        # PARSING is ON
        chunked_data = [
            data.iloc[i: i + self.preliminary_segments_per_prompt]
            for i in range(0, len(data), self.preliminary_segments_per_prompt)
        ]

        all_parsed_results: List[Tuple[int, List[Dict[str, str]]]] = []
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            futures = {}
            for idx, chunk in enumerate(chunked_data):
                future = executor.submit(
                    self._parse_chunk_of_data,
                    chunk_data=chunk,
                    parse_instructions=self.parse_instructions,
                    batch_index=idx
                )
                futures[future] = idx

            for future in as_completed(futures):
                batch_index, parsed_list = future.result()
                all_parsed_results.append((batch_index, parsed_list))

        # Sort the results based on batch_index to maintain order
        all_parsed_results.sort(key=lambda x: x[0])

        for _, parsed_list in all_parsed_results:
            for item in parsed_list:
                sid = item["source_id"]
                parsed_text = item["parsed_text"]
                # Retrieve the original record to get metadata
                matching_records = data[data['source_id'] == sid]
                if not matching_records.empty:
                    record = matching_records.iloc[0]
                    metadata = record.drop(labels=[self.content_field], errors='ignore').to_dict()
                else:
                    metadata = {}

                preliminary_segment = PreliminarySegment(
                    source_id=sid,
                    content=record.get(self.content_field, "") if not matching_records.empty else "",
                    metadata=metadata
                )

                # Generate UUID for meaning_unit_uuid
                meaning_unit_uuid = str(uuid.uuid4())

                mu = MeaningUnit(
                    meaning_unit_id=self.meaning_unit_counter,
                    meaning_unit_uuid=meaning_unit_uuid,
                    source_id=sid,  # NEW: Link meaning unit to the same source_id
                    meaning_unit_string=parsed_text,
                    assigned_code_list=[],
                    preliminary_segment=preliminary_segment
                )
                meaning_units.append(mu)
                self.meaning_unit_counter += 1

        logger.debug(f"Transformed data (with parsing) into {len(meaning_units)} meaning units.")
        return meaning_units
