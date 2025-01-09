# data_handlers.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from qual_functions import parse_transcript, MeaningUnit, SpeakingTurn
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class FlexibleDataHandler:
    def __init__(
        self,
        file_path: str,
        parse_instructions: str,
        completion_model: str,
        content_field: str,
        speaker_field: Optional[str],
        list_field: Optional[str] = None,
        source_id_field: Optional[str] = None,
        filter_rules: Optional[List[Dict[str, Any]]] = None,
        use_parsing: bool = True,
        speaking_turns_per_prompt: int = 1,
        # NEW: thread_count for concurrency
        thread_count: int = 1
    ):
        self.file_path = file_path
        self.parse_instructions = parse_instructions
        self.completion_model = completion_model
        self.content_field = content_field
        self.speaker_field = speaker_field
        self.list_field = list_field
        self.source_id_field = source_id_field
        self.filter_rules = filter_rules
        self.use_parsing = use_parsing
        self.speaking_turns_per_prompt = speaking_turns_per_prompt
        self.thread_count = thread_count  # store it
        self.document_metadata = {}  # Store document-level metadata
        self.full_data = None
        self.filtered_out_source_ids: Set[str] = set()

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
        else:
            data['source_id'] = [f"auto_{i}" for i in range(len(data))]

        self.full_data = data.copy()
        all_source_ids = set(data['source_id'])

        # Apply filter rules
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

    def _parse_chunk_of_data(
        self,
        chunk_data: pd.DataFrame,
        parse_instructions: str,
        completion_model: str
    ) -> List[Dict[str, Any]]:
        """
        Helper method for parsing a chunk of data.
        Returns a list of dicts: [{"source_id": <str>, "parsed_text": <str>}, ...]
        """
        speaking_turns_dicts = []
        for _, record in chunk_data.iterrows():
            speaking_turns_dicts.append({
                "source_id": str(record['source_id']),
                "content": record.get(self.content_field, ""),
                "metadata": record.drop(labels=[self.content_field], errors='ignore').to_dict()
            })

        # parse_transcript returns List[Tuple[source_id, parsed_text]]
        parsed_units = parse_transcript(
            speaking_turns=speaking_turns_dicts,
            prompt=parse_instructions,
            completion_model=completion_model
        )

        results = []
        for source_id, parsed_text in parsed_units:
            results.append({"source_id": source_id, "parsed_text": parsed_text})
        return results

    def transform_data(self, data: pd.DataFrame) -> List[MeaningUnit]:
        """
        Transforms data into MeaningUnit objects, optionally using LLM-based parsing.
        If parsing is off, treat each row as a single meaning unit. Otherwise, parse in batches,
        potentially using threads to speed up LLM calls.
        """
        meaning_units: List[MeaningUnit] = []

        if not self.use_parsing:
            # No parsing needed
            meaning_unit_id_counter = 1
            for _, record in data.iterrows():
                content = record.get(self.content_field, "")
                metadata = record.drop(labels=[self.content_field], errors='ignore').to_dict()
                source_id = str(record['source_id'])

                speaking_turn = SpeakingTurn(
                    source_id=source_id,
                    content=content,
                    metadata=metadata
                )
                mu = MeaningUnit(
                    meaning_unit_id=meaning_unit_id_counter,
                    meaning_unit_string=content,
                    speaking_turn=speaking_turn
                )
                meaning_units.append(mu)
                meaning_unit_id_counter += 1

            logger.debug(f"Transformed data (no parsing) into {len(meaning_units)} meaning units.")
            return meaning_units

        # PARSING is ON
        chunked_data = [
            data.iloc[i: i + self.speaking_turns_per_prompt]
            for i in range(0, len(data), self.speaking_turns_per_prompt)
        ]

        all_parsed_results = []
        # Use concurrency for parsing if thread_count > 1
        with ThreadPoolExecutor(max_workers=self.thread_count) as executor:
            future_to_index = {}
            for idx, chunk in enumerate(chunked_data):
                future = executor.submit(
                    self._parse_chunk_of_data,
                    chunk_data=chunk,
                    parse_instructions=self.parse_instructions,
                    completion_model=self.completion_model
                )
                future_to_index[future] = idx

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    parsed_list = future.result()
                except Exception as e:
                    logger.error(f"Error parsing chunk {idx}: {e}")
                    parsed_list = []

                source_id_map = {}
                for _, row in chunked_data[idx].iterrows():
                    sid = str(row['source_id'])
                    st = SpeakingTurn(
                        source_id=sid,
                        content=row.get(self.content_field, ""),
                        metadata=row.drop(labels=[self.content_field], errors='ignore').to_dict()
                    )
                    source_id_map[sid] = st

                for item in parsed_list:
                    sid = item["source_id"]
                    parsed_text = item["parsed_text"]
                    speaking_turn = source_id_map.get(sid)
                    if not speaking_turn:
                        logger.warning(f"SpeakingTurn not found for source_id {sid}")
                        continue
                    all_parsed_results.append((speaking_turn, parsed_text))

        # Now create MeaningUnit objects
        meaning_unit_id_counter = 1
        for (speaking_turn, parsed_text) in all_parsed_results:
            mu = MeaningUnit(
                meaning_unit_id=meaning_unit_id_counter,
                meaning_unit_string=parsed_text,
                speaking_turn=speaking_turn
            )
            meaning_units.append(mu)
            meaning_unit_id_counter += 1

        logger.debug(f"Transformed data (with parsing) into {len(meaning_units)} meaning units.")
        return meaning_units
