# data_handlers.py

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from qual_functions import parse_transcript, MeaningUnit, SpeakingTurn

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
        speaking_turns_per_prompt: int = 1
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
        self.document_metadata = {}  # Store document-level metadata
        self.full_data = None  # To store pre-filtered data
        self.filtered_out_source_ids: Set[str] = set()  # Store filtered-out source_ids

    def load_data(self) -> pd.DataFrame:
        """
        (19) Optimize Data Loading with Pandas:
            * Use vectorized operations for filtering and ensure necessary fields 
              are loaded and processed efficiently.
            * Benefit: Enhances performance and readability when handling large datasets.

        Loads data from the JSON file and extracts the content list into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the data.
        """
        try:
            with Path(self.file_path).open('r', encoding='utf-8') as file:
                raw_data = json.load(file)
            logger.debug(f"Loaded data from '{self.file_path}'.")
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to load or parse data from '{self.file_path}': {e}")
            raise

        # Determine how the data is structured (list or dict)
        if isinstance(raw_data, list):
            self.document_metadata = {}
            content_list = raw_data
        elif isinstance(raw_data, dict):
            if self.list_field:
                # Extract document-level metadata
                self.document_metadata = {
                    k: v for k, v in raw_data.items() if k != self.list_field
                }
                # Extract the list of content items
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
            raise ValueError(f"Unexpected data format in '{self.file_path}'. Expected dict or list.")

        # Convert content_list to a pandas DataFrame
        data = pd.DataFrame(content_list)

        # Ensure source_id column is present and string-typed
        if self.source_id_field and self.source_id_field in data.columns:
            data['source_id'] = data[self.source_id_field].astype(str)
        else:
            # Assign a simple auto-increment source_id if none was provided
            data['source_id'] = [f"auto_{i}" for i in range(len(data))]

        # Store a copy of the full, pre-filtered data
        self.full_data = data.copy()

        # Keep track of all source_ids before filtering
        all_source_ids = set(data['source_id'])

        # Apply filter rules in a vectorized manner (if any)
        if self.filter_rules:
            # Start with a "keep all" mask
            mask = pd.Series(True, index=data.index)

            for rule in self.filter_rules:
                field = rule.get('field')
                operator = rule.get('operator', 'equals')
                value = rule.get('value')

                if field not in data.columns:
                    logger.warning(f"Field '{field}' not found in data columns. Skipping filter rule.")
                    continue

                if operator == 'equals':
                    mask &= (data[field] == value)
                elif operator == 'not_equals':
                    mask &= (data[field] != value)
                elif operator == 'contains':
                    # Use regex=False if you want a plain substring match, else remove to allow regex
                    mask &= data[field].astype(str).str.contains(str(value), na=False, regex=False)
                else:
                    logger.warning(f"Operator '{operator}' is not supported. Skipping this filter rule.")

            data = data[mask]
            logger.debug(f"Data shape after applying filter rules: {data.shape}")

        # Identify the source_ids that were filtered out
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

        Returns a list of parsed transcripts in the format:
            [
              {"source_id": <str>, "parsed_text": <str>},
              ...
            ]
        """
        # Build speaking_turns_dicts for this chunk
        speaking_turns_dicts = []
        for _, record in chunk_data.iterrows():
            speaking_turns_dicts.append({
                "source_id": str(record['source_id']),
                "content": record.get(self.content_field, ""),
                "metadata": record.drop(labels=[self.content_field], errors='ignore').to_dict()
            })

        # parse_transcript should handle the entire batch of speaking turns
        # parse_transcript is expected to return List[Tuple[source_id, parsed_text]]
        parsed_units = parse_transcript(
            speaking_turns=speaking_turns_dicts,
            prompt=parse_instructions,
            completion_model=completion_model
        )

        # Reshape so each returned element can map back easily to the original speaking_turn
        result = []
        for source_id, parsed_text in parsed_units:
            result.append({"source_id": source_id, "parsed_text": parsed_text})

        return result

    def transform_data(self, data: pd.DataFrame) -> List[MeaningUnit]:
        """
        (20) Enhance transform_data Method:
            * Utilize sequential processing for large datasets.
            * Ensure parse_transcript handles batch processing effectively.
            * Benefit: Maintains processing speed and scalability without parallelism.

        Transforms data into MeaningUnit objects, optionally using LLM-based parsing.
        If self.use_parsing is True, speaking turns are batched and parsed.

        Args:
            data (pd.DataFrame): DataFrame containing the data.

        Returns:
            List[MeaningUnit]: List of MeaningUnit objects.
        """
        meaning_units: List[MeaningUnit] = []

        # If parsing is off, just treat each row as a single meaning unit (no concurrency needed)
        if not self.use_parsing:
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
                meaning_unit = MeaningUnit(
                    meaning_unit_id=meaning_unit_id_counter,
                    meaning_unit_string=content,
                    speaking_turn=speaking_turn
                )
                meaning_units.append(meaning_unit)
                meaning_unit_id_counter += 1

            logger.debug(f"Transformed data (no parsing) into {len(meaning_units)} meaning units.")
            return meaning_units

        # Otherwise, we parse in batches using self.speaking_turns_per_prompt
        chunked_data = [
            data.iloc[i: i + self.speaking_turns_per_prompt]
            for i in range(0, len(data), self.speaking_turns_per_prompt)
        ]

        all_parsed_results = []
        for idx, chunk in enumerate(chunked_data):
            try:
                parsed_list = self._parse_chunk_of_data(
                    chunk_data=chunk,
                    parse_instructions=self.parse_instructions,
                    completion_model=self.completion_model
                )  # List[{"source_id": str, "parsed_text": str}]
            except Exception as e:
                logger.error(f"Error parsing chunk {idx}: {e}")
                parsed_list = []

            # Keep track of which source_id belongs to which chunk row
            # so we can reconstruct the SpeakingTurn
            source_id_map = {}
            for _, row in chunk.iterrows():
                row_source_id = str(row['source_id'])
                st = SpeakingTurn(
                    source_id=row_source_id,
                    content=row.get(self.content_field, ""),
                    metadata=row.drop(labels=[self.content_field], errors='ignore').to_dict()
                )
                source_id_map[row_source_id] = st

            # Create a combined result for each source_id in this chunk
            for item in parsed_list:
                sid = item["source_id"]
                parsed_text = item["parsed_text"]
                speaking_turn = source_id_map.get(sid)
                if not speaking_turn:
                    logger.warning(f"SpeakingTurn not found for source_id {sid}")
                    continue
                all_parsed_results.append((speaking_turn, parsed_text))

        # Now assign unique MeaningUnit IDs
        meaning_unit_id_counter = 1
        for speaking_turn, pu in all_parsed_results:
            meaning_unit = MeaningUnit(
                meaning_unit_id=meaning_unit_id_counter,
                meaning_unit_string=pu,
                speaking_turn=speaking_turn
            )
            meaning_units.append(meaning_unit)
            meaning_unit_id_counter += 1

        logger.debug(f"Transformed data (with parsing) into {len(meaning_units)} meaning units.")
        return meaning_units
