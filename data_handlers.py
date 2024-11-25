# data_handlers.py
import os
import json
import logging
from typing import Any, Dict, List, Optional
import pandas as pd
from qual_functions import parse_transcript, MeaningUnit

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
        use_parsing: bool = True,
        speaking_turns_per_prompt: int = 1
    ):
        self.file_path = file_path
        self.parse_instructions = parse_instructions
        self.completion_model = completion_model
        self.content_field = content_field
        self.speaker_field = speaker_field
        self.list_field = list_field
        self.use_parsing = use_parsing
        self.speaking_turns_per_prompt = speaking_turns_per_prompt
        self.document_metadata = {}  # Store document-level metadata

    def load_data(self) -> pd.DataFrame:
        """
        Loads data from the JSON file and extracts the content list into a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the data.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                raw_data = json.load(file)
            logger.debug(f"Loaded data from '{self.file_path}'.")
        except Exception as e:
            logger.error(f"Failed to load data from '{self.file_path}': {e}")
            raise

        # Extract document-level metadata
        self.document_metadata = {k: v for k, v in raw_data.items() if k != self.list_field}

        # Extract the list of content items
        content_list = raw_data.get(self.list_field, [])
        if not content_list:
            logger.error(f"No content found under the list_field '{self.list_field}'.")
            raise ValueError(f"No content found under the list_field '{self.list_field}'.")

        # Create DataFrame from the content list
        data = pd.DataFrame(content_list)

        # Ensure that the 'id' column is treated as 'source_id' if present
        if 'id' in data.columns and 'source_id' not in data.columns:
            data.rename(columns={'id': 'source_id'}, inplace=True)
            logger.debug("Renamed 'id' column to 'source_id'.")

        logger.debug(f"Data shape after loading: {data.shape}")
        return data

    def transform_data(self, data: pd.DataFrame) -> List[MeaningUnit]:
        """
        Transforms data into MeaningUnit objects, processing multiple speaking turns per prompt.

        Args:
            data (pd.DataFrame): DataFrame containing the data.

        Returns:
            List[MeaningUnit]: List of MeaningUnit objects.
        """
        meaning_units = []
        source_id_counter = 1  # Counter for generating unique source_ids
        meaning_unit_id_counter = 1  # Counter for assigning unique meaning_unit_ids

        if self.use_parsing:
            # Group speaking turns into batches
            for i in range(0, len(data), self.speaking_turns_per_prompt):
                batch = data.iloc[i:i + self.speaking_turns_per_prompt]
                speaking_turns = []
                source_id_map = {}  # Map source_id to metadata
                for _, record in batch.iterrows():
                    content = record.get(self.content_field, "")
                    metadata = record.drop(labels=[self.content_field], errors='ignore').to_dict()
                    source_id = record.get('source_id')
                    if source_id is None:
                        # Generate a unique source_id if not present
                        source_id = f"auto_{source_id_counter}"
                        metadata['source_id'] = source_id  # Assign back to the metadata
                        logger.warning(f"Assigned auto-generated 'source_id' '{source_id}' to speaking turn.")
                        source_id_counter += 1  # Increment for the next source_id
                    else:
                        # Ensure uniqueness even if source_id is present
                        source_id = f"{source_id}_{source_id_counter}"
                        metadata['source_id'] = source_id
                        source_id_counter += 1  # Increment for the next source_id
                    speaking_turn = {
                        "source_id": source_id,
                        "content": content,
                        "metadata": metadata
                    }
                    speaking_turns.append(speaking_turn)
                    source_id_map[source_id] = metadata

                # Parse the batch of speaking turns
                parsed_units = parse_transcript(
                    speaking_turns=speaking_turns,
                    prompt=self.parse_instructions,
                    completion_model=self.completion_model
                )
                for source_id, pu in parsed_units:
                    metadata = source_id_map.get(source_id, {})
                    # Ensure 'source_id' is included in metadata
                    metadata['source_id'] = source_id
                    # Create MeaningUnit with meaning_unit_id independent of source_id
                    meaning_unit = MeaningUnit(
                        meaning_unit_id=meaning_unit_id_counter,
                        meaning_unit_string=pu,
                        metadata=metadata
                    )
                    meaning_units.append(meaning_unit)
                    meaning_unit_id_counter += 1  # Increment the counter for the next meaning unit
        else:
            for _, record in data.iterrows():
                content = record.get(self.content_field, "")
                metadata = record.drop(labels=[self.content_field], errors='ignore').to_dict()
                source_id = record.get('source_id')
                if source_id is None:
                    # Generate a unique source_id if not present
                    source_id = f"auto_{source_id_counter}"
                    metadata['source_id'] = source_id  # Assign back to the metadata
                    logger.warning(f"Assigned auto-generated 'source_id' '{source_id}' to speaking turn.")
                    source_id_counter += 1  # Increment for the next source_id
                else:
                    # Ensure uniqueness even if source_id is present
                    source_id = f"{source_id}_{source_id_counter}"
                    metadata['source_id'] = source_id
                    source_id_counter += 1  # Increment for the next source_id
                # Create MeaningUnit with meaning_unit_id independent of source_id
                meaning_unit = MeaningUnit(
                    meaning_unit_id=meaning_unit_id_counter,
                    meaning_unit_string=content,
                    metadata=metadata
                )
                meaning_units.append(meaning_unit)
                meaning_unit_id_counter += 1  # Increment the counter for the next meaning unit

        logger.debug(f"Transformed data into {len(meaning_units)} meaning units.")
        return meaning_units
