# data_handlers.py
import os
import json
import logging
from typing import Any, Dict, List, Tuple
from pydantic import BaseModel
from qual_functions import parse_transcript, MeaningUnit

logger = logging.getLogger(__name__)

class FlexibleDataHandler:
    def __init__(
        self,
        file_path: str,
        parse_instructions: str,
        completion_model: str,
        model_class: Any,
        content_field: str,
        use_parsing: bool = True,
        speaking_turns_per_prompt: int = 1  # New parameter
    ):
        self.file_path = file_path
        self.parse_instructions = parse_instructions
        self.completion_model = completion_model
        self.model_class = model_class
        self.content_field = content_field
        self.use_parsing = use_parsing
        self.speaking_turns_per_prompt = speaking_turns_per_prompt  # Store the new parameter

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Loads data from the JSON file and validates it against the dynamic Pydantic model.

        Returns:
            List[Dict[str, Any]]: List of validated data records.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            logger.debug(f"Loaded {len(data)} records from '{self.file_path}'.")
        except Exception as e:
            logger.error(f"Failed to load data from '{self.file_path}': {e}")
            raise

        validated_data = []
        for record in data:
            try:
                validated_record = self.model_class(**record)
                validated_data.append(validated_record.model_dump())
            except Exception as e:
                logger.warning(f"Validation failed for record {record.get('id', 'unknown')}: {e}")
                continue

        logger.debug(f"Validated {len(validated_data)} records.")
        return validated_data

    def transform_data(self, validated_data: List[Dict[str, Any]]) -> List[MeaningUnit]:
        """
        Transforms validated data into MeaningUnit objects, processing multiple speaking turns per prompt.

        Args:
            validated_data (List[Dict[str, Any]]): List of validated data records.

        Returns:
            List[MeaningUnit]: List of MeaningUnit objects.
        """
        meaning_units = []
        unique_id_counter = 1  # Initialize a unique ID counter

        if self.use_parsing:
            # Group speaking turns into batches
            for i in range(0, len(validated_data), self.speaking_turns_per_prompt):
                batch = validated_data[i:i + self.speaking_turns_per_prompt]
                speaking_turns = []
                source_id_map = {}  # Map source_id to metadata
                for j, record in enumerate(batch):
                    content = record.get(self.content_field, "")
                    metadata = {k: v for k, v in record.items() if k != self.content_field}
                    source_id = record.get('id', i + j + 1)  # Use 'id' or generate one
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
                    meaning_unit = MeaningUnit(
                        unique_id=unique_id_counter,  # Assign unique ID
                        meaning_unit_string=pu,
                        metadata=metadata
                    )
                    meaning_units.append(meaning_unit)
                    unique_id_counter += 1  # Increment the counter for the next unit
        else:
            for record in validated_data:
                content = record.get(self.content_field, "")
                metadata = {k: v for k, v in record.items() if k != self.content_field}
                meaning_unit = MeaningUnit(
                    unique_id=unique_id_counter,  # Assign unique ID
                    meaning_unit_string=content,
                    metadata=metadata
                )
                meaning_units.append(meaning_unit)
                unique_id_counter += 1  # Increment the counter for the next unit

        logger.debug(f"Transformed data into {len(meaning_units)} meaning units.")
        return meaning_units
