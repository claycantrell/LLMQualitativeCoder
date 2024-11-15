#data_handlers.py
import os
import json
import logging
from typing import Any, Dict, List
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
        use_parsing: bool = True
    ):
        self.file_path = file_path
        self.parse_instructions = parse_instructions
        self.completion_model = completion_model
        self.model_class = model_class
        self.content_field = content_field
        self.use_parsing = use_parsing

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
                validated_data.append(validated_record.dict())
            except Exception as e:
                logger.warning(f"Validation failed for record {record.get('id', 'unknown')}: {e}")
                continue

        logger.debug(f"Validated {len(validated_data)} records.")
        return validated_data

    def transform_data(self, validated_data: List[Dict[str, Any]]) -> List[MeaningUnit]:
        """
        Transforms validated data into MeaningUnit objects.

        Args:
            validated_data (List[Dict[str, Any]]): List of validated data records.

        Returns:
            List[MeaningUnit]: List of MeaningUnit objects.
        """
        meaning_units = []
        for idx, record in enumerate(validated_data, start=1):
            content = record.get(self.content_field, "")
            metadata = {k: v for k, v in record.items() if k != self.content_field}
            if self.use_parsing:
                parsed_units = parse_transcript(
                    speaking_turn_string=content,
                    prompt=self.parse_instructions,
                    completion_model=self.completion_model,
                    metadata=metadata
                )
                for pu in parsed_units:
                    meaning_unit = MeaningUnit(
                        unique_id=idx,
                        meaning_unit_string=pu,
                        metadata=metadata
                    )
                    meaning_units.append(meaning_unit)
            else:
                meaning_unit = MeaningUnit(
                    unique_id=idx,
                    meaning_unit_string=content,
                    metadata=metadata
                )
                meaning_units.append(meaning_unit)

        logger.debug(f"Transformed data into {len(meaning_units)} meaning units.")
        return meaning_units
