# data_handlers.py
import os
import json
import logging
from typing import Any, Dict, List, Tuple, Optional
from pydantic import BaseModel
from qual_functions import parse_transcript, MeaningUnit

logger = logging.getLogger(__name__)

def get_nested_field(data: Dict[str, Any], field_path: str, default=None):
    keys = field_path.split('.')
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key, default)
        else:
            return default
    return data

class FlexibleDataHandler:
    def __init__(
        self,
        file_path: str,
        parse_instructions: str,
        completion_model: str,
        model_class: Any,
        item_model_class: Any,
        content_field: str,
        speaker_field: Optional[str],
        list_field: Optional[str] = None,
        use_parsing: bool = True,
        speaking_turns_per_prompt: int = 1
    ):
        self.file_path = file_path
        self.parse_instructions = parse_instructions
        self.completion_model = completion_model
        self.model_class = model_class
        self.item_model_class = item_model_class
        self.content_field = content_field
        self.speaker_field = speaker_field
        self.list_field = list_field
        self.use_parsing = use_parsing
        self.speaking_turns_per_prompt = speaking_turns_per_prompt
        self.document_metadata = {}  # Store document-level metadata

    def load_data(self) -> List[Dict[str, Any]]:
        """
        Loads data from the JSON file and validates it against the main data model.

        Returns:
            List[Dict[str, Any]]: List of validated script entries.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                raw_data = json.load(file)
            logger.debug(f"Loaded data from '{self.file_path}'.")
        except Exception as e:
            logger.error(f"Failed to load data from '{self.file_path}': {e}")
            raise

        # Validate the main data structure
        try:
            validated_data = self.model_class(**raw_data)
            validated_data_dict = validated_data.model_dump()
            logger.debug("Validated main data structure.")
        except Exception as e:
            logger.error(f"Validation failed for main data structure: {e}")
            raise

        # Extract the list of script entries
        script_entries = get_nested_field(validated_data_dict, self.list_field)
        if not isinstance(script_entries, list):
            logger.error(f"The list_field '{self.list_field}' does not point to a list.")
            raise ValueError(f"The list_field '{self.list_field}' does not point to a list.")

        # Store document-level metadata
        document_metadata = validated_data_dict.copy()
        # Remove the list_field from the document_metadata
        keys = self.list_field.split('.')
        current_dict = document_metadata
        for key in keys[:-1]:
            current_dict = current_dict.get(key, {})
        if keys[-1] in current_dict:
            del current_dict[keys[-1]]
        self.document_metadata = document_metadata  # Store document-level metadata

        # Validate each script entry
        validated_entries = []
        for entry in script_entries:
            try:
                validated_entry = self.item_model_class(**entry)
                validated_entries.append(validated_entry.model_dump())
            except Exception as e:
                logger.warning(f"Validation failed for script entry: {e}")
                continue

        logger.debug(f"Validated {len(validated_entries)} script entries.")
        return validated_entries

    def transform_data(self, validated_data: List[Dict[str, Any]]) -> List[MeaningUnit]:
        """
        Transforms validated data into MeaningUnit objects, processing multiple speaking turns per prompt.

        Args:
            validated_data (List[Dict[str, Any]]): List of validated script entries.

        Returns:
            List[MeaningUnit]: List of MeaningUnit objects.
        """
        meaning_units = []
        meaning_unit_id_counter = 1  # Initialize a unique ID counter for meaning units

        if self.use_parsing:
            # Group speaking turns into batches
            for i in range(0, len(validated_data), self.speaking_turns_per_prompt):
                batch = validated_data[i:i + self.speaking_turns_per_prompt]
                speaking_turns = []
                source_id_map = {}  # Map source_id to metadata
                for j, record in enumerate(batch):
                    content = get_nested_field(record, self.content_field, "")
                    metadata = {k: v for k, v in record.items() if k != self.content_field}
                    source_id = record.get('source_id')
                    if source_id is None:
                        # Generate a unique source_id if not present
                        source_id = f"auto_{i + j + 1}"
                        record['source_id'] = source_id  # Assign back to the record
                        logger.warning(f"Assigned auto-generated 'id' '{source_id}' to speaking turn: {record}")
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
                    # Ensure 'id' is included in metadata
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
            for record in validated_data:
                content = get_nested_field(record, self.content_field, "")
                metadata = {k: v for k, v in record.items() if k != self.content_field}
                source_id = record.get('source_id')
                if source_id is None:
                    # Generate a unique source_id if not present
                    source_id = f"auto_{meaning_unit_id_counter}"
                    record['id'] = source_id  # Assign back to the record
                    logger.warning(f"Assigned auto-generated 'source_id' '{source_id}' to speaking turn: {record}")
                # Ensure 'id' is included in metadata
                metadata['source_id'] = source_id
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
