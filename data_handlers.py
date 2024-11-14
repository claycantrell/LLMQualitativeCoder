import os
import json
import logging
from typing import List
from abc import ABC, abstractmethod
from qual_functions import (
    MeaningUnit,
    parse_transcript
)

# Configure logging for data_handlers module
logger = logging.getLogger(__name__)

# -------------------------------
# Abstract Base Class for Data Handlers
# -------------------------------
class BaseDataHandler(ABC):
    """
    Abstract base class defining how data should be loaded and transformed.
    Subclasses must implement the abstract methods:
    - load_data()
    - transform_data()
    """

    @abstractmethod
    def load_data(self) -> List[dict]:
        pass

    @abstractmethod
    def transform_data(self, data: List[dict]) -> List[MeaningUnit]:
        pass


# -------------------------------
# Flexible Data Handler for Various JSON Formats
# -------------------------------
class FlexibleDataHandler(BaseDataHandler):
    """
    A data handler that dynamically processes data based on a provided schema.
    """

    def __init__(self, file_path: str, parse_instructions: str, completion_model: str, model_class, use_parsing: bool = True):
        self.file_path = file_path
        self.parse_instructions = parse_instructions
        self.completion_model = completion_model
        self.model_class = model_class
        self.use_parsing = use_parsing

    def load_data(self) -> List[dict]:
        """
        Loads and validates the JSON data using the dynamic model.
        """
        if not os.path.exists(self.file_path):
            logger.error(f"JSON file '{self.file_path}' not found.")
            raise FileNotFoundError(f"JSON file '{self.file_path}' not found.")

        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                raw_data = json.load(file)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{self.file_path}': {e}")
            raise
        
        validated_data = []
        for item in raw_data:
            try:
                validated_item = self.model_class(**item)
                validated_data.append(validated_item)
            except Exception as e:
                logger.error(f"Data validation error: {e}, Item: {item}")
        return validated_data

    def transform_data(self, data: List) -> List[MeaningUnit]:
        """
        Transforms loaded data into a list of MeaningUnit objects.
        If use_parsing is True, it can parse textual content into meaning units.
        If use_parsing is False, it uses the entire content as a single meaning unit.
        The transformation logic here must be tailored based on actual fields in data and instructions.
        """
        meaning_unit_list = []
        for idx, item in enumerate(data, start=1):
            # Example logic: if the model has 'speaker_name' and 'text_context' fields
            speaker_id = getattr(item, 'speaker_name', 'Unknown')
            content = getattr(item, 'content', '') or getattr(item, 'text_context', '')
            if not content:
                logger.warning(f"No content found for item {idx}. Skipping.")
                continue

            if self.use_parsing:
                # Parse the content to extract meaning units
                formatted_prompt = self.parse_instructions.replace("{speaker_name}", speaker_id)
                parsed_units = parse_transcript(
                    content,
                    formatted_prompt,
                    completion_model=self.completion_model
                )
                if not parsed_units:
                    logger.warning(f"No meaning units extracted for item {idx}. Skipping.")
                    continue

                for unit_idx, unit in enumerate(parsed_units, start=1):
                    meaning_unit_object = MeaningUnit(
                        unique_id=(idx * 1000) + unit_idx,
                        speaker_id=unit.get('speaker_id', speaker_id),
                        meaning_unit_string=unit.get('meaning_unit_string', '')
                    )
                    logger.debug(f"Added Meaning Unit {meaning_unit_object.unique_id}: Speaker - {meaning_unit_object.speaker_id}, Quote - {meaning_unit_object.meaning_unit_string}")
                    meaning_unit_list.append(meaning_unit_object)
            else:
                # Use the entire content as one meaning unit
                meaning_unit_object = MeaningUnit(
                    unique_id=idx,
                    speaker_id=speaker_id,
                    meaning_unit_string=content
                )
                logger.debug(f"Added Meaning Unit: Speaker - {meaning_unit_object.speaker_id}, Quote - {meaning_unit_object.meaning_unit_string}")
                meaning_unit_list.append(meaning_unit_object)

        if not meaning_unit_list:
            logger.warning("No meaning units extracted from any items.")
        return meaning_unit_list
