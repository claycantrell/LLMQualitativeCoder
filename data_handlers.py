import os
import json
import logging
from typing import List, Dict, Any
from abc import ABC, abstractmethod
from qual_functions import MeaningUnit, parse_transcript

# Configure logging for data_handlers module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG)  # Ensure DEBUG logs are captured

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
    def load_data(self) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def transform_data(self, data: List[Any]) -> List[MeaningUnit]:
        pass

# -------------------------------
# Flexible Data Handler for Various JSON Formats
# -------------------------------
class FlexibleDataHandler(BaseDataHandler):
    """
    A data handler that dynamically processes data based on a provided schema.
    """

    def __init__(
        self, 
        file_path: str, 
        parse_instructions: str, 
        completion_model: str, 
        model_class, 
        content_field: str,
        use_parsing: bool = True
    ):
        self.file_path = file_path
        self.parse_instructions = parse_instructions
        self.completion_model = completion_model
        self.model_class = model_class
        self.content_field = content_field
        self.use_parsing = use_parsing

    def load_data(self) -> List[Any]:
        """
        Loads and validates the JSON data using the dynamic model.
        """
        if not os.path.exists(self.file_path):
            logger.error(f"JSON file '{self.file_path}' not found.")
            raise FileNotFoundError(f"JSON file '{self.file_path}' not found.")

        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                raw_data = json.load(file)
            logger.debug(f"Loaded {len(raw_data)} raw data items from '{self.file_path}'.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{self.file_path}': {e}")
            raise
        except Exception as e:
            logger.error(f"Error reading JSON file '{self.file_path}': {e}")
            raise
        
        validated_data = []
        for item in raw_data:
            try:
                validated_item = self.model_class(**item)
                validated_data.append(validated_item)
            except Exception as e:
                logger.error(f"Data validation error: {e}, Item: {item}")
        logger.debug(f"Validated {len(validated_data)} data items.")
        return validated_data

    def transform_data(self, data: List[Any]) -> List[MeaningUnit]:
        """
        Transforms loaded data into a list of MeaningUnit objects.
        If use_parsing is True, it can parse textual content into meaning units.
        If use_parsing is False, it uses the entire content as a single meaning unit.
        """
        meaning_unit_list = []
        for idx, item in enumerate(data, start=1):
            item_dict = item.dict()
            # Use the content_field from configuration
            content = item_dict.pop(self.content_field, '')
            if not content:
                logger.warning(f"No content found in field '{self.content_field}' for item {idx}. Skipping.")
                continue

            # Determine speaker ID or name if present
            speaker_id = 'Unknown'
            if 'speaker_name' in item_dict:
                speaker_id = item_dict.pop('speaker_name')
                logger.debug(f"Extracted speaker_id: {speaker_id} for item {idx}")

            if self.use_parsing and self.parse_instructions:
                # Parse the content to extract meaning units
                parsed_units = parse_transcript(
                    speaking_turn_string=content,
                    prompt=self.parse_instructions,
                    completion_model=self.completion_model,
                    metadata=item_dict
                )
                if not parsed_units:
                    logger.warning(f"No meaning units extracted for item {idx}. Using entire text as a single meaning unit.")
                    parsed_units = [content]  # Use entire content as a single meaning unit

                for unit_string in parsed_units:
                    unique_id = len(meaning_unit_list) + 1
                    current_metadata = dict(item_dict)  # Shallow copy of metadata
                    current_metadata['speaker_id'] = speaker_id
                    meaning_unit_object = MeaningUnit(
                        unique_id=unique_id,
                        metadata=current_metadata,
                        meaning_unit_string=unit_string
                    )
                    logger.debug(f"Added Meaning Unit {meaning_unit_object.unique_id}: Metadata - {meaning_unit_object.metadata}, Quote - {meaning_unit_object.meaning_unit_string}")
                    meaning_unit_list.append(meaning_unit_object)
            else:
                # Without parsing, use the entire content as a single meaning unit
                unique_id = len(meaning_unit_list) + 1
                current_metadata = dict(item_dict)  # Shallow copy of metadata
                current_metadata['speaker_id'] = speaker_id
                meaning_unit_object = MeaningUnit(
                    unique_id=unique_id,
                    metadata=current_metadata,
                    meaning_unit_string=content
                )
                logger.debug(f"Added Meaning Unit {meaning_unit_object.unique_id}: Metadata - {meaning_unit_object.metadata}, Quote - {meaning_unit_object.meaning_unit_string}")
                meaning_unit_list.append(meaning_unit_object)

        if not meaning_unit_list:
            logger.warning("No meaning units extracted from any items.")
        else:
            logger.debug(f"Total Meaning Units Extracted: {len(meaning_unit_list)}")
        return meaning_unit_list
