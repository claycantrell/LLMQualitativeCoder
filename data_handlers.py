# data_handlers.py

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
# Specialized Data Handler for Interview Transcripts
# -------------------------------
class InterviewDataHandler(BaseDataHandler):
    """
    A data handler for interview transcripts.
    Expects JSON data with fields:
    - id
    - length_of_time_spoken_seconds
    - text_context
    - speaker_name
    """

    def __init__(self, file_path: str, parse_instructions: str, completion_model: str, coding_mode: str = "deductive", use_parsing: bool = True):
        self.file_path = file_path
        self.parse_instructions = parse_instructions
        self.completion_model = completion_model
        self.coding_mode = coding_mode
        self.use_parsing = use_parsing

    def load_data(self) -> List[dict]:
        """
        Loads the JSON data from a file containing interview transcripts.
        """
        if not os.path.exists(self.file_path):
            logger.error(f"JSON file '{self.file_path}' not found.")
            raise FileNotFoundError(f"JSON file '{self.file_path}' not found.")

        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{self.file_path}': {e}")
            raise

    def transform_data(self, data: List[dict]) -> List[MeaningUnit]:
        """
        Transforms loaded data into a list of MeaningUnit objects.
        If use_parsing is True, it parses each speaking turn into meaning units.
        If use_parsing is False, it uses entire speaking turns as meaning units.
        """
        meaning_unit_list = []
        for idx, speaking_turn in enumerate(data, start=1):
            speaker_id = speaking_turn.get('speaker_name', 'Unknown')
            speaking_turn_string = speaking_turn.get('text_context', '')
            if not speaking_turn_string:
                logger.warning(f"No speaking turn text found for Speaking Turn {idx}. Skipping.")
                continue

            if self.use_parsing:
                # Parsing speaking turns
                logger.info(f"Parsing Speaking Turn {idx}: Speaker - {speaker_id}")
                formatted_prompt = self.parse_instructions.replace("{speaker_name}", speaker_id)
                parsed_units = parse_transcript(
                    speaking_turn_string,
                    formatted_prompt,
                    completion_model=self.completion_model
                )
                if not parsed_units:
                    logger.warning(f"No meaning units extracted from Speaking Turn {idx}. Skipping.")
                    continue

                for unit_idx, unit in enumerate(parsed_units, start=1):
                    meaning_unit_object = MeaningUnit(
                        speaker_id=unit.get('speaker_id', speaker_id),
                        meaning_unit_string=unit.get('meaning_unit_string', '')
                    )
                    logger.debug(f"Added Meaning Unit {unit_idx}: Speaker - {meaning_unit_object.speaker_id}, Quote - {meaning_unit_object.meaning_unit_string}")
                    meaning_unit_list.append(meaning_unit_object)
            else:
                # Not parsing speaking turns
                logger.info(f"Using entire speaking turn {idx} as a meaning unit: Speaker - {speaker_id}")
                meaning_unit_object = MeaningUnit(
                    speaker_id=speaker_id,
                    meaning_unit_string=speaking_turn_string
                )
                logger.debug(f"Added Meaning Unit: Speaker - {meaning_unit_object.speaker_id}, Quote - {meaning_unit_object.meaning_unit_string}")
                meaning_unit_list.append(meaning_unit_object)

        if not meaning_unit_list:
            logger.warning("No meaning units extracted from any speaking turns.")
        return meaning_unit_list


# -------------------------------
# Specialized Data Handler for News Articles
# -------------------------------
class NewsDataHandler(BaseDataHandler):
    """
    A data handler for news articles.
    Expects JSON data with fields:
    - id
    - title
    - publication_date
    - author
    - content
    - section
    - source
    - url
    - tags (optional)
    """

    def __init__(self, file_path: str, parse_instructions: str, completion_model: str, coding_mode: str = "deductive", use_parsing: bool = True):
        self.file_path = file_path
        self.parse_instructions = parse_instructions
        self.completion_model = completion_model
        self.coding_mode = coding_mode
        self.use_parsing = use_parsing

    def load_data(self) -> List[dict]:
        """
        Loads the JSON data from a file containing news articles.
        """
        if not os.path.exists(self.file_path):
            logger.error(f"News JSON file '{self.file_path}' not found.")
            raise FileNotFoundError(f"News JSON file '{self.file_path}' not found.")

        try:
            with open(self.file_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from '{self.file_path}': {e}")
            raise

    def transform_data(self, data: List[dict]) -> List[MeaningUnit]:
        """
        Transforms loaded data into a list of MeaningUnit objects.
        If use_parsing is True, it parses each article's content into meaning units.
        If use_parsing is False, it uses the entire content as a single meaning unit.
        """
        meaning_unit_list = []
        for idx, article in enumerate(data, start=1):
            title = article.get('title', 'Untitled')
            author = article.get('author', 'Unknown Author')
            content = article.get('content', '')
            if not content:
                logger.warning(f"No content found for News Article {idx}. Skipping.")
                continue

            speaker_id = f"Author: {author}"  # For consistency with MeaningUnit structure

            if self.use_parsing:
                # Parsing article content
                logger.info(f"Parsing News Article {idx}: Title - {title}")
                formatted_prompt = self.parse_instructions.replace("{speaker_name}", author)
                parsed_units = parse_transcript(
                    content,
                    formatted_prompt,
                    completion_model=self.completion_model
                )
                if not parsed_units:
                    logger.warning(f"No meaning units extracted from News Article {idx}. Skipping.")
                    continue

                for unit_idx, unit in enumerate(parsed_units, start=1):
                    meaning_unit_object = MeaningUnit(
                        speaker_id=unit.get('speaker_id', speaker_id),
                        meaning_unit_string=unit.get('meaning_unit_string', '')
                    )
                    logger.debug(f"Added Meaning Unit {unit_idx}: Speaker - {meaning_unit_object.speaker_id}, Quote - {meaning_unit_object.meaning_unit_string}")
                    meaning_unit_list.append(meaning_unit_object)
            else:
                # Not parsing article content
                logger.info(f"Using entire content of News Article {idx} as a meaning unit: Title - {title}")
                meaning_unit_object = MeaningUnit(
                    speaker_id=speaker_id,
                    meaning_unit_string=content
                )
                logger.debug(f"Added Meaning Unit: Speaker - {meaning_unit_object.speaker_id}, Quote - {meaning_unit_object.meaning_unit_string}")
                meaning_unit_list.append(meaning_unit_object)

        if not meaning_unit_list:
            logger.warning("No meaning units extracted from any news articles.")
        return meaning_unit_list
