# main.py

import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
from abc import ABC, abstractmethod
from qual_functions import (
    MeaningUnit,
    TextData,
    parse_transcript,
    assign_codes_to_meaning_units,
    initialize_faiss_index_from_formatted_file,
    CodeAssigned
)
import faiss

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
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

# -------------------------------
# Utility Functions
# -------------------------------

def load_environment_variables() -> None:
    """
    Loads and validates required environment variables.
    """
    openai_api_key = os.getenv('OPENAI_API_KEY')
    if not openai_api_key:
        logger.error("OPENAI_API_KEY environment variable is not set.")
        raise ValueError("Set the OPENAI_API_KEY environment variable.")

def load_coding_instructions(prompts_folder: str) -> str:
    """
    Loads coding instructions for deductive coding from a file.
    """
    coding_instructions_file = os.path.join(prompts_folder, 'coding_prompt.txt')
    if not os.path.exists(coding_instructions_file):
        logger.error(f"Coding instructions file '{coding_instructions_file}' not found.")
        raise FileNotFoundError(f"Coding instructions file '{coding_instructions_file}' not found.")

    with open(coding_instructions_file, 'r', encoding='utf-8') as file:
        coding_instructions = file.read().strip()

    return coding_instructions

def load_parse_instructions(prompts_folder: str) -> str:
    """
    Loads parse instructions from a file for breaking down speaking turns into meaning units.
    """
    parse_prompt_file = os.path.join(prompts_folder, 'parse_prompt.txt')
    if not os.path.exists(parse_prompt_file):
        logger.error(f"Parse instructions file '{parse_prompt_file}' not found.")
        raise FileNotFoundError(f"Parse instructions file '{parse_prompt_file}' not found.")

    with open(parse_prompt_file, 'r', encoding='utf-8') as file:
        parse_instructions = file.read().strip()

    return parse_instructions

def load_custom_coding_prompt(prompts_folder: str) -> str:
    """
    Loads a custom coding prompt for inductive coding from a file.
    """
    custom_coding_prompt_file = os.path.join(prompts_folder, 'custom_coding_prompt.txt')
    if not os.path.exists(custom_coding_prompt_file):
        logger.error(f"Custom coding prompt file '{custom_coding_prompt_file}' not found.")
        raise FileNotFoundError(f"Custom coding prompt file '{custom_coding_prompt_file}' not found.")

    with open(custom_coding_prompt_file, 'r', encoding='utf-8') as file:
        custom_coding_prompt = file.read().strip()

    if not custom_coding_prompt:
        logger.error("Custom coding prompt file is empty.")
        raise ValueError("Custom coding prompt file is empty.")

    return custom_coding_prompt

def initialize_deductive_resources(
    codebase_folder: str,
    prompts_folder: str,
    initialize_embedding_model: str
) -> Tuple[List[Dict[str, Any]], faiss.IndexFlatL2, str]:
    """
    Initializes resources needed for deductive coding: loads code instructions, codebase, and builds a FAISS index.
    Returns processed_codes, faiss_index, and coding_instructions.
    """
    # Load coding instructions for deductive coding
    coding_instructions = load_coding_instructions(prompts_folder)

    # Initialize FAISS index and get processed codes
    list_of_codes_file = os.path.join(codebase_folder, 'new_schema.txt')
    if not os.path.exists(list_of_codes_file):
        logger.error(f"List of codes file '{list_of_codes_file}' not found.")
        raise FileNotFoundError(f"List of codes file '{list_of_codes_file}' not found.")

    faiss_index, processed_codes = initialize_faiss_index_from_formatted_file(
        list_of_codes_file,
        embedding_model=initialize_embedding_model
    )

    if not processed_codes:
        logger.warning(f"No codes found in '{list_of_codes_file}' or failed to process correctly.")
        processed_codes = []

    return processed_codes, faiss_index, coding_instructions

# -------------------------------
# Main Function (Pipeline Orchestration)
# -------------------------------
def main(
    coding_mode: str = "deductive",
    use_parsing: bool = True,
    use_rag: bool = True,
    parse_model: str = "gpt-4o-mini",
    assign_model: str = "gpt-4o-mini",
    initialize_embedding_model: str = "text-embedding-3-small",
    retrieve_embedding_model: str = "text-embedding-3-small",
    data_format: str = "interview"  # Added parameter to specify data format
):
    """
    Orchestrates the entire process of assigning qualitative codes to transcripts or articles based on the provided modes and configurations.
    """

    # Stage 1: Environment Setup
    load_environment_variables()
    logger.debug("Environment variables loaded and validated.")

    # Define paths
    prompts_folder = 'prompts'
    codebase_folder = 'qual_codebase'
    json_folder = 'json_transcripts'
    config_folder = 'configs'

    # Load parse instructions if parsing is enabled
    parse_instructions = load_parse_instructions(prompts_folder) if use_parsing else ""

    # Load schema mapping configuration
    config_file = os.path.join(config_folder, f'{data_format}_config.json')
    if not os.path.exists(config_file):
        logger.error(f"Configuration file '{config_file}' not found.")
        raise FileNotFoundError(f"Configuration file '{config_file}' not found.")

    with open(config_file, 'r', encoding='utf-8') as file:
        schema_mapping = json.load(file)

    # Initialize Data Handler based on schema mapping
    if data_format == "interview":
        data_handler = InterviewDataHandler(
            file_path=os.path.join(json_folder, 'output_cues.json'),
            parse_instructions=parse_instructions,
            completion_model=parse_model,
            coding_mode=coding_mode,
            use_parsing=use_parsing
        )
    elif data_format == "news":
        data_handler = NewsDataHandler(
            file_path=os.path.join(json_folder, 'news_articles.json'),
            parse_instructions=parse_instructions,
            completion_model=parse_model,
            coding_mode=coding_mode,
            use_parsing=use_parsing
        )
    else:
        logger.error(f"Unsupported data format: {data_format}")
        raise ValueError(f"Unsupported data format: {data_format}")

    # Stage 2: Data Loading and Transformation
    raw_data = data_handler.load_data()
    meaning_unit_object_list = data_handler.transform_data(raw_data)

    if not meaning_unit_object_list:
        logger.warning("No meaning units to process. Exiting.")
        return

    # Stage 3: Code Assignment
    if coding_mode == "deductive":
        processed_codes, faiss_index, coding_instructions = initialize_deductive_resources(
            codebase_folder=codebase_folder,
            prompts_folder=prompts_folder,
            initialize_embedding_model=initialize_embedding_model
        )

        if not processed_codes:
            logger.warning("No processed codes available for deductive coding. Exiting.")
            return

        # Assign codes to meaning units in deductive mode
        if use_rag:
            # Deductive coding with RAG
            coded_meaning_unit_list = assign_codes_to_meaning_units(
                meaning_unit_list=meaning_unit_object_list,
                coding_instructions=coding_instructions,
                processed_codes=processed_codes,
                index=faiss_index,
                top_k=5,
                context_size=5,
                use_rag=True,
                completion_model=assign_model,
                embedding_model=retrieve_embedding_model
            )
        else:
            # Deductive coding without RAG (using full codebase)
            coded_meaning_unit_list = assign_codes_to_meaning_units(
                meaning_unit_list=meaning_unit_object_list,
                coding_instructions=coding_instructions,
                processed_codes=processed_codes,
                index=faiss_index,
                top_k=5,
                context_size=5,
                use_rag=False,
                codebase=processed_codes,
                completion_model=assign_model,
                embedding_model=retrieve_embedding_model
            )

    else:  # Inductive coding
        # Load custom coding prompt
        coding_instructions = load_custom_coding_prompt(prompts_folder)

        # Assign codes to meaning units in inductive mode
        coded_meaning_unit_list = assign_codes_to_meaning_units(
            meaning_unit_list=meaning_unit_object_list,
            coding_instructions=coding_instructions,
            processed_codes=None,
            index=None,
            top_k=None,
            context_size=5,
            use_rag=False,
            codebase=None,
            completion_model=assign_model,
            embedding_model=None  # Embeddings not needed in inductive mode
        )

    # Stage 4: Output Results
    for unit in coded_meaning_unit_list:
        logger.info(f"\nID: {unit.unique_id}")
        logger.info(f"Speaker: {unit.speaker_id}")
        logger.info(f"Quote: {unit.meaning_unit_string}")
        if unit.assigned_code_list:
            for code in unit.assigned_code_list:
                logger.info(f"  Code: {code.code_name}")
                logger.info(f"  Justification: {code.code_justification}")
        else:
            logger.info("  No codes assigned.")

if __name__ == "__main__":
    # Example usage:

    # Deductive Coding with Parsing and RAG for Interview
    # main(
    #     coding_mode="deductive",
    #     use_parsing=True,
    #     use_rag=True,
    #     parse_model="gpt-4o-mini",
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",
    #     retrieve_embedding_model="text-embedding-3-small",
    #     data_format="interview"
    # )

    # Deductive Coding without Parsing and without RAG for Interview
    # main(
    #     coding_mode="deductive",
    #     use_parsing=False,
    #     use_rag=False,
    #     parse_model="gpt-4o-mini",
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",
    #     retrieve_embedding_model="text-embedding-3-small",
    #     data_format="interview"
    # )

    # Inductive Coding with Parsing for Interview
    # main(
    #     coding_mode="inductive",
    #     use_parsing=True,  # Enable parsing for inductive coding
    #     use_rag=False,     # RAG not applicable in inductive mode
    #     parse_model="gpt-4o-mini",  # Relevant only if use_parsing=True
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",  # Irrelevant in inductive mode
    #     retrieve_embedding_model="text-embedding-3-small",    # Irrelevant in inductive mode
    #     data_format="interview"
    # )

    # Inductive Coding without Parsing for Interview
    # main(
    #     coding_mode="inductive",
    #     use_parsing=False,
    #     use_rag=False,
    #     parse_model="gpt-4o-mini",  # Irrelevant if use_parsing=False
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",  # Irrelevant in inductive mode
    #     retrieve_embedding_model="text-embedding-3-small",    # Irrelevant in inductive mode
    #     data_format="interview"
    # )

    # Deductive Coding with Parsing and RAG for News Articles
    main(
        coding_mode="deductive",
        use_parsing=True,
        use_rag=True,
        parse_model="gpt-4o-mini",
        assign_model="gpt-4o-mini",
        initialize_embedding_model="text-embedding-3-small",
        retrieve_embedding_model="text-embedding-3-small",
        data_format="news"
    )

    # Deductive Coding without Parsing and without RAG for News Articles
    # main(
    #     coding_mode="deductive",
    #     use_parsing=False,
    #     use_rag=False,
    #     parse_model="gpt-4o-mini",
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",
    #     retrieve_embedding_model="text-embedding-3-small",
    #     data_format="news"
    # )

    # Inductive Coding with Parsing for News Articles
    # main(
    #     coding_mode="inductive",
    #     use_parsing=True,
    #     use_rag=False,
    #     parse_model="gpt-4o-mini",
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",
    #     retrieve_embedding_model="text-embedding-3-small",
    #     data_format="news"
    # )

    # Inductive Coding without Parsing for News Articles
    # main(
    #     coding_mode="inductive",
    #     use_parsing=False,
    #     use_rag=False,
    #     parse_model="gpt-4o-mini",
    #     assign_model="gpt-4o-mini",
    #     initialize_embedding_model="text-embedding-3-small",
    #     retrieve_embedding_model="text-embedding-3-small",
    #     data_format="news"
    # )
