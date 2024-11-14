import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple, Type
import faiss
from data_handlers import BaseDataHandler, FlexibleDataHandler
from utils import (
    load_environment_variables,
    load_parse_instructions,
    load_custom_coding_prompt,
    initialize_deductive_resources,
    create_dynamic_model_for_format,
    load_schema_config
)
from qual_functions import (
    MeaningUnit,
    assign_codes_to_meaning_units
)

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
    data_format: str = "interview"
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

    # Load schema mapping configuration for dynamic model creation
    schema_config_path = os.path.join(config_folder, 'data_format_config.json')
    schema_config = load_schema_config(schema_config_path)

    if data_format not in schema_config:
        logger.error(f"No schema configuration found for data format: {data_format}")
        raise ValueError(f"No schema configuration found for data format: {data_format}")

    # Create a dynamic Pydantic model for the specified data format
    dynamic_data_model = create_dynamic_model_for_format(data_format, schema_config)

    # Determine the data file to load based on data format
    data_file_map = {
        "interview": "output_cues.json",
        "news": "news_articles.json"
    }
    data_file = data_file_map.get(data_format, None)
    if not data_file:
        logger.error(f"No data file mapped for data format: {data_format}")
        raise ValueError(f"No data file mapped for data format: {data_format}")

    file_path = os.path.join(json_folder, data_file)
    if not os.path.exists(file_path):
        logger.error(f"Data file '{file_path}' not found.")
        raise FileNotFoundError(f"Data file '{file_path}' not found.")

    # Stage 2: Data Loading and Validation Using Dynamic Model
    with open(file_path, 'r', encoding='utf-8') as file:
        raw_data = json.load(file)

    validated_data = []
    for item in raw_data:
        try:
            validated_item = dynamic_data_model(**item)
            validated_data.append(validated_item)
        except Exception as e:
            logger.error(f"Data validation error for item {item}: {e}")
            continue

    if not validated_data:
        logger.warning("No valid data items after validation. Exiting.")
        return

    # Stage 3: Data Transformation
    # We create meaning units directly, as data handlers have been generalized
    # to handle partially unknown schemas. For advanced transformations, consider dynamic transformations.
    meaning_unit_object_list = []
    for idx, item in enumerate(validated_data, start=1):
        if data_format == "interview":
            # For interviews, assume entire speaking turn or parsed meaning units
            speaker_id = getattr(item, 'speaker_name', 'Unknown')
            text_content = getattr(item, 'text_context', '')
            meaning_unit_object = MeaningUnit(
                unique_id=idx,
                speaker_id=speaker_id,
                meaning_unit_string=text_content
            )
            meaning_unit_object_list.append(meaning_unit_object)
        elif data_format == "news":
            # For news articles, assume entire content as a meaning unit or parse further
            author = getattr(item, 'author', 'Unknown')
            content = getattr(item, 'content', '')
            meaning_unit_object = MeaningUnit(
                unique_id=idx,
                speaker_id=author,
                meaning_unit_string=content
            )
            meaning_unit_object_list.append(meaning_unit_object)
        else:
            logger.warning(f"Unsupported data format for transformation: {data_format}")
            continue

    if not meaning_unit_object_list:
        logger.warning("No meaning units to process. Exiting.")
        return

    # Stage 4: Code Assignment
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

    # Stage 5: Output Results
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
    main(
        coding_mode="deductive",
        use_parsing=True,
        use_rag=True,
        parse_model="gpt-4o-mini",
        assign_model="gpt-4o-mini",
        initialize_embedding_model="text-embedding-3-small",
        retrieve_embedding_model="text-embedding-3-small",
        data_format="interview"
    )
