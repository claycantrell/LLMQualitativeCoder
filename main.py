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
    level=logging.DEBUG,  # Changed from INFO to DEBUG for detailed logs
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
    data_format: str = "interview"  # Default to "interview" as per user issue
):
    """
    Orchestrates the entire process of assigning qualitative codes to transcripts or articles based on the provided modes and configurations.
    """
    logger.info("Starting the main pipeline.")
    
    # Stage 1: Environment Setup
    try:
        load_environment_variables()
        logger.debug("Environment variables loaded and validated.")
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return

    # Define paths
    prompts_folder = 'prompts'
    codebase_folder = 'qual_codebase'
    json_folder = 'json_transcripts'
    config_folder = 'configs'

    # Load parse instructions if parsing is enabled
    parse_instructions = ""
    if use_parsing:
        try:
            parse_instructions = load_parse_instructions(prompts_folder)
            logger.debug("Parse instructions loaded.")
        except Exception as e:
            logger.error(f"Failed to load parse instructions: {e}")
            return

    # Load schema mapping configuration for dynamic model creation
    schema_config_path = os.path.join(config_folder, 'data_format_config.json')
    try:
        schema_config = load_schema_config(schema_config_path)
        logger.debug("Schema configuration loaded.")
    except Exception as e:
        logger.error(f"Failed to load schema configuration: {e}")
        return

    if data_format not in schema_config:
        logger.error(f"No schema configuration found for data format: {data_format}")
        raise ValueError(f"No schema configuration found for data format: {data_format}")

    # Create a dynamic Pydantic model for the specified data format
    try:
        dynamic_data_model, content_field = create_dynamic_model_for_format(data_format, schema_config)
        logger.debug(f"Dynamic data model for '{data_format}' created.")
    except Exception as e:
        logger.error(f"Failed to create dynamic data model: {e}")
        return

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

    # Stage 2: Data Loading and Validation Using FlexibleDataHandler
    try:
        data_handler = FlexibleDataHandler(
            file_path=file_path,
            parse_instructions=parse_instructions,
            completion_model=parse_model,
            model_class=dynamic_data_model,
            content_field=content_field,
            use_parsing=use_parsing
        )
        validated_data = data_handler.load_data()
        logger.debug(f"Loaded {len(validated_data)} validated data items.")
        meaning_unit_object_list = data_handler.transform_data(validated_data)
        logger.debug(f"Transformed data into {len(meaning_unit_object_list)} meaning units.")
    except Exception as e:
        logger.error(f"Data loading and transformation failed: {e}")
        return

    if not meaning_unit_object_list:
        logger.warning("No meaning units to process. Exiting.")
        return

    # Stage 4: Code Assignment
    if coding_mode == "deductive":
        try:
            processed_codes, faiss_index, coding_instructions = initialize_deductive_resources(
                codebase_folder=codebase_folder,
                prompts_folder=prompts_folder,
                initialize_embedding_model=initialize_embedding_model
            )
            logger.debug(f"Initialized deductive resources with {len(processed_codes)} processed codes.")
        except Exception as e:
            logger.error(f"Failed to initialize deductive resources: {e}")
            return

        if not processed_codes:
            logger.warning("No processed codes available for deductive coding. Exiting.")
            return

        # Assign codes to meaning units in deductive mode
        try:
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
                logger.debug("Assigned codes using deductive mode with RAG.")
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
                logger.debug("Assigned codes using deductive mode without RAG.")
        except Exception as e:
            logger.error(f"Failed to assign codes in deductive mode: {e}")
            return

    else:  # Inductive coding
        try:
            # Load custom coding prompt
            coding_instructions = load_custom_coding_prompt(prompts_folder)
            logger.debug("Loaded custom coding prompt for inductive coding.")

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
            logger.debug("Assigned codes using inductive mode.")
        except Exception as e:
            logger.error(f"Failed to assign codes in inductive mode: {e}")
            return

    # Stage 5: Output Results
    logger.info("Outputting coded meaning units:")
    for unit in coded_meaning_unit_list:
        logger.info(f"\nID: {unit.unique_id}")
        logger.info(f"Metadata: {unit.metadata}")
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
        data_format="interview"  # Changed to "interview" as per user issue
    )
