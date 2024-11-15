import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple, Type
from data_handlers import BaseDataHandler, FlexibleDataHandler
from utils import (
    load_environment_variables,
    load_parse_instructions,
    load_custom_coding_prompt,
    initialize_deductive_resources,
    create_dynamic_model_for_format,
    load_schema_config,
    load_config  # New function to load configurations
)
from qual_functions import (
    MeaningUnit,
    assign_codes_to_meaning_units
)

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Ensure DEBUG logs are captured
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# -------------------------------
# Main Function (Pipeline Orchestration)
# -------------------------------
def main(config: Dict[str, Any]):
    """
    Orchestrates the entire process of assigning qualitative codes to transcripts or articles based on the provided modes and configurations.
    """
    logger.info("Starting the main pipeline.")

    # Load configurations
    coding_mode = config.get('coding_mode', 'deductive')
    use_parsing = config.get('use_parsing', True)
    use_rag = config.get('use_rag', True)
    parse_model = config.get('parse_model', 'gpt-4o-mini')
    assign_model = config.get('assign_model', 'gpt-4o-mini')
    initialize_embedding_model = config.get('initialize_embedding_model', 'text-embedding-3-small')
    retrieve_embedding_model = config.get('retrieve_embedding_model', 'text-embedding-3-small')
    data_format = config.get('data_format', 'interview')

    paths = config.get('paths', {})
    prompts_folder = paths.get('prompts_folder', 'prompts')
    codebase_folder = paths.get('codebase_folder', 'qual_codebase')
    json_folder = paths.get('json_folder', 'json_transcripts')
    config_folder = paths.get('config_folder', 'configs')

    # Stage 1: Environment Setup
    try:
        load_environment_variables()
        logger.debug("Environment variables loaded and validated.")
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return

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
        "news": "news_articles.json",
        # Add other data formats and corresponding files here
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
            # Initialize deductive resources with conditional FAISS initialization
            processed_codes, faiss_index, coding_instructions = initialize_deductive_resources(
                codebase_folder=codebase_folder,
                prompts_folder=prompts_folder,
                initialize_embedding_model=initialize_embedding_model,
                use_rag=use_rag  # Pass the use_rag flag
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
            coded_meaning_unit_list = assign_codes_to_meaning_units(
                meaning_unit_list=meaning_unit_object_list,
                coding_instructions=coding_instructions,
                processed_codes=processed_codes,
                index=faiss_index if use_rag else None,  # Pass FAISS index only if RAG is used
                top_k=5,
                context_size=5,
                use_rag=use_rag,
                codebase=processed_codes if not use_rag else None,  # Provide full codebase if not using RAG
                completion_model=assign_model,
                embedding_model=retrieve_embedding_model if use_rag else None  # Embeddings not needed if not using RAG
            )
            logger.debug(f"Assigned codes using deductive mode with {'RAG' if use_rag else 'full codebase'}.")
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
    # Load configurations from config.json
    config_file_path = 'config.json'  # Adjust the path if needed
    try:
        config = load_config(config_file_path)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        exit(1)

    # Run the main function with loaded configurations
    main(config)
