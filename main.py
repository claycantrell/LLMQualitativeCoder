# main.py
import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple, Type
from data_handlers import FlexibleDataHandler
from utils import (
    load_environment_variables,
    load_parse_instructions,
    load_deductive_coding_prompt,
    load_inductive_coding_prompt,
    initialize_deductive_resources,
    create_dynamic_model_for_format,
    load_schema_config,
    load_config
)
from qual_functions import (
    MeaningUnit,
    assign_codes_to_meaning_units
)

def main(config: Dict[str, Any]):
    # Load configurations
    coding_mode = config.get('coding_mode', 'deductive')
    use_parsing = config.get('use_parsing', True)
    use_rag = config.get('use_rag', True)
    parse_model = config.get('parse_model', 'gpt-4o-mini')
    assign_model = config.get('assign_model', 'gpt-4o-mini')
    initialize_embedding_model = config.get('initialize_embedding_model', 'text-embedding-3-small')
    retrieve_embedding_model = config.get('retrieve_embedding_model', 'text-embedding-3-small')
    data_format = config.get('data_format', 'interview')
    speaking_turns_per_prompt = config.get('speaking_turns_per_prompt', 1)  # New parameter

    # Paths configuration
    paths = config.get('paths', {})
    prompts_folder = paths.get('prompts_folder', 'prompts')
    codebase_folder = paths.get('codebase_folder', 'qual_codebase')
    json_folder = paths.get('json_folder', 'json_transcripts')
    config_folder = paths.get('config_folder', 'configs')

    # Selected files
    selected_codebase = config.get('selected_codebase', 'new_schema.jsonl')
    selected_json_file = config.get('selected_json_file', 'output_cues.json')
    parse_prompt_file = config.get('parse_prompt_file', 'parse_prompt.txt')
    inductive_coding_prompt_file = config.get('inductive_coding_prompt_file', 'inductive_prompt.txt')
    deductive_coding_prompt_file = config.get('deductive_coding_prompt_file', 'deductive_prompt.txt')

    # Output configuration
    output_folder = config.get('output_folder', 'outputs')
    output_format = config.get('output_format', 'json')
    output_file_name = config.get('output_file_name', 'coded_meaning_units')

    # Logging configuration
    enable_logging = config.get('enable_logging', True)
    logging_level_str = config.get('logging_level', 'DEBUG')
    logging_level = getattr(logging, logging_level_str.upper(), logging.DEBUG)
    log_to_file = config.get('log_to_file', True)
    log_file_path = config.get('log_file_path', 'logs/application.log')

    # Configure logging
    if enable_logging:
        handlers = [logging.StreamHandler()]
        if log_to_file:
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
            handlers.append(logging.FileHandler(log_file_path))
        logging.basicConfig(
            level=logging_level,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=handlers
        )
    else:
        logging.basicConfig(level=logging.CRITICAL)
    logger = logging.getLogger(__name__)

    logger.info("Starting the main pipeline.")

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
            parse_instructions = load_parse_instructions(prompts_folder, parse_prompt_file)
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

    # Determine the data file to load based on selected_json_file
    data_file = selected_json_file
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
            use_parsing=use_parsing,
            speaking_turns_per_prompt=speaking_turns_per_prompt  # Pass the new parameter
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

    # Stage 3: Code Assignment
    if coding_mode == "deductive":
        try:
            # Initialize deductive resources with conditional FAISS initialization
            processed_codes, faiss_index, coding_instructions = initialize_deductive_resources(
                codebase_folder=codebase_folder,
                prompts_folder=prompts_folder,
                initialize_embedding_model=initialize_embedding_model,
                use_rag=use_rag,
                selected_codebase=selected_codebase,
                deductive_prompt_file=deductive_coding_prompt_file  # Pass the prompt file from config
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
                index=faiss_index if use_rag else None,
                top_k=config.get('top_k', 5),
                context_size=config.get('context_size', 5),
                use_rag=use_rag,
                codebase=processed_codes if not use_rag else None,
                completion_model=assign_model,
                embedding_model=retrieve_embedding_model if use_rag else None
            )
            logger.debug(f"Assigned codes using deductive mode with {'RAG' if use_rag else 'full codebase'}.")
        except Exception as e:
            logger.error(f"Failed to assign codes in deductive mode: {e}")
            return

    else:  # Inductive coding
        try:
            # Load inductive coding prompt
            inductive_coding_prompt = load_inductive_coding_prompt(prompts_folder, inductive_coding_prompt_file)
            logger.debug("Inductive coding prompt loaded.")

            # Assign codes to meaning units in inductive mode
            coded_meaning_unit_list = assign_codes_to_meaning_units(
                meaning_unit_list=meaning_unit_object_list,
                coding_instructions=inductive_coding_prompt,
                processed_codes=None,
                index=None,
                top_k=None,
                context_size=config.get('context_size', 5),
                use_rag=False,
                codebase=None,
                completion_model=assign_model,
                embedding_model=None
            )
            logger.debug("Assigned codes using inductive mode.")
        except Exception as e:
            logger.error(f"Failed to assign codes in inductive mode: {e}")
            return

    # Stage 4: Output Results
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = os.path.join(output_folder, f"{output_file_name}.{output_format}")

    try:
        if output_format == 'json':
            with open(output_file_path, 'w', encoding='utf-8') as outfile:
                json.dump([unit.to_dict() for unit in coded_meaning_unit_list], outfile, indent=2)
            logger.info(f"Coded meaning units saved to '{output_file_path}'.")
        elif output_format == 'csv':
            import csv

            with open(output_file_path, 'w', encoding='utf-8', newline='') as csvfile:
                fieldnames = ['unique_id', 'meaning_unit_string', 'metadata', 'assigned_codes']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                for unit in coded_meaning_unit_list:
                    writer.writerow({
                        'unique_id': unit.unique_id,
                        'meaning_unit_string': unit.meaning_unit_string,
                        'metadata': json.dumps(unit.metadata),
                        'assigned_codes': json.dumps([code.__dict__ for code in unit.assigned_code_list])
                    })
            logger.info(f"Coded meaning units saved to '{output_file_path}'.")
        else:
            logger.error(f"Unsupported output format: {output_format}")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")

if __name__ == "__main__":
    # Load configurations from config.json
    config_file_path = 'configs/config.json'  # Adjust the path if needed
    try:
        config = load_config(config_file_path)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        exit(1)

    # Run the main function with loaded configurations
    main(config)
