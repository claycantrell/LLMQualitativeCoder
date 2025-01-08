# main.py

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Import the centralized logging setup function
from logging_config import setup_logging

from data_handlers import FlexibleDataHandler
from utils import (
    load_environment_variables,
    load_config,
    load_prompt_file,
    initialize_deductive_resources
)
from qual_functions import (
    assign_codes_to_meaning_units
)
from validator import run_validation, replace_nan_with_null

def main(config: Dict[str, Any]):
    """
    Main function to execute the qualitative coding pipeline.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary loaded from config.json.
    """
    # Load configurations
    coding_mode = config.get('coding_mode', 'deductive')
    use_parsing = config.get('use_parsing', True)
    parse_model = config.get('parse_model', 'gpt-4o-mini')
    assign_model = config.get('assign_model', 'gpt-4o-mini')
    data_format = config.get('data_format', 'interview')
    speaking_turns_per_prompt = config.get('speaking_turns_per_prompt', 1)
    meaning_units_per_assignment_prompt = config.get('meaning_units_per_assignment_prompt', 1)
    context_size = config.get('context_size', 5)

    # Paths configuration
    paths = config.get('paths', {})
    prompts_folder = paths.get('prompts_folder', 'prompts')
    codebase_folder = paths.get('codebase_folder', 'qual_codebase')
    json_folder = paths.get('json_folder', 'json_transcripts')
    config_folder = paths.get('config_folder', 'configs')

    # Selected files
    selected_codebase = config.get('selected_codebase', 'new_schema.jsonl')
    selected_json_file = config.get('selected_json_file', 'your_movie_script.json')
    parse_prompt_file = config.get('parse_prompt_file', 'parse_prompt.txt')
    inductive_coding_prompt_file = config.get('inductive_coding_prompt_file', 'inductive_prompt.txt')
    deductive_coding_prompt_file = config.get('deductive_coding_prompt_file', 'deductive_prompt.txt')

    # Output configuration
    output_folder = config.get('output_folder', 'outputs')

    # Logging configuration (centralized)
    enable_logging = config.get('enable_logging', True)
    logging_level_str = config.get('logging_level', 'DEBUG')
    log_to_file = config.get('log_to_file', True)
    log_file_path = config.get('log_file_path', 'logs/application.log')
    setup_logging(enable_logging, logging_level_str, log_to_file, log_file_path)
    logger = logging.getLogger(__name__)

    logger.info("Starting the main pipeline.")

    # Stage 1: Environment Setup
    try:
        load_environment_variables()
        logger.debug("Environment variables loaded and validated.")
    except ValueError as e:
        logger.error(f"Failed to load environment variables: {e}")
        return
    except OSError as e:
        logger.error(f"OS error encountered while loading environment variables: {e}")
        return

    # Load parse instructions if parsing is enabled
    parse_instructions = ""
    if use_parsing:
        try:
            parse_instructions = load_prompt_file(prompts_folder, parse_prompt_file, description='parse instructions')
            logger.debug("Parse instructions loaded.")
        except (FileNotFoundError, ValueError, OSError) as e:
            logger.error(f"Failed to load parse instructions: {e}")
            return

    # Load data format configuration
    data_format_config_path = Path(config_folder) / 'data_format_config.json'
    try:
        data_format_config = load_config(str(data_format_config_path))
        logger.debug("Data format configuration loaded.")
    except (FileNotFoundError, json.JSONDecodeError, OSError, ValueError) as e:
        logger.error(f"Failed to load data format configuration: {e}")
        return

    if data_format not in data_format_config:
        logger.error(f"No configuration found for data format: {data_format}")
        raise ValueError(f"No configuration found for data format: {data_format}")

    format_config = data_format_config[data_format]
    content_field = format_config.get('content_field')
    speaker_field = format_config.get('speaker_field')
    list_field = format_config.get('list_field')
    source_id_field = format_config.get('source_id_field')
    filter_rules = format_config.get('filter_rules', [])  # Get filter_rules from format_config

    if not content_field:
        logger.error(f"'content_field' not specified in data format configuration for '{data_format}'")
        raise ValueError(f"'content_field' not specified in data format configuration for '{data_format}'")

    # Determine the data file to load based on selected_json_file
    data_file = selected_json_file
    file_path = Path(json_folder) / data_file
    if not file_path.exists():
        logger.error(f"Data file '{file_path}' not found.")
        raise FileNotFoundError(f"Data file '{file_path}' not found.")

    # Stage 2: Data Loading and Transformation Using FlexibleDataHandler
    try:
        data_handler = FlexibleDataHandler(
            file_path=str(file_path),
            parse_instructions=parse_instructions,
            completion_model=parse_model,
            content_field=content_field,
            speaker_field=speaker_field,
            list_field=list_field,
            filter_rules=filter_rules,
            use_parsing=use_parsing,
            source_id_field=source_id_field,
            speaking_turns_per_prompt=speaking_turns_per_prompt
        )
        data_df = data_handler.load_data()
        logger.debug(f"Loaded data with shape {data_df.shape}.")
        meaning_unit_object_list = data_handler.transform_data(data_df)
        logger.debug(f"Transformed data into {len(meaning_unit_object_list)} meaning units.")
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as e:
        logger.error(f"Data loading and transformation failed: {e}")
        return

    if not meaning_unit_object_list:
        logger.warning("No meaning units to process. Exiting.")
        return

    # Stage 3: Code Assignment
    if coding_mode == "deductive":
        try:
            # Initialize deductive resources (always load the full codebase)
            processed_codes, coding_instructions = initialize_deductive_resources(
                codebase_folder=codebase_folder,
                prompts_folder=prompts_folder,
                selected_codebase=selected_codebase,
                deductive_prompt_file=deductive_coding_prompt_file
            )
            logger.debug(f"Initialized deductive resources with {len(processed_codes)} processed codes.")
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to initialize deductive resources: {e}")
            return

        if not processed_codes:
            logger.warning("No processed codes available for deductive coding. Exiting.")
            return

        # Assign codes to meaning units in deductive mode (include full codebase in the prompt)
        try:
            coded_meaning_unit_list = assign_codes_to_meaning_units(
                meaning_unit_list=meaning_unit_object_list,
                coding_instructions=coding_instructions,
                processed_codes=processed_codes,  # Not partial retrieval; full codebase
                codebase=processed_codes,         # Always add entire codebase in the prompt
                completion_model=assign_model,
                context_size=context_size,
                meaning_units_per_assignment_prompt=meaning_units_per_assignment_prompt,
                speaker_field=speaker_field,
                content_field=content_field,
                full_speaking_turns=data_handler.full_data.to_dict(orient='records')
            )
            logger.debug("Assigned codes using deductive mode with full codebase in the prompt.")
        except Exception as e:
            logger.error(f"Failed to assign codes in deductive mode: {e}")
            return

    else:  # Inductive coding
        try:
            # Load inductive coding prompt
            inductive_coding_prompt = load_prompt_file(prompts_folder, inductive_coding_prompt_file, description='inductive coding prompt')
            logger.debug("Inductive coding prompt loaded.")

            # Assign codes to meaning units in inductive mode (no predefined codebase)
            coded_meaning_unit_list = assign_codes_to_meaning_units(
                meaning_unit_list=meaning_unit_object_list,
                coding_instructions=inductive_coding_prompt,
                processed_codes=None,
                codebase=None,
                completion_model=assign_model,
                context_size=context_size,
                meaning_units_per_assignment_prompt=meaning_units_per_assignment_prompt,
                speaker_field=speaker_field,
                content_field=content_field,
                full_speaking_turns=data_handler.full_data.to_dict(orient='records')
            )
            logger.debug("Assigned codes using inductive mode.")
        except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as e:
            logger.error(f"Failed to assign codes in inductive mode: {e}")
            return
        except Exception as e:
            # If there's some unexpected error from the model or similar
            logger.error(f"Unexpected error in inductive coding: {e}")
            return

    # Stage 4: Output Results
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file_pathlib = Path(selected_json_file)
    output_file_basename = input_file_pathlib.stem  # e.g., 'your_movie_script'
    output_file_path = Path(output_folder) / f"{output_file_basename}_output_{timestamp}.json"

    # Get document-level metadata from data_handler
    document_metadata = data_handler.document_metadata

    try:
        # Prepare the output data with document-level metadata at the top
        output_data = {
            "document_metadata": document_metadata,
            "meaning_units": [unit.to_dict() for unit in coded_meaning_unit_list]
        }

        # Replace NaN values with null
        output_data = replace_nan_with_null(output_data)

        with output_file_path.open('w', encoding='utf-8') as outfile:
            json.dump(output_data, outfile, indent=2)
        logger.info(f"Coded meaning units saved to '{output_file_path}'.")
    except (OSError, TypeError, ValueError) as e:
        logger.error(f"Failed to save output: {e}")
        return

    # Update master log file with output file name and config
    try:
        master_log_file_obj = Path('logs') / 'master_log.jsonl'
        master_log_file_obj.parent.mkdir(parents=True, exist_ok=True)
        with master_log_file_obj.open('a', encoding='utf-8') as log_file:
            log_entry = {
                'timestamp': timestamp,
                'output_file': str(output_file_path),
                'config': config
            }
            log_file.write(json.dumps(log_entry) + '\n')
        logger.info(f"Master log updated at '{master_log_file_obj}'.")
    except OSError as e:
        logger.error(f"Failed to update master log: {e}")

    # Stage 5: Validation
    try:
        logger.info("Starting validation process.")
        
        validation_report_filename = f"{output_file_basename}_validation_report.json"
        validation_report_path = Path(output_folder) / validation_report_filename
        
        run_validation(
            input_file=str(Path(json_folder) / selected_json_file),
            output_file=str(output_file_path),
            report_file=str(validation_report_path),
            similarity_threshold=1.0,  # Exact match
            filter_rules=filter_rules,
            input_list_field=list_field,
            output_list_field='meaning_units',
            text_field=content_field,
            source_id_field=source_id_field
        )
        logger.info(f"Validation process completed. Report saved to '{validation_report_path}'.")
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as e:
        logger.error(f"Validation process failed: {e}")
        return
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        return

if __name__ == "__main__":
    """
    Entry point for the application.
    """
    config_file_path = 'configs/config.json'  # Adjust the path if needed
    try:
        config = load_config(config_file_path)
    except (FileNotFoundError, ValueError, OSError, json.JSONDecodeError) as e:
        print(f"Failed to load configuration: {e}")
        exit(1)

    main(config)
