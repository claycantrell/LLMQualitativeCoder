# main.py

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from validator import replace_nan_with_null
from data_handlers import FlexibleDataHandler
from utils import (
    load_environment_variables,
    load_config,
    load_parse_instructions,
    load_inductive_coding_prompt,
    load_deductive_coding_prompt,
    initialize_deductive_resources,
    load_data_format_config  # Updated function
)
from qual_functions import (
    MeaningUnit,
    assign_codes_to_meaning_units
)
from validator import run_validation  # Import the validation function

def main(config: Dict[str, Any]):
    """
    Main function to execute the qualitative coding pipeline.
    
    Args:
        config (Dict[str, Any]): Configuration dictionary loaded from config.json.
    """
    # Load configurations
    coding_mode = config.get('coding_mode', 'deductive')
    use_parsing = config.get('use_parsing', True)
    use_rag = config.get('use_rag', True)
    parse_model = config.get('parse_model', 'gpt-4o-mini')
    assign_model = config.get('assign_model', 'gpt-4o-mini')
    initialize_embedding_model = config.get('initialize_embedding_model', 'text-embedding-3-small')
    retrieve_embedding_model = config.get('retrieve_embedding_model', 'text-embedding-3-small')
    data_format = config.get('data_format', 'interview')
    speaking_turns_per_prompt = config.get('speaking_turns_per_prompt', 1)  # Existing parameter
    meaning_units_per_assignment_prompt = config.get('meaning_units_per_assignment_prompt', 1)  # New parameter

    # Paths configuration
    paths = config.get('paths', {})
    prompts_folder = paths.get('prompts_folder', 'prompts')
    codebase_folder = paths.get('codebase_folder', 'qual_codebase')
    json_folder = paths.get('json_folder', 'json_transcripts')
    config_folder = paths.get('config_folder', 'configs')

    # Selected files
    selected_codebase = config.get('selected_codebase', 'new_schema.jsonl')
    selected_json_file = config.get('selected_json_file', 'your_movie_script.json')  # Replace with your actual file name
    parse_prompt_file = config.get('parse_prompt_file', 'parse_prompt.txt')
    inductive_coding_prompt_file = config.get('inductive_coding_prompt_file', 'inductive_prompt.txt')
    deductive_coding_prompt_file = config.get('deductive_coding_prompt_file', 'deductive_prompt.txt')

    # Output configuration
    output_folder = config.get('output_folder', 'outputs')
    # Removed output_format as JSON is the only supported format

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

    # Load data format configuration
    data_format_config_path = os.path.join(config_folder, 'data_format_config.json')
    try:
        data_format_config = load_data_format_config(data_format_config_path)
        logger.debug("Data format configuration loaded.")
    except Exception as e:
        logger.error(f"Failed to load data format configuration: {e}")
        return

    if data_format not in data_format_config:
        logger.error(f"No configuration found for data format: {data_format}")
        raise ValueError(f"No configuration found for data format: {data_format}")

    format_config = data_format_config[data_format]
    content_field = format_config.get('content_field')
    speaker_field = format_config.get('speaker_field')
    list_field = format_config.get('list_field')

    if not content_field:
        logger.error(f"'content_field' not specified in data format configuration for '{data_format}'")
        raise ValueError(f"'content_field' not specified in data format configuration for '{data_format}'")

    # Determine the data file to load based on selected_json_file
    data_file = selected_json_file
    file_path = os.path.join(json_folder, data_file)
    if not os.path.exists(file_path):
        logger.error(f"Data file '{file_path}' not found.")
        raise FileNotFoundError(f"Data file '{file_path}' not found.")

    # Stage 2: Data Loading and Transformation Using FlexibleDataHandler
    try:
        data_handler = FlexibleDataHandler(
            file_path=file_path,
            parse_instructions=parse_instructions,
            completion_model=parse_model,
            content_field=content_field,
            speaker_field=speaker_field,
            list_field=list_field,  # Pass the list_field to the handler
            use_parsing=use_parsing,
            speaking_turns_per_prompt=speaking_turns_per_prompt  # Pass the existing parameter
        )
        data_df = data_handler.load_data()
        logger.debug(f"Loaded data with shape {data_df.shape}.")
        meaning_unit_object_list = data_handler.transform_data(data_df)
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
                embedding_model=retrieve_embedding_model if use_rag else None,
                meaning_units_per_assignment_prompt=meaning_units_per_assignment_prompt,  # Pass the new parameter
                speaker_field=speaker_field  # Pass the speaker_field
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
                embedding_model=None,
                meaning_units_per_assignment_prompt=meaning_units_per_assignment_prompt,  # Pass the new parameter
                speaker_field=speaker_field  # Pass the speaker_field
            )
            logger.debug("Assigned codes using inductive mode.")
        except Exception as e:
            logger.error(f"Failed to assign codes in inductive mode: {e}")
            return

    # Stage 4: Output Results

    os.makedirs(output_folder, exist_ok=True)
    # Timestamp and pathlib for output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_file_pathlib = Path(selected_json_file)
    output_file_basename = input_file_pathlib.stem  # e.g., 'output_cues'
    output_file_path = os.path.join(output_folder, f"{output_file_basename}_output_{timestamp}.json")  # Fixed to .json extension

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

        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(output_data, outfile, indent=2)
        logger.info(f"Coded meaning units saved to '{output_file_path}'.")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
        return

    # Update master log file with output file name and config
    try:
        master_log_file = 'logs/master_log.jsonl'
        os.makedirs(os.path.dirname(master_log_file), exist_ok=True)
        with open(master_log_file, 'a', encoding='utf-8') as log_file:
            log_entry = {
                'timestamp': timestamp,
                'output_file': output_file_path,
                'config': config
            }
            log_file.write(json.dumps(log_entry) + '\n')
        logger.info(f"Master log updated at '{master_log_file}'.")
    except Exception as e:
        logger.error(f"Failed to update master log: {e}")

    # Stage 5: Validation
    try:
        logger.info("Starting validation process.")
        
        # Derive the validation report filename based on the output file's name
        output_file_basename = Path(output_file_path).stem  # e.g., 'output_cues_output_20240427_150000'
        validation_report_filename = f"{output_file_basename}_validation_report.json"  # e.g., 'output_cues_output_20240427_150000_validation_report.json'
        validation_report_path = os.path.join(output_folder, validation_report_filename)
        
        # Run validation with the new report file path, pass content_field as text_field
        validation_report = run_validation(
            input_file=os.path.join(json_folder, selected_json_file),
            output_file=output_file_path,
            report_file=validation_report_path,  # Updated report file path
            similarity_threshold=1.0,  # Exact match
            input_list_field=list_field,    # Pass the list_field for input
            output_list_field='meaning_units',  # Specify the path to the list in output JSON
            text_field=content_field  # Pass the content_field as text_field
        )
        logger.info(f"Validation process completed. Report saved to '{validation_report_path}'.")
    except Exception as e:
        logger.error(f"Validation process failed: {e}")
        return

if __name__ == "__main__":
    """
    Entry point for the application.
    """
    # Load configurations from config.json
    config_file_path = 'configs/config.json'  # Adjust the path if needed
    try:
        config = load_config(config_file_path)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        exit(1)

    # Run the main function with loaded configurations
    main(config)
