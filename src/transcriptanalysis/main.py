# transcriptanalysis/main.py

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import sys

from importlib import resources

# Define Project Root, assuming this file is src/transcriptanalysis/main.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

from .logging_config import setup_logging  # Relative import
from .config_schemas import ConfigModel, DataFormatConfig
from .utils import (
    load_environment_variables,
    load_config,
    load_data_format_config,
    load_prompt_file
)
from .data_handlers import FlexibleDataHandler
from .qual_functions import assign_codes_to_meaning_units
from .validator import run_validation, replace_nan_with_null
from .langchain_llm import LangChainLLM


def main(config: ConfigModel):
    """
    Main function to execute the qualitative coding pipeline.
    """
    setup_logging(
        enable_logging=config.enable_logging,
        logging_level_str=config.logging_level,
        log_to_file=config.log_to_file,
        log_file_path=config.log_file_path
    )
    logger = logging.getLogger(__name__)

    logger.info("Starting the main pipeline.")

    # Stage 1: Environment Setup
    env_vars = load_environment_variables()
    if not config.parse_llm_config.api_key and env_vars.get("OPENAI_API_KEY"):
        config.parse_llm_config.api_key = env_vars["OPENAI_API_KEY"]
    if not config.assign_llm_config.api_key and env_vars.get("HUGGINGFACE_API_KEY"):
        config.assign_llm_config.api_key = env_vars["HUGGINGFACE_API_KEY"]

    # Load prompts
    prompt_dir = Path(__file__).resolve().parent / "prompts"

    # Check for custom prompts in user directory first
    project_root = Path(__file__).resolve().parent.parent.parent
    user_prompt_dir = project_root / "data" / "user_uploads" / "prompts"
    default_prompt_dir = Path(__file__).resolve().parent / "prompts"

    parse_instructions = ""
    if config.use_parsing:
        parse_instructions = load_prompt_file(
            'transcriptanalysis.prompts',      # Package path
            config.parse_prompt_file,          # Filename, e.g., 'parse.txt'
            description='parse instructions'
        )

    # Load data format configuration using importlib.resources
    data_format_config = None
    format_config = None
    
    # Check if we're using a custom data format from dynamic configuration
    if config.data_format == "custom" and hasattr(config, "custom_format_config"):
        # Use the custom format config directly
        logger.info("Using custom data format configuration from dynamic config")
        format_config = config.custom_format_config
    else:
        # Load from standard config file
        try:
            with resources.path('transcriptanalysis.configs', 'data_format_config.json') as data_format_config_path:
                data_format_config: DataFormatConfig = load_data_format_config(str(data_format_config_path))
        except FileNotFoundError:
            logger.error("File 'transcriptanalysis.configs/data_format_config.json' not found.")
            raise

        if config.data_format not in data_format_config:
            logger.error(f"No configuration found for data format: {config.data_format}")
            raise ValueError(f"No configuration found for data format: {config.data_format}")

        format_config = data_format_config[config.data_format]

    # Determine the data file to load
    # Check if it's a full path to a file first (for user-uploaded files)
    if Path(config.selected_json_file).exists() and Path(config.selected_json_file).is_file():
        file_path = Path(config.selected_json_file)
        logger.info(f"Using specified file path: {file_path}")
    # Check user uploads directory if available
    elif config.paths.user_uploads_folder and Path(config.paths.user_uploads_folder).exists():
        user_file_path = Path(config.paths.user_uploads_folder) / config.selected_json_file
        if user_file_path.exists():
            file_path = user_file_path
            logger.info(f"Using user-uploaded file: {file_path}")
        else:
            # Fall back to package resources
            try:
                with resources.path('transcriptanalysis.json_inputs', config.selected_json_file) as data_file_path:
                    file_path = Path(data_file_path)
                logger.info(f"Using package resource file: {file_path}")
            except FileNotFoundError:
                logger.error(f"File '{config.selected_json_file}' not found in package resources.")
                raise
    else:
        # Fall back to package resources
        try:
            with resources.path('transcriptanalysis.json_inputs', config.selected_json_file) as data_file_path:
                file_path = Path(data_file_path)
            logger.info(f"Using package resource file: {file_path}")
        except FileNotFoundError:
            logger.error(f"File '{config.selected_json_file}' not found in package resources.")
            raise

    if not file_path.exists():
        logger.error(f"Data file '{file_path}' not found.")
        raise FileNotFoundError(f"Data file '{file_path}' not found.")

    # Stage 2: Data Loading & Transform
    # CHANGED: pass context_fields instead of speaker_field
    data_handler = FlexibleDataHandler(
        file_path=str(file_path),
        parse_instructions=parse_instructions,
        completion_model=config.parse_llm_config.model_name,  # Use parse LLM config
        content_field=format_config.content_field,
        context_fields=format_config.context_fields,  # CHANGED
        list_field=format_config.list_field,
        source_id_field=format_config.source_id_field,
        filter_rules=[rule.model_dump() for rule in format_config.filter_rules],
        use_parsing=config.use_parsing,
        preliminary_segments_per_prompt=config.preliminary_segments_per_prompt,  # Renamed
        thread_count=config.thread_count
    )
    data_df = data_handler.load_data()
    meaning_unit_object_list = data_handler.transform_data(data_df)

    if not meaning_unit_object_list:
        logger.warning("No meaning units to process. Exiting.")
        return

    # Stage 3: Code Assignment
    if config.coding_mode == "deductive":
        # Check if there's a custom deductive prompt
        custom_prompt_file = user_prompt_dir / "deductive.txt"
        default_prompt_file = default_prompt_dir / config.deductive_coding_prompt_file
        prompt_path = custom_prompt_file if custom_prompt_file.exists() else default_prompt_file
        
        # Load the prompt content
        with open(prompt_path, 'r') as f:
            coding_instructions = f.read()
        
        logger.info(f"Using {'custom' if custom_prompt_file.exists() else 'default'} deductive prompt")

        # Look for codebase in multiple locations
        codebase_file = Path(config.paths.codebase_folder) / config.selected_codebase
        user_codebase_file = None
        
        # Check if user_uploads_folder is defined in paths
        if hasattr(config.paths, 'user_uploads_folder') and config.paths.user_uploads_folder:
            user_codebase_dir = Path(config.paths.user_uploads_folder) / "codebases"
            user_codebase_file = user_codebase_dir / config.selected_codebase
        
        # Try user codebase first, then default codebase
        if user_codebase_file and user_codebase_file.exists():
            logger.info(f"Using user codebase: {user_codebase_file}")
            codebase_file = user_codebase_file
        elif not codebase_file.exists():
            logger.error(f"List of codes file '{codebase_file}' not found.")
            logger.error(f"Also checked user location: {user_codebase_file}")
            raise FileNotFoundError(f"List of codes file '{codebase_file}' not found.")

        with codebase_file.open('r', encoding='utf-8') as file:
            processed_codes = [json.loads(line) for line in file if line.strip()]

        # Prepare preliminary segments from data_df for context
        preliminary_segments = []
        for _, row in data_df.iterrows():
            preliminary_segments.append({
                'source_id': str(row['source_id']),
                'content': row.get(format_config.content_field, ''),
                **{field: row.get(field, '') for field in format_config.context_fields if field in row}
            })

        coded_meaning_unit_list = assign_codes_to_meaning_units(
            meaning_unit_list=meaning_unit_object_list,
            coding_instructions=coding_instructions,
            processed_codes=processed_codes,
            codebase=processed_codes,
            llm_config=config.assign_llm_config,
            context_size=config.context_size,
            meaning_units_per_assignment_prompt=config.meaning_units_per_assignment_prompt,
            context_fields=format_config.context_fields,
            content_field=format_config.content_field,
            full_preliminary_segments=preliminary_segments,
            thread_count=config.thread_count
        )
    else:  # inductive
        # Check if there's a custom inductive prompt
        custom_prompt_file = user_prompt_dir / "inductive.txt"
        default_prompt_file = default_prompt_dir / config.inductive_coding_prompt_file
        prompt_path = custom_prompt_file if custom_prompt_file.exists() else default_prompt_file
        
        # Load the prompt content
        with open(prompt_path, 'r') as f:
            coding_instructions = f.read()
        
        logger.info(f"Using {'custom' if custom_prompt_file.exists() else 'default'} inductive prompt")

        # Prepare preliminary segments from data_df for context
        preliminary_segments = []
        for _, row in data_df.iterrows():
            preliminary_segments.append({
                'source_id': str(row['source_id']),
                'content': row.get(format_config.content_field, ''),
                **{field: row.get(field, '') for field in format_config.context_fields if field in row}
            })

        coded_meaning_unit_list = assign_codes_to_meaning_units(
            meaning_unit_list=meaning_unit_object_list,
            coding_instructions=coding_instructions,
            processed_codes=None,
            codebase=None,
            llm_config=config.assign_llm_config,
            context_size=config.context_size,
            meaning_units_per_assignment_prompt=config.meaning_units_per_assignment_prompt, 
            context_fields=format_config.context_fields,
            content_field=format_config.content_field,
            full_preliminary_segments=preliminary_segments,
            thread_count=config.thread_count
        )

    # Stage 4: Output Results
    output_folder_name = config.output_folder # e.g., "outputs"
    output_dir = PROJECT_ROOT / output_folder_name
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_basename = Path(config.selected_json_file).stem
    output_file_path = output_dir / f"{output_file_basename}_output_{timestamp}.json"

    document_metadata = data_handler.document_metadata

    output_data = {
        "document_metadata": document_metadata,
        "meaning_units": [unit.to_dict() for unit in coded_meaning_unit_list]
    }
    output_data = replace_nan_with_null(output_data)

    with output_file_path.open('w', encoding='utf-8') as outfile:
        json.dump(output_data, outfile, indent=2)

    logger.info(f"Coded meaning units saved to '{output_file_path}'.")

    # Stage 5: Validation
    validation_report_filename = f"{output_file_basename}_validation_report.json"
    validation_report_path = output_dir / validation_report_filename

    run_validation(
        input_file=str(file_path),
        output_file=str(output_file_path),
        report_file=str(validation_report_path),
        similarity_threshold=1.0,
        filter_rules=[rule.model_dump() for rule in format_config.filter_rules],
        input_list_field=format_config.list_field,
        output_list_field='meaning_units',
        text_field=format_config.content_field,
        source_id_field=format_config.source_id_field,
        meaning_unit_uuid_field='meaning_unit_uuid'  # NEW: Include meaning_unit_uuid in validation
    )
    logger.info(f"Validation completed. Report saved to '{validation_report_path}'.")


def run():
    """
    Entry point for the script. Loads the configuration and invokes the main function.
    """
    try:
        # Access the config.json from the package resources using importlib.resources
        with resources.path('transcriptanalysis.configs', 'config.json') as config_path:
            # Convert Path object to string if necessary
            config: ConfigModel = load_config(str(config_path))
    except Exception as e:
        print(f"Failed to load configuration: {e}", file=sys.stderr)
        sys.exit(1)

    main(config)


if __name__ == "__main__":
    run()
