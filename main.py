# main.py

import os
import json
import logging
from typing import List, Dict, Optional, Any, Tuple
import faiss
from data_handlers import (
    InterviewDataHandler,
    NewsDataHandler
)
from utils import (
    load_environment_variables,
    load_parse_instructions,
    load_custom_coding_prompt,
    initialize_deductive_resources
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
        use_parsing=False,
        use_rag=False,
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
