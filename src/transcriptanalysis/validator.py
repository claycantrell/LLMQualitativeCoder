# validator.py

import os
import json
import logging
from typing import Dict, Any, List, Tuple, Optional, Set
import difflib
import re  # Import regex module for text normalization
import math  # Import math module for NaN checks

# Import FlexibleDataHandler
from transcriptanalysis.data_handlers import FlexibleDataHandler

# Configure logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for detailed logs; adjust as needed

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)  # Adjust as needed

# Create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
ch.setFormatter(formatter)

# Add the handlers to the logger
if not logger.handlers:
    logger.addHandler(ch)


def replace_nan_with_null(obj):
    """
    Recursively replace NaN values with None in a data structure.
    """
    if isinstance(obj, float):
        if math.isnan(obj):
            return None
        else:
            return obj
    elif isinstance(obj, dict):
        return {k: replace_nan_with_null(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_nan_with_null(v) for v in obj]
    else:
        return obj


def normalize_text(text: str) -> str:
    """
    Normalizes text by removing extra spaces and newline characters.
    Converts all whitespace sequences to a single space and strips leading/trailing spaces.

    Args:
        text (str): The text to normalize.

    Returns:
        str: The normalized text.
    """
    # Replace any sequence of whitespace characters with a single space
    normalized = re.sub(r'\s+', ' ', text).strip()
    return normalized


def compare_texts(original: str, concatenated: str) -> Tuple[bool, str]:
    """
    Compares two texts after normalizing them by ignoring extra spaces and newline characters.
    Returns whether they are identical and the diff.

    Args:
        original (str): Original preliminary segment text.
        concatenated (str): Concatenated meaning units text.

    Returns:
        Tuple[bool, str]: (is_identical, diff_string)
    """
    normalized_original = normalize_text(original)
    normalized_concatenated = normalize_text(concatenated)

    is_identical = normalized_original == normalized_concatenated
    if is_identical:
        return True, ""
    else:
        # Generate a unified diff based on normalized texts
        original_lines = normalized_original.splitlines()
        concatenated_lines = normalized_concatenated.splitlines()
        diff = difflib.unified_diff(
            original_lines,
            concatenated_lines,
            fromfile='Original Preliminary Segment',
            tofile='Concatenated Meaning Units',
            lineterm=''
        )
        diff_string = '\n'.join(diff)
        return False, diff_string


def generate_report(
    preliminary_segments: Dict[str, Dict[str, Any]],
    meaning_units: Dict[str, List[Dict[str, Any]]],
    similarity_threshold: float = 1.0,
    report_file: str = 'validation_report.json',
    text_field: str = 'text',
    source_id_field: Optional[str] = None,
    meaning_unit_uuid_field: Optional[str] = None,  # NEW: Field name for meaning_unit_uuid
    filtered_source_ids: Optional[Set[str]] = None  # New parameter
) -> Dict[str, Any]:
    """
    Generates a report of inconsistencies and missing meaning units.

    Args:
        preliminary_segments (Dict[str, Dict[str, Any]]): Mapping from source_id to preliminary segment data.
        meaning_units (Dict[str, List[Dict[str, Any]]]): Mapping from source_id to list of meaning units.
        similarity_threshold (float, optional): Threshold for similarity. Defaults to 1.0 for exact match.
        report_file (str, optional): Path to save the validation report JSON file. Defaults to 'validation_report.json'.
        text_field (str, optional): The field name that contains the preliminary segment text.
        source_id_field (Optional[str], optional): The field name that contains the source_id.
        meaning_unit_uuid_field (Optional[str], optional): The field name that contains the meaning_unit_uuid.
        filtered_source_ids (Optional[Set[str]], optional): Set of source_ids that were filtered out.

    Returns:
        Dict[str, Any]: Report containing skipped and inconsistent preliminary segments.
    """
    total_preliminary_segments = len(preliminary_segments)
    total_meaning_units = sum(len(units) for units in meaning_units.values())
    logger.info(f"Total preliminary segments: {total_preliminary_segments}")
    logger.info(f"Total meaning units: {total_meaning_units}")

    skipped_preliminary_segments = []
    inconsistent_preliminary_segments = []

    for source_id, segment in preliminary_segments.items():
        # Skip if the preliminary segment was filtered out
        if filtered_source_ids and source_id in filtered_source_ids:
            continue

        original_text = segment.get(text_field, '').strip()
        units = meaning_units.get(source_id, [])

        if not units:
            skipped_preliminary_segments.append({
                'source_id': source_id,
                'preliminary_segment_text': original_text,
                'metadata': {k: v for k, v in segment.items() if k != text_field}
            })
            continue

        # Concatenate meaning unit strings in order
        concatenated_text = ' '.join(unit.get('meaning_unit_string', '').strip() for unit in units)

        is_identical, diff = compare_texts(original_text, concatenated_text)

        if not is_identical:
            # Collect all meaning_unit_uuids associated with this source_id
            meaning_unit_uuids = [unit.get(meaning_unit_uuid_field, "N/A") for unit in units if meaning_unit_uuid_field and unit.get(meaning_unit_uuid_field)]

            inconsistent_preliminary_segments.append({
                'source_id': source_id,
                'meaning_unit_uuids': meaning_unit_uuids,  # NEW: Include all UUIDs
                'preliminary_segment_text': original_text,
                'concatenated_meaning_units_text': concatenated_text,
                'diff': diff,
                'metadata': {k: v for k, v in segment.items() if k != text_field}
            })

    # Prepare the report
    report = {
        'total_preliminary_segments': total_preliminary_segments,
        'total_meaning_units': total_meaning_units,
        'skipped_preliminary_segments': skipped_preliminary_segments,
        'inconsistent_preliminary_segments': inconsistent_preliminary_segments
    }

    # Replace NaN values with null
    report = replace_nan_with_null(report)

    # Save the report to a JSON file
    try:
        logger.debug(f"Saving validation report to '{report_file}'.")
        with open(report_file, 'w', encoding='utf-8') as outfile:
            json.dump(report, outfile, indent=2)
        logger.info(f"Validation report saved to '{report_file}'.")
    except Exception as e:
        logger.error(f"Error saving validation report to '{report_file}': {e}")
        raise e

    return report


def run_validation(
    input_file: str,
    output_file: str,
    report_file: str = 'validation_report.json',
    similarity_threshold: float = 1.0,
    input_list_field: Optional[str] = None,
    output_list_field: Optional[str] = None,
    text_field: str = 'text',
    source_id_field: Optional[str] = None,
    meaning_unit_uuid_field: Optional[str] = None,  # NEW: Field name for meaning_unit_uuid
    filter_rules: Optional[List[Dict[str, Any]]] = None,  # Ensure filter_rules can be passed
    context_fields: Optional[List[str]] = None,
    use_parsing: bool = True,
    parse_instructions: str = '',
    completion_model: str = 'gpt-4o-mini',
    preliminary_segments_per_prompt: int = 1  # Renamed from speaking_turns_per_prompt
) -> Dict[str, Any]:
    """
    Runs the validation process.

    Args:
        input_file (str): Path to the input JSON file containing preliminary segments.
        output_file (str): Path to the output JSON file containing meaning units.
        report_file (str, optional): Path to save the validation report JSON file. Defaults to 'validation_report.json'.
        similarity_threshold (float, optional): Threshold for similarity. Defaults to 1.0 for exact match.
        input_list_field (Optional[str], optional): Dot-separated path to the list of items within the input JSON. Defaults to None.
        output_list_field (Optional[str], optional): Dot-separated path to the list of items within the output JSON.
        text_field (str, optional): The field name that contains the preliminary segment text.
        source_id_field (Optional[str], optional): The field name that contains the source_id.
        meaning_unit_uuid_field (Optional[str], optional): The field name that contains the meaning_unit_uuid.
        filter_rules (Optional[List[Dict[str, Any]]], optional): List of filter rules to apply.
        context_fields (Optional[List[str]], optional): List of fields to include as context.
        use_parsing (bool, optional): Whether parsing was used in the main processing.
        parse_instructions (str, optional): Parse instructions used in the main processing.
        completion_model (str, optional): Completion model used for parsing.
        preliminary_segments_per_prompt (int, optional): Number of preliminary segments per parsing prompt.

    Returns:
        Dict[str, Any]: The validation report.
    """
    # Load input data using FlexibleDataHandler
    try:
        data_handler = FlexibleDataHandler(
            file_path=input_file,
            parse_instructions=parse_instructions,
            completion_model=completion_model,
            content_field=text_field,
            context_fields=context_fields,
            list_field=input_list_field,
            filter_rules=filter_rules,  # Pass the filter_rules here
            use_parsing=False,  # We don't need to parse again
            source_id_field=source_id_field,
            preliminary_segments_per_prompt=preliminary_segments_per_prompt  # Renamed
        )
        data_df = data_handler.load_data()
        logger.debug(f"Loaded data with shape {data_df.shape}.")
        filtered_out_source_ids = data_handler.filtered_out_source_ids  # Get filtered out source_ids
    except Exception as e:
        logger.error(f"Data loading failed: {e}")
        raise e

    # Use the full_data to include all preliminary_segments in the report
    preliminary_segments = data_handler.full_data.set_index('source_id').to_dict(orient='index')

    # Load meaning units from output file
    meaning_units = load_output_file(output_file, list_field=output_list_field)

    # Generate report
    report = generate_report(
        preliminary_segments,
        meaning_units,
        similarity_threshold=similarity_threshold,
        report_file=report_file,
        text_field=text_field,
        source_id_field=source_id_field,
        meaning_unit_uuid_field=meaning_unit_uuid_field,  # NEW: Pass meaning_unit_uuid_field
        filtered_source_ids=filtered_out_source_ids  # Pass filtered source_ids
    )

    return report


def load_output_file(
    output_file_path: str,
    list_field: Optional[str] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Loads the output file and returns a dictionary mapping source_id to a list of meaning units.

    Args:
        output_file_path (str): Path to the output JSON file.
        list_field (Optional[str]): Dot-separated path to the list of items within the JSON.

    Returns:
        Dict[str, List[Dict[str, Any]]]: Mapping from source_id to list of meaning units.
    """
    data = load_json_file(output_file_path, list_field=list_field)

    # If no list_field is provided, assume the list is directly under a key (e.g., "meaning_units")
    if not list_field and isinstance(data, dict):
        data = data.get('meaning_units', [])

    meaning_units = {}
    for unit in data:
        preliminary_segment = unit.get('preliminary_segment', {})
        source_id = preliminary_segment.get('source_id')
        if source_id is None:
            logger.warning(f"Skipping meaning unit without 'source_id' in preliminary_segment: {unit}")
            continue
        if source_id not in meaning_units:
            meaning_units[source_id] = []
        meaning_units[source_id].append(unit)
    logger.info(f"Loaded meaning units derived from {len(meaning_units)} unique source_ids.")
    return meaning_units


def load_json_file(
    file_path: str,
    list_field: Optional[str] = None
) -> Any:
    """
    Loads a JSON file and optionally navigates to a list of items using list_field.

    Args:
        file_path (str): Path to the JSON file.
        list_field (Optional[str]): Dot-separated path to the list of items within the JSON.

    Returns:
        Any: The data loaded from the JSON file, possibly after navigating to the list_field.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except Exception as e:
        logger.error(f"Error loading JSON file '{file_path}': {e}")
        raise e

    # Navigate to the list of items using list_field if provided
    if list_field:
        keys = list_field.split('.')
        for key in keys:
            if isinstance(data, dict):
                data = data.get(key, [])
            else:
                logger.error(f"Expected dict while accessing '{key}' in 'list_field', but got {type(data)}")
                data = []
                break
    return data
