# validator.py
import os
import json
import logging
from typing import Dict, Any, List, Tuple
import difflib

# Configure logging
logger = logging.getLogger(__name__)

def load_input_file(input_file_path: str, text_field: str = 'text_context') -> Dict[int, Dict[str, Any]]:
    """
    Loads the input file and returns a dictionary mapping id to speaking turn data.

    Args:
        input_file_path (str): Path to the input JSON file.
        text_field (str): The field name that contains the speaking turn text.

    Returns:
        Dict[int, Dict[str, Any]]: Mapping from id to speaking turn data.
    """
    try:
        with open(input_file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)
    except Exception as e:
        logger.error(f"Error loading input file '{input_file_path}': {e}")
        raise e

    speaking_turns = {}
    for item in data:
        source_id = item.get('id')
        if source_id is None:
            logger.warning(f"Skipping speaking turn without 'id': {item}")
            continue
        speaking_turns[source_id] = item
    logger.info(f"Loaded {len(speaking_turns)} speaking turns from input file.")
    return speaking_turns

def load_output_file(output_file_path: str) -> Dict[int, List[Dict[str, Any]]]:
    """
    Loads the output file and returns a dictionary mapping id to a list of meaning units.

    Args:
        output_file_path (str): Path to the output JSON file.

    Returns:
        Dict[int, List[Dict[str, Any]]]: Mapping from id to list of meaning units.
    """
    try:
        with open(output_file_path, 'r', encoding='utf-8') as outfile:
            data = json.load(outfile)
    except Exception as e:
        logger.error(f"Error loading output file '{output_file_path}': {e}")
        raise e

    meaning_units = {}
    for unit in data:
        metadata = unit.get('metadata', {})
        # Use 'id' instead of 'source_id'
        source_id = metadata.get('id')
        if source_id is None:
            logger.warning(f"Skipping meaning unit without 'id': {unit}")
            continue
        if source_id not in meaning_units:
            meaning_units[source_id] = []
        meaning_units[source_id].append(unit)
    logger.info(f"Loaded meaning units derived from {len(meaning_units)} unique ids.")
    return meaning_units

def compare_texts(original: str, concatenated: str) -> Tuple[bool, str]:
    """
    Compares two texts and returns whether they are identical and the diff.

    Args:
        original (str): Original speaking turn text.
        concatenated (str): Concatenated meaning units text.

    Returns:
        Tuple[bool, str]: (is_identical, diff_string)
    """
    is_identical = original.strip() == concatenated.strip()
    if is_identical:
        return True, ""
    else:
        # Generate a unified diff
        original_lines = original.strip().splitlines()
        concatenated_lines = concatenated.strip().splitlines()
        diff = difflib.unified_diff(
            original_lines,
            concatenated_lines,
            fromfile='Original Speaking Turn',
            tofile='Concatenated Meaning Units',
            lineterm=''
        )
        diff_string = '\n'.join(diff)
        return False, diff_string

def generate_report(
    speaking_turns: Dict[int, Dict[str, Any]],
    meaning_units: Dict[int, List[Dict[str, Any]]],
    similarity_threshold: float = 1.0,  # Exact match
    report_file: str = 'validation_report.json'
) -> Dict[str, Any]:
    """
    Generates a report of inconsistencies and missing meaning units.

    Args:
        speaking_turns (Dict[int, Dict[str, Any]]): Mapping from id to speaking turn data.
        meaning_units (Dict[int, List[Dict[str, Any]]]): Mapping from id to list of meaning units.
        similarity_threshold (float, optional): Threshold for similarity. Defaults to 1.0 for exact match.
        report_file (str, optional): Path to save the validation report JSON file. Defaults to 'validation_report.json'.

    Returns:
        Dict[str, Any]: Report containing skipped and inconsistent speaking turns.
    """
    total_speaking_turns = len(speaking_turns)
    total_meaning_units = sum(len(units) for units in meaning_units.values())
    logger.info(f"Total speaking turns: {total_speaking_turns}")
    logger.info(f"Total meaning units: {total_meaning_units}")

    skipped_speaking_turns = []
    inconsistent_speaking_turns = []

    for source_id, turn in speaking_turns.items():
        original_text = turn.get('text_context', '').strip()
        units = meaning_units.get(source_id, [])

        if not units:
            skipped_speaking_turns.append({
                'source_id': source_id,
                'speaking_turn_text': original_text,
                'metadata': {k: v for k, v in turn.items() if k != 'text_context'}
            })
            continue

        # Concatenate meaning unit strings in order
        concatenated_text = ' '.join(unit.get('meaning_unit_string', '').strip() for unit in units)

        is_identical, diff = compare_texts(original_text, concatenated_text)

        if not is_identical:
            inconsistent_speaking_turns.append({
                'source_id': source_id,
                'speaking_turn_text': original_text,
                'concatenated_meaning_units_text': concatenated_text,
                'diff': diff,
                'metadata': {k: v for k, v in turn.items() if k != 'text_context'}
            })

    # Prepare the report
    report = {
        'skipped_speaking_turns': skipped_speaking_turns,
        'inconsistent_speaking_turns': inconsistent_speaking_turns
    }

    # Save the report to a JSON file
    try:
        logger.debug(f"Saving validation report to '{report_file}'.")
        with open(report_file, 'w', encoding='utf-8') as outfile:
            json.dump(report, outfile, indent=2)
        logger.info(f"Validation report saved to '{report_file}'.")
    except Exception as e:
        logger.error(f"Error saving validation report to '{report_file}': {e}")
        raise e

    # Optionally, print summary
    print("\nValidation Report Summary:")
    print(f"Total Speaking Turns: {total_speaking_turns}")
    print(f"Total Meaning Units: {total_meaning_units}")
    print(f"Skipped Speaking Turns: {len(skipped_speaking_turns)}")
    print(f"Inconsistent Speaking Turns: {len(inconsistent_speaking_turns)}")
    print(f"Detailed report saved to '{report_file}'.")
    print("\nDetails:")
    if skipped_speaking_turns:
        print("\nSkipped Speaking Turns:")
        for skipped in skipped_speaking_turns:
            print(f"- Source ID: {skipped['source_id']}")
            print(f"  Text: {skipped['speaking_turn_text']}")
            print(f"  Metadata: {skipped['metadata']}")
    if inconsistent_speaking_turns:
        print("\nInconsistent Speaking Turns:")
        for inconsistent in inconsistent_speaking_turns:
            print(f"- Source ID: {inconsistent['source_id']}")
            print(f"  Original Text: {inconsistent['speaking_turn_text']}")
            print(f"  Concatenated Meaning Units Text: {inconsistent['concatenated_meaning_units_text']}")
            print(f"  Differences:\n{inconsistent['diff']}")
            print(f"  Metadata: {inconsistent['metadata']}")

    return report

def run_validation(
    input_file: str,
    output_file: str,
    report_file: str = 'validation_report.json',
    similarity_threshold: float = 1.0
) -> Dict[str, Any]:
    """
    Runs the validation process.

    Args:
        input_file (str): Path to the input JSON file containing speaking turns.
        output_file (str): Path to the output JSON file containing meaning units.
        report_file (str, optional): Path to save the validation report JSON file. Defaults to 'validation_report.json'.
        similarity_threshold (float, optional): Threshold for similarity. Defaults to 1.0 for exact match.

    Returns:
        Dict[str, Any]: The validation report.
    """
    # Load files
    speaking_turns = load_input_file(input_file)
    meaning_units = load_output_file(output_file)

    # Generate report
    report = generate_report(speaking_turns, meaning_units, similarity_threshold=similarity_threshold, report_file=report_file)

    return report
