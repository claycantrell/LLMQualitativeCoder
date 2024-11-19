# tests/conftest.py
import pytest
from unittest.mock import patch, mock_open
import json

@pytest.fixture
def mock_files():
    """
    Fixture to mock file I/O operations for different file paths.
    """
    # Sample JSON data to be returned when 'json_data/test_data.json' is opened
    sample_json_data = [
        {
            "id": 1,
            "length_of_time_spoken_seconds": 120.5,
            "text_context": "This is a sample transcript from Participant A.",
            "speaker_name": "Participant A"
        },
        {
            "id": 2,
            "length_of_time_spoken_seconds": 95.0,
            "text_context": "This is another sample transcript from Participant B.",
            "speaker_name": "Participant B"
        }
    ]

    # Mock schema configuration for 'configs/data_format_config.json'
    schema_config = {
        "transcript": {
            "content_field": "text_context",
            "fields": {
                "id": "int",
                "length_of_time_spoken_seconds": "float",
                "text_context": "str",
                "speaker_name": "str"
            }
        },
        "news": {
            "content_field": "content",
            "fields": {
                "id": "int",
                "title": "str",
                "content": "str"
            }
        }
    }

    # Mock data for 'codebase/test_codebase.jsonl'
    codebase_data = [
        {"text": "Code 1", "metadata": {"id": 1}},
        {"text": "Code 2", "metadata": {"id": 2}}
    ]

    # Mock content for prompt files
    prompt_content = "This is a mock prompt."

    def mocked_open_func(file, mode='r', *args, **kwargs):
        if file == 'json_data/test_data.json':
            return mock_open(read_data=json.dumps(sample_json_data)).return_value
        elif file == 'configs/data_format_config.json':
            return mock_open(read_data=json.dumps(schema_config)).return_value
        elif file == 'codebase/test_codebase.jsonl':
            # Join the JSON lines for the .jsonl file
            return mock_open(read_data='\n'.join([json.dumps(item) for item in codebase_data])).return_value
        elif file.startswith('prompts/'):
            return mock_open(read_data=prompt_content).return_value
        else:
            # For any other files, return an empty mock
            return mock_open().return_value

    # Patch 'builtins.open' with our mocked_open_func
    with patch('builtins.open', new=mocked_open_func):
        yield
