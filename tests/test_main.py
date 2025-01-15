# tests/test_main.py
import pytest
from unittest.mock import patch, MagicMock, ANY
import TranscriptAnalysis.src.TranscriptAnalysis.main as main
import json
import warnings

# Suppress DeprecationWarnings from faiss.loader
warnings.filterwarnings("ignore", category=DeprecationWarning, module="faiss.loader")

@pytest.mark.parametrize(
    "use_parsing,use_rag",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
)
def test_main_execution_deductive(use_parsing, use_rag, mock_files):
    """
    Parameterized test for main.main(config) in deductive coding mode with various configurations.

    Parameters:
        use_parsing (bool): Whether to use parsing.
        use_rag (bool): Whether to use Retrieval-Augmented Generation.
        mock_files (fixture): Fixture to mock file I/O.
    """
    coding_mode = 'deductive'
    prompt_file = 'deductive_prompt.txt'

    # Define the configuration dictionary based on parameters
    config = {
        'coding_mode': coding_mode,
        'use_parsing': use_parsing,
        'use_rag': use_rag,
        'parse_model': 'test-parse-model',
        'assign_model': 'test-assign-model',
        'initialize_embedding_model': 'test-embedding-model',
        'retrieve_embedding_model': 'test-embedding-model',
        'data_format': 'transcript',  # Ensure this aligns with your data_format_config.json
        'paths': {
            'prompts_folder': 'prompts',
            'codebase_folder': 'codebase',
            'json_folder': 'json_data',
            'config_folder': 'configs'
        },
        'selected_codebase': 'test_codebase.jsonl',
        'selected_json_file': 'test_data.json',
        'parse_prompt_file': 'parse_prompt.txt',
        'inductive_coding_prompt_file': 'inductive_prompt.txt',
        'deductive_coding_prompt_file': 'deductive_prompt.txt',
        'output_folder': 'outputs',
        'output_format': 'json',
        'output_file_name': 'output',
        'enable_logging': False
    }

    # Patch dependencies accordingly
    with patch('main.load_environment_variables'), \
         patch('main.load_parse_instructions', return_value='Parse instructions'), \
         patch('main.create_dynamic_model_for_format') as mock_create_model, \
         patch('main.FlexibleDataHandler') as mock_data_handler_class, \
         patch('main.initialize_deductive_resources') as mock_init_resources, \
         patch('main.assign_codes_to_meaning_units') as mock_assign_codes, \
         patch('main.os.makedirs'), \
         patch('main.json.dump') as mock_json_dump, \
         patch('main.os.path.exists') as mock_exists:

        # Configure the mock for os.path.exists
        def exists_side_effect(path):
            # Define which files exist
            existing_files = [
                'json_data/test_data.json',
                'configs/data_format_config.json',
                'codebase/test_codebase.jsonl',
                'prompts/parse_prompt.txt',
                'prompts/inductive_prompt.txt',
                'prompts/deductive_prompt.txt'
            ]
            return path in existing_files

        mock_exists.side_effect = exists_side_effect

        # Mock the dynamic model creation
        mock_model = MagicMock()
        mock_create_model.return_value = (mock_model, 'text_context')  # Updated content_field to 'text_context'

        # Mock the FlexibleDataHandler
        mock_data_handler = MagicMock()
        # Mock the load_data method to return validated data
        mock_data_handler.load_data.return_value = [
            {"id": 1, "length_of_time_spoken_seconds": 120.5, "text_context": "This is a sample transcript from Participant A.", "speaker_name": "Participant A"},
            {"id": 2, "length_of_time_spoken_seconds": 95.0, "text_context": "This is another sample transcript from Participant B.", "speaker_name": "Participant B"}
        ]
        # Mock the transform_data method to return MeaningUnit objects
        mock_meaning_unit_1 = MagicMock()
        mock_meaning_unit_1.unique_id = 1
        mock_meaning_unit_1.meaning_unit_string = "Parsed Meaning Unit 1"
        mock_meaning_unit_1.metadata = {"id": 1, "length_of_time_spoken_seconds": 120.5, "speaker_name": "Participant A"}

        mock_meaning_unit_2 = MagicMock()
        mock_meaning_unit_2.unique_id = 2
        mock_meaning_unit_2.meaning_unit_string = "Parsed Meaning Unit 2"
        mock_meaning_unit_2.metadata = {"id": 2, "length_of_time_spoken_seconds": 95.0, "speaker_name": "Participant B"}

        mock_data_handler.transform_data.return_value = [mock_meaning_unit_1, mock_meaning_unit_2]
        mock_data_handler_class.return_value = mock_data_handler

        # Mock initialize_deductive_resources to return expected values (three mocks)
        mock_init_resources.return_value = (MagicMock(), MagicMock(), MagicMock())

        # Mock assign_codes_to_meaning_units
        mock_coded_unit_1 = MagicMock()
        mock_coded_unit_1.to_dict.return_value = {
            "unique_id": 1,
            "meaning_unit_string": "Parsed Meaning Unit 1",
            "assigned_code_list": [{"code_name": "Code 1", "code_justification": "Justification 1"}],
            "metadata": {"id": 1, "length_of_time_spoken_seconds": 120.5, "speaker_name": "Participant A"}
        }

        mock_coded_unit_2 = MagicMock()
        mock_coded_unit_2.to_dict.return_value = {
            "unique_id": 2,
            "meaning_unit_string": "Parsed Meaning Unit 2",
            "assigned_code_list": [{"code_name": "Code 2", "code_justification": "Justification 2"}],
            "metadata": {"id": 2, "length_of_time_spoken_seconds": 95.0, "speaker_name": "Participant B"}
        }

        mock_assign_codes.return_value = [mock_coded_unit_1, mock_coded_unit_2]

        # Run the main function with the mocked config
        main.main(config)

        # Assertions to ensure initialization functions were called as expected
        mock_init_resources.assert_called_once_with(
            codebase_folder='codebase',
            prompts_folder='prompts',
            initialize_embedding_model='test-embedding-model',
            use_rag=use_rag,
            selected_codebase='test_codebase.jsonl',
            deductive_prompt_file='deductive_prompt.txt'
        )

        # Assert that assign_codes_to_meaning_units was called once
        mock_assign_codes.assert_called_once()

        # Assertions to ensure data loading and transformation
        mock_data_handler.load_data.assert_called_once()
        mock_data_handler.transform_data.assert_called_once()

        # Ensure that the output was attempted to be written
        mock_coded_unit_1.to_dict.assert_called_once()
        mock_coded_unit_2.to_dict.assert_called_once()

        # Capture the actual call arguments to json.dump
        args, kwargs = mock_json_dump.call_args

        expected_data = [
            {
                "unique_id": 1,
                "meaning_unit_string": "Parsed Meaning Unit 1",
                "assigned_code_list": [{"code_name": "Code 1", "code_justification": "Justification 1"}],
                "metadata": {"id": 1, "length_of_time_spoken_seconds": 120.5, "speaker_name": "Participant A"}
            },
            {
                "unique_id": 2,
                "meaning_unit_string": "Parsed Meaning Unit 2",
                "assigned_code_list": [{"code_name": "Code 2", "code_justification": "Justification 2"}],
                "metadata": {"id": 2, "length_of_time_spoken_seconds": 95.0, "speaker_name": "Participant B"}
            }
        ]

        # Assertions on the data passed to json.dump
        assert args[0] == expected_data, "json.dump was called with incorrect data."
        assert kwargs['indent'] == 2, "json.dump was called with incorrect indentation."
        assert isinstance(args[1], MagicMock), "json.dump was not called with a mocked file handle."

        # Alternatively, use ANY to ignore the specific file handle
        mock_json_dump.assert_called_once_with(
            expected_data,
            ANY,  # Use ANY to match any file handle
            indent=2
        )

@pytest.mark.parametrize(
    "use_parsing,use_rag",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
)
def test_main_execution_inductive(use_parsing, use_rag, mock_files):
    """
    Parameterized test for main.main(config) in inductive coding mode with various configurations.

    Parameters:
        use_parsing (bool): Whether to use parsing.
        use_rag (bool): Whether to use Retrieval-Augmented Generation.
        mock_files (fixture): Fixture to mock file I/O.
    """
    coding_mode = 'inductive'
    prompt_file = 'inductive_prompt.txt'

    # Define the configuration dictionary based on parameters
    config = {
        'coding_mode': coding_mode,
        'use_parsing': use_parsing,
        'use_rag': use_rag,
        'parse_model': 'test-parse-model',
        'assign_model': 'test-assign-model',
        'initialize_embedding_model': 'test-embedding-model',
        'retrieve_embedding_model': 'test-embedding-model',
        'data_format': 'transcript',  # Ensure this aligns with your data_format_config.json
        'paths': {
            'prompts_folder': 'prompts',
            'codebase_folder': 'codebase',
            'json_folder': 'json_data',
            'config_folder': 'configs'
        },
        'selected_codebase': 'test_codebase.jsonl',
        'selected_json_file': 'test_data.json',
        'parse_prompt_file': 'parse_prompt.txt',
        'inductive_coding_prompt_file': 'inductive_prompt.txt',
        'deductive_coding_prompt_file': 'deductive_prompt.txt',
        'output_folder': 'outputs',
        'output_format': 'json',
        'output_file_name': 'output',
        'enable_logging': False
    }

    # No initialize_inductive_resources to patch since it doesn't exist
    # Patch dependencies accordingly
    with patch('main.load_environment_variables'), \
         patch('main.load_parse_instructions', return_value='Parse instructions'), \
         patch('main.create_dynamic_model_for_format') as mock_create_model, \
         patch('main.FlexibleDataHandler') as mock_data_handler_class, \
         patch('main.assign_codes_to_meaning_units') as mock_assign_codes, \
         patch('main.os.makedirs'), \
         patch('main.json.dump') as mock_json_dump, \
         patch('main.os.path.exists') as mock_exists:

        # Configure the mock for os.path.exists
        def exists_side_effect(path):
            # Define which files exist
            existing_files = [
                'json_data/test_data.json',
                'configs/data_format_config.json',
                'codebase/test_codebase.jsonl',
                'prompts/parse_prompt.txt',
                'prompts/inductive_prompt.txt',
                'prompts/deductive_prompt.txt'
            ]
            return path in existing_files

        mock_exists.side_effect = exists_side_effect

        # Mock the dynamic model creation
        mock_model = MagicMock()
        mock_create_model.return_value = (mock_model, 'text_context')  # Updated content_field to 'text_context'

        # Mock the FlexibleDataHandler
        mock_data_handler = MagicMock()
        # Mock the load_data method to return validated data
        mock_data_handler.load_data.return_value = [
            {"id": 1, "length_of_time_spoken_seconds": 120.5, "text_context": "This is a sample transcript from Participant A.", "speaker_name": "Participant A"},
            {"id": 2, "length_of_time_spoken_seconds": 95.0, "text_context": "This is another sample transcript from Participant B.", "speaker_name": "Participant B"}
        ]
        # Mock the transform_data method to return MeaningUnit objects
        mock_meaning_unit_1 = MagicMock()
        mock_meaning_unit_1.unique_id = 1
        mock_meaning_unit_1.meaning_unit_string = "Parsed Meaning Unit 1"
        mock_meaning_unit_1.metadata = {"id": 1, "length_of_time_spoken_seconds": 120.5, "speaker_name": "Participant A"}

        mock_meaning_unit_2 = MagicMock()
        mock_meaning_unit_2.unique_id = 2
        mock_meaning_unit_2.meaning_unit_string = "Parsed Meaning Unit 2"
        mock_meaning_unit_2.metadata = {"id": 2, "length_of_time_spoken_seconds": 95.0, "speaker_name": "Participant B"}

        mock_data_handler.transform_data.return_value = [mock_meaning_unit_1, mock_meaning_unit_2]
        mock_data_handler_class.return_value = mock_data_handler

        # Mock assign_codes_to_meaning_units
        mock_coded_unit_1 = MagicMock()
        mock_coded_unit_1.to_dict.return_value = {
            "unique_id": 1,
            "meaning_unit_string": "Parsed Meaning Unit 1",
            "assigned_code_list": [{"code_name": "Code 1", "code_justification": "Justification 1"}],
            "metadata": {"id": 1, "length_of_time_spoken_seconds": 120.5, "speaker_name": "Participant A"}
        }

        mock_coded_unit_2 = MagicMock()
        mock_coded_unit_2.to_dict.return_value = {
            "unique_id": 2,
            "meaning_unit_string": "Parsed Meaning Unit 2",
            "assigned_code_list": [{"code_name": "Code 2", "code_justification": "Justification 2"}],
            "metadata": {"id": 2, "length_of_time_spoken_seconds": 95.0, "speaker_name": "Participant B"}
        }

        mock_assign_codes.return_value = [mock_coded_unit_1, mock_coded_unit_2]

        # Run the main function with the mocked config
        main.main(config)

        # Assert that assign_codes_to_meaning_units was called once
        mock_assign_codes.assert_called_once()

        # Assertions to ensure data loading and transformation
        mock_data_handler.load_data.assert_called_once()
        mock_data_handler.transform_data.assert_called_once()

        # Ensure that the output was attempted to be written
        mock_coded_unit_1.to_dict.assert_called_once()
        mock_coded_unit_2.to_dict.assert_called_once()

        # Capture the actual call arguments to json.dump
        args, kwargs = mock_json_dump.call_args

        expected_data = [
            {
                "unique_id": 1,
                "meaning_unit_string": "Parsed Meaning Unit 1",
                "assigned_code_list": [{"code_name": "Code 1", "code_justification": "Justification 1"}],
                "metadata": {"id": 1, "length_of_time_spoken_seconds": 120.5, "speaker_name": "Participant A"}
            },
            {
                "unique_id": 2,
                "meaning_unit_string": "Parsed Meaning Unit 2",
                "assigned_code_list": [{"code_name": "Code 2", "code_justification": "Justification 2"}],
                "metadata": {"id": 2, "length_of_time_spoken_seconds": 95.0, "speaker_name": "Participant B"}
            }
        ]

        # Assertions on the data passed to json.dump
        assert args[0] == expected_data, "json.dump was called with incorrect data."
        assert kwargs['indent'] == 2, "json.dump was called with incorrect indentation."
        assert isinstance(args[1], MagicMock), "json.dump was not called with a mocked file handle."

        # Alternatively, use ANY to ignore the specific file handle
        mock_json_dump.assert_called_once_with(
            expected_data,
            ANY,  # Use ANY to match any file handle
            indent=2
        )

@pytest.mark.parametrize(
    "use_parsing,use_rag",
    [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]
)
def test_main_execution_inductive(use_parsing, use_rag, mock_files):
    """
    Parameterized test for main.main(config) in inductive coding mode with various configurations.

    Parameters:
        use_parsing (bool): Whether to use parsing.
        use_rag (bool): Whether to use Retrieval-Augmented Generation.
        mock_files (fixture): Fixture to mock file I/O.
    """
    coding_mode = 'inductive'
    prompt_file = 'inductive_prompt.txt'

    # Define the configuration dictionary based on parameters
    config = {
        'coding_mode': coding_mode,
        'use_parsing': use_parsing,
        'use_rag': use_rag,
        'parse_model': 'test-parse-model',
        'assign_model': 'test-assign-model',
        'initialize_embedding_model': 'test-embedding-model',
        'retrieve_embedding_model': 'test-embedding-model',
        'data_format': 'transcript',  # Ensure this aligns with your data_format_config.json
        'paths': {
            'prompts_folder': 'prompts',
            'codebase_folder': 'codebase',
            'json_folder': 'json_data',
            'config_folder': 'configs'
        },
        'selected_codebase': 'test_codebase.jsonl',
        'selected_json_file': 'test_data.json',
        'parse_prompt_file': 'parse_prompt.txt',
        'inductive_coding_prompt_file': 'inductive_prompt.txt',
        'deductive_coding_prompt_file': 'deductive_prompt.txt',
        'output_folder': 'outputs',
        'output_format': 'json',
        'output_file_name': 'output',
        'enable_logging': False
    }

    # No initialize_inductive_resources to patch since it doesn't exist
    # Patch dependencies accordingly
    with patch('main.load_environment_variables'), \
         patch('main.load_parse_instructions', return_value='Parse instructions'), \
         patch('main.create_dynamic_model_for_format') as mock_create_model, \
         patch('main.FlexibleDataHandler') as mock_data_handler_class, \
         patch('main.assign_codes_to_meaning_units') as mock_assign_codes, \
         patch('main.os.makedirs'), \
         patch('main.json.dump') as mock_json_dump, \
         patch('main.os.path.exists') as mock_exists:

        # Configure the mock for os.path.exists
        def exists_side_effect(path):
            # Define which files exist
            existing_files = [
                'json_data/test_data.json',
                'configs/data_format_config.json',
                'codebase/test_codebase.jsonl',
                'prompts/parse_prompt.txt',
                'prompts/inductive_prompt.txt',
                'prompts/deductive_prompt.txt'
            ]
            return path in existing_files

        mock_exists.side_effect = exists_side_effect

        # Mock the dynamic model creation
        mock_model = MagicMock()
        mock_create_model.return_value = (mock_model, 'text_context')  # Updated content_field to 'text_context'

        # Mock the FlexibleDataHandler
        mock_data_handler = MagicMock()
        # Mock the load_data method to return validated data
        mock_data_handler.load_data.return_value = [
            {"id": 1, "length_of_time_spoken_seconds": 120.5, "text_context": "This is a sample transcript from Participant A.", "speaker_name": "Participant A"},
            {"id": 2, "length_of_time_spoken_seconds": 95.0, "text_context": "This is another sample transcript from Participant B.", "speaker_name": "Participant B"}
        ]
        # Mock the transform_data method to return MeaningUnit objects
        mock_meaning_unit_1 = MagicMock()
        mock_meaning_unit_1.unique_id = 1
        mock_meaning_unit_1.meaning_unit_string = "Parsed Meaning Unit 1"
        mock_meaning_unit_1.metadata = {"id": 1, "length_of_time_spoken_seconds": 120.5, "speaker_name": "Participant A"}

        mock_meaning_unit_2 = MagicMock()
        mock_meaning_unit_2.unique_id = 2
        mock_meaning_unit_2.meaning_unit_string = "Parsed Meaning Unit 2"
        mock_meaning_unit_2.metadata = {"id": 2, "length_of_time_spoken_seconds": 95.0, "speaker_name": "Participant B"}

        mock_data_handler.transform_data.return_value = [mock_meaning_unit_1, mock_meaning_unit_2]
        mock_data_handler_class.return_value = mock_data_handler

        # Mock assign_codes_to_meaning_units
        mock_coded_unit_1 = MagicMock()
        mock_coded_unit_1.to_dict.return_value = {
            "unique_id": 1,
            "meaning_unit_string": "Parsed Meaning Unit 1",
            "assigned_code_list": [{"code_name": "Code 1", "code_justification": "Justification 1"}],
            "metadata": {"id": 1, "length_of_time_spoken_seconds": 120.5, "speaker_name": "Participant A"}
        }

        mock_coded_unit_2 = MagicMock()
        mock_coded_unit_2.to_dict.return_value = {
            "unique_id": 2,
            "meaning_unit_string": "Parsed Meaning Unit 2",
            "assigned_code_list": [{"code_name": "Code 2", "code_justification": "Justification 2"}],
            "metadata": {"id": 2, "length_of_time_spoken_seconds": 95.0, "speaker_name": "Participant B"}
        }

        mock_assign_codes.return_value = [mock_coded_unit_1, mock_coded_unit_2]

        # Run the main function with the mocked config
        main.main(config)

        # Assert that assign_codes_to_meaning_units was called once
        mock_assign_codes.assert_called_once()

        # Assertions to ensure data loading and transformation
        mock_data_handler.load_data.assert_called_once()
        mock_data_handler.transform_data.assert_called_once()

        # Ensure that the output was attempted to be written
        mock_coded_unit_1.to_dict.assert_called_once()
        mock_coded_unit_2.to_dict.assert_called_once()

        # Capture the actual call arguments to json.dump
        args, kwargs = mock_json_dump.call_args

        expected_data = [
            {
                "unique_id": 1,
                "meaning_unit_string": "Parsed Meaning Unit 1",
                "assigned_code_list": [{"code_name": "Code 1", "code_justification": "Justification 1"}],
                "metadata": {"id": 1, "length_of_time_spoken_seconds": 120.5, "speaker_name": "Participant A"}
            },
            {
                "unique_id": 2,
                "meaning_unit_string": "Parsed Meaning Unit 2",
                "assigned_code_list": [{"code_name": "Code 2", "code_justification": "Justification 2"}],
                "metadata": {"id": 2, "length_of_time_spoken_seconds": 95.0, "speaker_name": "Participant B"}
            }
        ]

        # Assertions on the data passed to json.dump
        assert args[0] == expected_data, "json.dump was called with incorrect data."
        assert kwargs['indent'] == 2, "json.dump was called with incorrect indentation."
        assert isinstance(args[1], MagicMock), "json.dump was not called with a mocked file handle."

        # Alternatively, use ANY to ignore the specific file handle
        mock_json_dump.assert_called_once_with(
            expected_data,
            ANY,  # Use ANY to match any file handle
            indent=2
        )
