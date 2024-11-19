# tests/test_qual_functions.py
import os
import json
import tempfile
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from qual_functions import (
    parse_transcript,
    initialize_faiss_index_from_formatted_file,
    retrieve_relevant_codes,
    assign_codes_to_meaning_units,
    MeaningUnit,
    CodeAssigned
)

def test_parse_transcript_success():
    speaking_turn_string = "This is a test speaking turn."
    prompt = "Please parse the following speaking turn."
    completion_model = "test-model"
    metadata = {"speaker": "Test Speaker"}

    # Mock the OpenAI client
    with patch('qual_functions.client') as mock_client:
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed.meaning_unit_string_list = ["Meaning unit 1", "Meaning unit 2"]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        result = parse_transcript(speaking_turn_string, prompt, completion_model, metadata)
        assert result == ["Meaning unit 1", "Meaning unit 2"]
        mock_client.beta.chat.completions.parse.assert_called_once()

def test_parse_transcript_empty_response():
    speaking_turn_string = "This is a test speaking turn."
    prompt = "Please parse the following speaking turn."
    completion_model = "test-model"
    metadata = {"speaker": "Test Speaker"}

    with patch('qual_functions.client') as mock_client:
        mock_response = MagicMock()
        mock_response.choices = []
        mock_client.beta.chat.completions.parse.return_value = mock_response

        result = parse_transcript(speaking_turn_string, prompt, completion_model, metadata)
        assert result == []
        mock_client.beta.chat.completions.parse.assert_called_once()

def test_parse_transcript_exception():
    speaking_turn_string = "This is a test speaking turn."
    prompt = "Please parse the following speaking turn."
    completion_model = "test-model"
    metadata = {"speaker": "Test Speaker"}

    with patch('qual_functions.client') as mock_client:
        mock_client.beta.chat.completions.parse.side_effect = Exception("API Error")

        result = parse_transcript(speaking_turn_string, prompt, completion_model, metadata)
        assert result == []
        mock_client.beta.chat.completions.parse.assert_called_once()

def test_initialize_faiss_index_from_formatted_file_success():
    codes_list_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    sample_data = [
        {"text": "Code 1", "metadata": {"id": 1}},
        {"text": "Code 2", "metadata": {"id": 2}}
    ]
    for item in sample_data:
        codes_list_file.write(json.dumps(item) + '\n')
    codes_list_file.close()
    embedding_model = "test-embedding-model"

    with patch('qual_functions.client') as mock_client:
        # Mock the embeddings response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3]) for _ in sample_data]
        mock_client.embeddings.create.return_value = mock_response

        index, processed_codes = initialize_faiss_index_from_formatted_file(codes_list_file.name, embedding_model)

        # Check that index is created and processed_codes is correct
        assert len(processed_codes) == len(sample_data)
        assert index.ntotal == len(sample_data)

    os.remove(codes_list_file.name)

def test_initialize_faiss_index_from_formatted_file_empty_file():
    codes_list_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    codes_list_file.close()
    embedding_model = "test-embedding-model"

    with pytest.raises(ValueError) as exc_info:
        initialize_faiss_index_from_formatted_file(codes_list_file.name, embedding_model)
    assert "No valid embeddings found" in str(exc_info.value)

    os.remove(codes_list_file.name)

def test_retrieve_relevant_codes_success():
    meaning_unit = MeaningUnit(unique_id=1, meaning_unit_string="Test meaning unit", metadata={})
    index = MagicMock()
    processed_codes = [{'text': 'Code 1', 'metadata': {}}, {'text': 'Code 2', 'metadata': {}}]
    top_k = 2
    embedding_model = "test-embedding-model"

    with patch('qual_functions.client') as mock_client:
        # Mock the embeddings response
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        mock_client.embeddings.create.return_value = mock_response

        # Mock the index search
        index.search.return_value = (np.array([[0.1, 0.2]]), np.array([[0, 1]]))

        relevant_codes = retrieve_relevant_codes(meaning_unit, index, processed_codes, top_k, embedding_model)
        assert len(relevant_codes) == 2
        assert relevant_codes[0]['text'] == 'Code 1'
        assert relevant_codes[1]['text'] == 'Code 2'

def test_assign_codes_to_meaning_units_deductive_rag():
    meaning_unit_list = [
        MeaningUnit(unique_id=1, meaning_unit_string="Meaning unit 1", metadata={})
    ]
    coding_instructions = "Apply codes based on the following."
    processed_codes = [{'text': 'Code 1', 'metadata': {}}, {'text': 'Code 2', 'metadata': {}}]
    index = MagicMock()
    embedding_model = "test-embedding-model"
    assign_model = "test-completion-model"

    with patch('qual_functions.retrieve_relevant_codes') as mock_retrieve_codes, \
         patch('qual_functions.client') as mock_client:
        mock_retrieve_codes.return_value = processed_codes

        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed.codeList = [
            CodeAssigned(code_name="Code 1", code_justification="Justification 1")
        ]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        result = assign_codes_to_meaning_units(
            meaning_unit_list,
            coding_instructions,
            processed_codes=processed_codes,
            index=index,
            top_k=2,
            context_size=0,
            use_rag=True,
            codebase=None,
            completion_model=assign_model,
            embedding_model=embedding_model
        )
        assert len(result[0].assigned_code_list) == 1
        assert result[0].assigned_code_list[0].code_name == "Code 1"

def test_assign_codes_to_meaning_units_inductive():
    meaning_unit_list = [
        MeaningUnit(unique_id=1, meaning_unit_string="Meaning unit 1", metadata={})
    ]
    coding_instructions = "Generate codes based on the following guidelines."
    assign_model = "test-completion-model"

    with patch('qual_functions.client') as mock_client:
        # Mock the LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.parsed.codeList = [
            CodeAssigned(code_name="Generated Code", code_justification="Generated Justification")
        ]
        mock_client.beta.chat.completions.parse.return_value = mock_response

        result = assign_codes_to_meaning_units(
            meaning_unit_list,
            coding_instructions,
            processed_codes=None,
            index=None,
            top_k=None,
            context_size=0,
            use_rag=False,
            codebase=None,
            completion_model=assign_model,
            embedding_model=None
        )
        assert len(result[0].assigned_code_list) == 1
        assert result[0].assigned_code_list[0].code_name == "Generated Code"
