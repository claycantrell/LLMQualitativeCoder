# tests/test_data_handlers.py
import os
import json
import tempfile
import pytest
from unittest.mock import patch, MagicMock
from pydantic import BaseModel
from TranscriptAnalysis.src.TranscriptAnalysis.qual_functions import parse_transcript, MeaningUnit
from TranscriptAnalysis.src.TranscriptAnalysis.data_handlers import FlexibleDataHandler  # Ensure correct import

class SampleModel(BaseModel):  # Renamed from TestModel
    text: str
    speaker: str

def test_load_data_success():
    sample_data = [
        {'text': 'Sample text 1', 'speaker': 'Speaker 1'},
        {'text': 'Sample text 2', 'speaker': 'Speaker 2'}
    ]
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
        json.dump(sample_data, tmp_file)
        tmp_file_path = tmp_file.name
    try:
        handler = FlexibleDataHandler(
            file_path=tmp_file_path,
            parse_instructions="Parse instructions",
            completion_model="test-model",
            model_class=SampleModel,  # Updated class name
            content_field='text',
            use_parsing=False
        )
        validated_data = handler.load_data()
        assert len(validated_data) == 2
        assert validated_data[0]['text'] == 'Sample text 1'
    finally:
        os.remove(tmp_file_path)

def test_load_data_validation_failure():
    sample_data = [
        {'text': 'Sample text 1', 'speaker': 'Speaker 1'},
        {'text': 'Sample text 2'}  # Missing 'speaker' field
    ]
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
        json.dump(sample_data, tmp_file)
        tmp_file_path = tmp_file.name
    try:
        handler = FlexibleDataHandler(
            file_path=tmp_file_path,
            parse_instructions="Parse instructions",
            completion_model="test-model",
            model_class=SampleModel,  # Updated class name
            content_field='text',
            use_parsing=False
        )
        validated_data = handler.load_data()
        assert len(validated_data) == 1  # Only one record should be valid
    finally:
        os.remove(tmp_file_path)

def test_transform_data_no_parsing():
    validated_data = [
        {'text': 'Sample text 1', 'speaker': 'Speaker 1'}
    ]
    handler = FlexibleDataHandler(
        file_path='dummy_path',
        parse_instructions="Parse instructions",
        completion_model="test-model",
        model_class=SampleModel,  # Updated class name
        content_field='text',
        use_parsing=False
    )
    meaning_units = handler.transform_data(validated_data)
    assert len(meaning_units) == 1
    assert meaning_units[0].meaning_unit_string == 'Sample text 1'
    assert meaning_units[0].metadata['speaker'] == 'Speaker 1'

def test_transform_data_with_parsing():
    validated_data = [
        {'text': 'Sample text 1', 'speaker': 'Speaker 1'}
    ]
    handler = FlexibleDataHandler(
        file_path='dummy_path',
        parse_instructions="Parse instructions",
        completion_model="test-model",
        model_class=SampleModel,  # Updated class name
        content_field='text',
        use_parsing=True
    )
    with patch('data_handlers.parse_transcript') as mock_parse_transcript:
        mock_parse_transcript.return_value = ['Parsed unit 1', 'Parsed unit 2']
        meaning_units = handler.transform_data(validated_data)
        assert len(meaning_units) == 2
        assert meaning_units[0].meaning_unit_string == 'Parsed unit 1'
        assert meaning_units[1].meaning_unit_string == 'Parsed unit 2'
