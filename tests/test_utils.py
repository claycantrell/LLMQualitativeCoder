# tests/test_utils.py
import os
import json
import tempfile
import pytest
from unittest.mock import patch
from utils import (
    load_environment_variables,
    load_config,
    load_coding_instructions,
    load_parse_instructions,
    load_inductive_coding_prompt,
    load_deductive_coding_prompt,
    initialize_deductive_resources,
    load_schema_config,
    create_dynamic_model_for_format
)

def test_load_environment_variables_success():
    with patch.dict(os.environ, {'OPENAI_API_KEY': 'test_key'}):
        # Should not raise an exception
        load_environment_variables()

def test_load_environment_variables_missing():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc_info:
            load_environment_variables()
        assert str(exc_info.value) == "Set the OPENAI_API_KEY environment variable."

def test_load_config_success():
    config_data = {'key': 'value'}
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
        json.dump(config_data, tmp_file)
        tmp_file_path = tmp_file.name
    try:
        config = load_config(tmp_file_path)
        assert config == config_data
    finally:
        os.remove(tmp_file_path)

def test_load_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_config('non_existent_file.json')

def test_load_config_invalid_json():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
        tmp_file.write('invalid json')
        tmp_file_path = tmp_file.name
    try:
        with pytest.raises(json.JSONDecodeError):
            load_config(tmp_file_path)
    finally:
        os.remove(tmp_file_path)

def test_load_coding_instructions_success():
    prompts_folder = tempfile.mkdtemp()
    prompt_file = 'test_prompt.txt'
    prompt_content = 'This is a test prompt.'
    prompt_file_path = os.path.join(prompts_folder, prompt_file)
    with open(prompt_file_path, 'w') as f:
        f.write(prompt_content)
    try:
        coding_instructions = load_coding_instructions(prompts_folder, prompt_file)
        assert coding_instructions == prompt_content
    finally:
        os.remove(prompt_file_path)
        os.rmdir(prompts_folder)

def test_load_coding_instructions_file_not_found():
    prompts_folder = tempfile.mkdtemp()
    try:
        with pytest.raises(FileNotFoundError):
            load_coding_instructions(prompts_folder, 'non_existent_file.txt')
    finally:
        os.rmdir(prompts_folder)

def test_load_coding_instructions_empty_file():
    prompts_folder = tempfile.mkdtemp()
    prompt_file = 'test_prompt.txt'
    prompt_file_path = os.path.join(prompts_folder, prompt_file)
    with open(prompt_file_path, 'w') as f:
        pass  # Create empty file
    try:
        with pytest.raises(ValueError) as exc_info:
            load_coding_instructions(prompts_folder, prompt_file)
        assert str(exc_info.value) == "Coding instructions file is empty."
    finally:
        os.remove(prompt_file_path)
        os.rmdir(prompts_folder)

def test_load_parse_instructions_success():
    prompts_folder = tempfile.mkdtemp()
    parse_prompt_file = 'test_parse_prompt.txt'
    parse_prompt_content = 'This is a test parse prompt.'
    parse_prompt_file_path = os.path.join(prompts_folder, parse_prompt_file)
    with open(parse_prompt_file_path, 'w') as f:
        f.write(parse_prompt_content)
    try:
        parse_instructions = load_parse_instructions(prompts_folder, parse_prompt_file)
        assert parse_instructions == parse_prompt_content
    finally:
        os.remove(parse_prompt_file_path)
        os.rmdir(prompts_folder)

def test_load_parse_instructions_file_not_found():
    prompts_folder = tempfile.mkdtemp()
    try:
        with pytest.raises(FileNotFoundError):
            load_parse_instructions(prompts_folder, 'non_existent_file.txt')
    finally:
        os.rmdir(prompts_folder)

def test_load_parse_instructions_empty_file():
    prompts_folder = tempfile.mkdtemp()
    parse_prompt_file = 'test_parse_prompt.txt'
    parse_prompt_file_path = os.path.join(prompts_folder, parse_prompt_file)
    with open(parse_prompt_file_path, 'w') as f:
        pass  # Create empty file
    try:
        with pytest.raises(ValueError) as exc_info:
            load_parse_instructions(prompts_folder, parse_prompt_file)
        assert str(exc_info.value) == "Parse instructions file is empty."
    finally:
        os.remove(parse_prompt_file_path)
        os.rmdir(prompts_folder)

def test_load_inductive_coding_prompt_success():
    prompts_folder = tempfile.mkdtemp()
    inductive_prompt_file = 'test_inductive_prompt.txt'
    inductive_prompt_content = 'This is a test inductive coding prompt.'
    inductive_prompt_file_path = os.path.join(prompts_folder, inductive_prompt_file)
    with open(inductive_prompt_file_path, 'w') as f:
        f.write(inductive_prompt_content)
    try:
        inductive_prompt = load_inductive_coding_prompt(prompts_folder, inductive_prompt_file)
        assert inductive_prompt == inductive_prompt_content
    finally:
        os.remove(inductive_prompt_file_path)
        os.rmdir(prompts_folder)

def test_load_inductive_coding_prompt_file_not_found():
    prompts_folder = tempfile.mkdtemp()
    try:
        with pytest.raises(FileNotFoundError):
            load_inductive_coding_prompt(prompts_folder, 'non_existent_file.txt')
    finally:
        os.rmdir(prompts_folder)

def test_load_inductive_coding_prompt_empty_file():
    prompts_folder = tempfile.mkdtemp()
    inductive_prompt_file = 'test_inductive_prompt.txt'
    inductive_prompt_file_path = os.path.join(prompts_folder, inductive_prompt_file)
    with open(inductive_prompt_file_path, 'w') as f:
        pass  # Create empty file
    try:
        with pytest.raises(ValueError) as exc_info:
            load_inductive_coding_prompt(prompts_folder, inductive_prompt_file)
        assert str(exc_info.value) == "Inductive coding prompt file is empty."
    finally:
        os.remove(inductive_prompt_file_path)
        os.rmdir(prompts_folder)

def test_load_schema_config_success():
    schema_data = {'interview': {'fields': {'text': 'str'}, 'content_field': 'text'}}
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
        json.dump(schema_data, tmp_file)
        tmp_file_path = tmp_file.name
    try:
        config = load_schema_config(tmp_file_path)
        assert config == schema_data
    finally:
        os.remove(tmp_file_path)

def test_load_schema_config_file_not_found():
    with pytest.raises(FileNotFoundError):
        load_schema_config('non_existent_schema.json')

def test_load_schema_config_invalid_json():
    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as tmp_file:
        tmp_file.write('invalid json')
        tmp_file_path = tmp_file.name
    try:
        with pytest.raises(json.JSONDecodeError):
            load_schema_config(tmp_file_path)
    finally:
        os.remove(tmp_file_path)

def test_create_dynamic_model_for_format_success():
    schema_config = {
        'interview': {
            'fields': {'text': 'str', 'speaker': 'str'},
            'content_field': 'text'
        }
    }
    dynamic_model, content_field = create_dynamic_model_for_format('interview', schema_config)
    assert content_field == 'text'
    # Instantiate the dynamic model to ensure it works
    instance = dynamic_model(text='Sample text', speaker='John Doe')
    assert instance.text == 'Sample text'
    assert instance.speaker == 'John Doe'

def test_create_dynamic_model_for_format_missing_format():
    schema_config = {}
    with pytest.raises(ValueError) as exc_info:
        create_dynamic_model_for_format('interview', schema_config)
    assert "No schema configuration found for data format 'interview'" in str(exc_info.value)

def test_create_dynamic_model_for_format_missing_fields():
    schema_config = {
        'interview': {
            'content_field': 'text'
        }
    }
    with pytest.raises(ValueError) as exc_info:
        create_dynamic_model_for_format('interview', schema_config)
    assert "must include 'fields' and 'content_field'" in str(exc_info.value)
