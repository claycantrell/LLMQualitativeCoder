# config_schemas.py

from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, field_validator, model_validator, RootModel

# ---------------------------
# Enums
# ---------------------------

class OperatorEnum(str, Enum):
    EQUALS = "equals"
    NOT_EQUALS = "not_equals"
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    CONTAINS = "contains"
    # Add other operators as needed

class CodingModeEnum(str, Enum):
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"

class LoggingLevelEnum(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# ---------------------------
# New Enums & Models for LLM Configuration
# ---------------------------

class ProviderEnum(str, Enum):
    OPENAI = "openai"
    HUGGINGFACE = "huggingface"

class LLMConfig(BaseModel):
    provider: ProviderEnum
    model_name: str
    temperature: float = 0.7
    max_tokens: int = 2000
    api_key: Optional[str] = None

# ---------------------------
# Data Format Config Models
# ---------------------------

class FilterRule(BaseModel):
    field: str
    operator: OperatorEnum
    value: str

class DataFormatConfigItem(BaseModel):
    """
    Replaces the old 'speaker_field' with 'context_fields', allowing multiple fields
    from the JSON to be injected as context per preliminary segment.
    """
    content_field: str
    context_fields: Optional[List[str]] = None  # CHANGED
    list_field: Optional[str] = None
    source_id_field: Optional[str] = None
    filter_rules: List[FilterRule] = []

    @model_validator(mode='after')
    def check_required_fields(cls, values):
        """
        We only keep a check for 'movie_script' requiring 'list_field'.
        The old requirement for speaker_field is removed.
        """
        if values.content_field == 'movie_script' and not values.list_field:
            raise ValueError("list_field is required for movie_script data_format")
        return values

class DataFormatConfig(RootModel[Dict[str, DataFormatConfigItem]]):
    """
    A RootModel where each top-level key (e.g. "transcript", "movie_script")
    maps to a DataFormatConfigItem.
    """

    def __getitem__(self, item: str) -> DataFormatConfigItem:
        return self.root[item]

    def __contains__(self, item: str) -> bool:
        return item in self.root

# ---------------------------
# Main Config Models
# ---------------------------

class PathsModel(BaseModel):
    prompts_folder: str
    codebase_folder: str
    json_folder: str
    config_folder: str

class ConfigModel(BaseModel):
    coding_mode: CodingModeEnum
    use_parsing: bool
    preliminary_segments_per_prompt: int  # Renamed from speaking_turns_per_prompt
    meaning_units_per_assignment_prompt: int
    context_size: int
    data_format: str
    paths: PathsModel
    selected_codebase: str
    selected_json_file: str
    parse_prompt_file: str
    inductive_coding_prompt_file: str
    deductive_coding_prompt_file: str
    output_folder: str
    enable_logging: bool
    logging_level: LoggingLevelEnum
    log_to_file: bool
    log_file_path: str

    # NEW FIELD: specify how many threads (concurrent requests) to use
    thread_count: int = 1

    # NEW FIELDS: Separate LLM configurations for parse and assign tasks
    parse_llm_config: LLMConfig
    assign_llm_config: LLMConfig

    @field_validator('data_format')
    def validate_data_format(cls, v):
        allowed_formats = ['transcript', 'movie_script', 'other_format']  # Update as needed
        if v not in allowed_formats:
            raise ValueError(f"'data_format' must be one of {allowed_formats}, got '{v}'")
        return v

# Example Usage
if __name__ == "__main__":
    try:
        config = ConfigModel(
            coding_mode="deductive",
            use_parsing=True,
            preliminary_segments_per_prompt=5,  # Updated field name
            meaning_units_per_assignment_prompt=10,
            context_size=2048,
            data_format="transcript",
            paths={
                "prompts_folder": "/path/to/prompts",
                "codebase_folder": "/path/to/codebase",
                "json_folder": "/path/to/json",
                "config_folder": "/path/to/config"
            },
            selected_codebase="default",
            selected_json_file="data.json",
            parse_prompt_file="parse_prompt.txt",
            inductive_coding_prompt_file="inductive_prompt.txt",
            deductive_coding_prompt_file="deductive_prompt.txt",
            output_folder="/path/to/output",
            enable_logging=True,
            logging_level="INFO",
            log_to_file=True,
            log_file_path="/path/to/logfile.log",
            thread_count=4,  # Example: 4 concurrent requests
            parse_llm_config={
                "provider": "openai",
                "model_name": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2000,
                "api_key": "YOUR_OPENAI_API_KEY_FOR_PARSE"
            },
            assign_llm_config={
                "provider": "huggingface",
                "model_name": "gpt2",
                "temperature": 0.6,
                "max_tokens": 1500,
                "api_key": "YOUR_HUGGINGFACE_API_KEY_IF_NEEDED"
            }
        )
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading configuration: {e}")
