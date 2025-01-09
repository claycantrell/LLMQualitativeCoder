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
# Data Format Config Models
# ---------------------------

class FilterRule(BaseModel):
    field: str
    operator: OperatorEnum
    value: str

class DataFormatConfigItem(BaseModel):
    content_field: str
    speaker_field: Optional[str] = None
    list_field: Optional[str] = None
    source_id_field: Optional[str] = None
    filter_rules: List[FilterRule] = []

    @model_validator(mode='after')
    def check_required_fields(cls, values):
        data_format = values.content_field  # Access attributes directly
        if data_format == 'transcript' and not values.speaker_field:
            raise ValueError("speaker_field is required for transcript data_format")
        if data_format == 'movie_script' and not values.list_field:
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
    speaking_turns_per_prompt: int
    meaning_units_per_assignment_prompt: int
    context_size: int
    parse_model: str
    assign_model: str
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
            speaking_turns_per_prompt=5,
            meaning_units_per_assignment_prompt=10,
            context_size=2048,
            parse_model="parse-model-v1",
            assign_model="assign-model-v1",
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
            thread_count=4  # Example: 4 concurrent requests
        )
        print("Configuration loaded successfully.")
    except Exception as e:
        print(f"Error loading configuration: {e}")
