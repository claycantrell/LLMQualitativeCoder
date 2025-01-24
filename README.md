## 1. Overview
LLMQualitativeCoder is a tool for automated qualitative coding using Large Language Models (LLMs). The workflow includes:

- **Flexible Data Handling:** Supports diverse JSON input data formats with customizable fields and basic filtering.
- **Automated Parsing:** Breaks down large texts into smaller meaning units for coding based on specifications in LLM user prompt.
- **Deductive and Inductive Coding:** Offers both predefined (deductive) and emergent (inductive) coding approaches.
- **Customizable Configuration:** Config files allow for customization of batch size, context size, and LLM provider/model for coding taks.

## 2. Installation & Setup
### Using Poetry
LLMQualitativeCoder uses Poetry for dependency management and packaging.

### Prerequisites
- **Python 3.8+:** Ensure Python is installed.
- **Poetry:** Install Poetry if not already available:
  ```
  curl -sSL https://install.python-poetry.org | python3 -
  ```
  Verify the installation:
  ```
  poetry --version
  ```

### Installation Steps
1. **Clone the Repository:**
   ```
   git clone https://github.com/iggygraceful/LLMQualitativeCoder.git
   cd LLMQualitativeCoder
   ```

2. **Install Dependencies:**
   ```
   poetry install
   ```

3. **Activate the Virtual Environment:**
   ```
   poetry shell
   ```

4. **Set Environment Variables:**
   Configure API keys before running the pipeline:
   - On Linux/macOS:
     ```
     export OPENAI_API_KEY='your-openai-api-key'
     export HUGGINGFACE_API_KEY='your-huggingface-api-key'
     ```
   - On Windows CMD:
     ```
     set OPENAI_API_KEY=your-openai-api-key
     set HUGGINGFACE_API_KEY=your-huggingface-api-key
     ```
   **Note:** I couldnt get the HuggingFace API to work correctly; only OpenAI models are supported at this time.


## 4. Key Components
The codebase includes several modules:

### `main.py`
**Purpose:** Coordinates the workflow from data loading to validation.
**Key Tasks:**
- Reads configuration files.
- Initializes the environment and logging.
- Loads and filters data.
- Optionally parses data into smaller meaning units.
- Assigns codes (deductive or inductive).
- Saves coded meaning units to an output JSON file.
- Runs validation and generates a report.

### `data_handlers.py`
**Purpose:** Manages data loading, transformation, and filtering based on configuration.
**Key Tasks:**
- Loads JSON files.
- Applies filter rules.
- Transforms data into `MeaningUnit` objects.

### `qual_functions.py`
**Purpose:** Includes core functionalities for parsing transcripts and assigning codes.
**Key Tasks:**
- Interfaces with LLMs for parsing and coding.

### `utils.py`
**Purpose:** Provides helper functions for environment setup and resource initialization.
**Key Tasks:**
- Loads JSON and text files.
- Manages environment variables.
- Generates structured LLM responses.

### `validator.py`
**Purpose:** Ensures output consistency and completeness through validation reports.
**Key Tasks:**
- Compares original segments with meaning units.
- Identifies skipped or inconsistent segments.
- Generates JSON validation reports.

### `logging_config.py`
**Purpose:** Centralizes logging setup.
**Key Tasks:**
- Configures log levels (DEBUG, INFO, etc.).
- Manages console and file outputs.

### `api.py` (In Development)
**Purpose:** Provides a FastAPI server for asynchronous pipeline execution.
**Key Endpoints:**
- `POST /run-pipeline`
- `GET /status/{job_id}`
- `GET /output/{job_id}`
- `GET /reports/{job_id}/{report_name}`

## 5. Configuration
### Pipeline Configuration (`config.json`)
Defines the pipeline behavior, including coding modes, model selections, paths, and logging settings.

**Example:**
```json
{
  "coding_mode": "deductive",
  "use_parsing": true,
  "preliminary_segments_per_prompt": 5,
  "meaning_units_per_assignment_prompt": 10,
  "context_size": 5,
  "data_format": "transcript",
  "paths": {
    "prompts_folder": "transcriptanalysis/prompts",
    "codebase_folder": "transcriptanalysis/codebases",
    "json_folder": "transcriptanalysis/json_inputs",
    "config_folder": "transcriptanalysis/configs"
  },
  "selected_codebase": "default_codebase.json",
  "selected_json_file": "teacher_transcript.json",
  "parse_prompt_file": "parse_prompt.txt",
  "inductive_coding_prompt_file": "inductive_prompt.txt",
  "deductive_coding_prompt_file": "deductive_prompt.txt",
  "output_folder": "outputs",
  "enable_logging": true,
  "logging_level": "INFO",
  "log_to_file": true,
  "log_file_path": "logs/application.log",
  "thread_count": 4,
  "parse_llm_config": {
    "provider": "openai",
    "model_name": "gpt-4",
    "temperature": 0.7,
    "max_tokens": 2000,
    "api_key": "YOUR_OPENAI_API_KEY"
  },
  "assign_llm_config": {
    "provider": "huggingface",
    "model_name": "some-hf-model",
    "temperature": 0.6,
    "max_tokens": 1500,
    "api_key": "YOUR_HUGGINGFACE_API_KEY"
  }
}
```

### Data Format Configuration (`data_format_config.json`)
Specifies your JSON input file, including fields for content, speaker, source IDs, and filtering rules.

Context fields are input fields that you want to include during the LLM coding task.

**Example:**
```json
{
  "transcript": {
    "content_field": "text",
    "context_fields": ["speaker", "timestamp"],
    "list_field": "dialogues",
    "filter_rules": []
  }
}
```

## 6. Running the Pipeline (CLI)
### Setting Up Environment Variables
Set the necessary API keys:
- On Linux/macOS:
  ```
  export OPENAI_API_KEY='your-openai-api-key'
  export HUGGINGFACE_API_KEY='your-huggingface-api-key'
  ```
- On Windows CMD:
  ```
  set OPENAI_API_KEY=your-openai-api-key
  set HUGGINGFACE_API_KEY=your-huggingface-api-key
  ```

**Note:** Hugging Face API is currently unavailable.

### Execute the Pipeline
Run the main script:
```sh
cd LLMQualitativeCoder
python main.py
```

## 8. Validation Process
Validation ensures the final meaning units accurately represent the original segments. Discrepancies are reported in `validation_report.json`.

## 9. Logging
Logs capture pipeline operations and are saved in the specified `log_file_path`.

## 10. Input and Output
### Input
- **Preliminary Segments:** JSON files containing raw data.
- **Codebase Files:** JSONL files for deductive coding.
- **Prompts:** Text files with LLM instructions.

### Output
- **Coded JSON Files:** Contain meaning units with assigned codes.
- **Validation Reports:** Detail discrepancies between input and output.
- **Logs:** Available in the console and specified files.

