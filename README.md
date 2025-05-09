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
- **Python:** The project requires Python 3.10+, but can be modified to work with Python 3.9 by changing the `python` version in `pyproject.toml`.
- **Poetry:** Install Poetry using one of the following methods:
  
  **Method 1: Using curl (recommended):**
  ```
  curl -sSL https://install.python-poetry.org | python3 -
  ```
  
  **Method 2: Using pip (alternative):**
  ```
  python3 -m pip install --user poetry
  ```
  
  **Add Poetry to your PATH:** After installation, make sure Poetry's bin directory is in your PATH:
  ```
  # For curl installation
  export PATH=$PATH:$HOME/.local/bin
  
  # For pip installation
  export PATH=$PATH:$HOME/Library/Python/3.9/bin  # Adjust Python version as needed
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

2. **Optional: Adjust Python Version (if using Python 3.9):**
   Edit the `pyproject.toml` file to change:
   ```
   python = ">=3.10,<4.0"
   ```
   to
   ```
   python = ">=3.9,<4.0"
   ```

3. **Install Dependencies:**
   ```
   poetry lock    # Generate/update lock file if needed
   poetry install
   ```

4. **Activate the Virtual Environment:**
   ```
   # For Poetry 2.0+ (newer versions)
   poetry env activate
   
   # For older Poetry versions
   poetry shell
   ```
   
   The `poetry env activate` command will output the source command you need to run, for example:
   ```
   source /path/to/virtualenvs/myproject-py3.9/bin/activate
   ```

5. **Set Environment Variables:**
   Configure API keys for the pipeline using one of these methods:
   
   **Option 1: Using local config file (recommended for development):**
   
   A template file `src/transcriptanalysis/configs/local_config.template.json` is included in the repository. The application will automatically create a copy of this template as `local_config.json` when you first run it.
   
   To set up your credentials:
   
   1. Edit the `src/transcriptanalysis/configs/local_config.json` file with your API keys:
   ```json
   {
     "api_keys": {
       "openai_api_key": "your-actual-openai-api-key",
       "huggingface_api_key": "your-actual-huggingface-api-key" 
     }
   }
   ```
   
   This file is included in .gitignore to prevent accidentally committing your API keys.
   
   **Option 2: Using environment variables (better for production):**
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
   
   **Note:** 
   - You can use both methods simultaneously. The system will check for credentials in the local config file first, then fall back to environment variables.
   - Currently, only OpenAI models are fully supported.

### Troubleshooting
- **Poetry Not Found:** If you get a "command not found" error, ensure Poetry's bin directory is added to your PATH.
- **Python Version Mismatch:** If you see "Python version X is not supported by the project", modify the `pyproject.toml` file as described above.
- **SSL/OpenSSL Warnings:** You may see warnings about OpenSSL versions with urllib3. These are usually harmless and can be ignored.
- **Missing Dependencies:** If you encounter errors about missing modules, try running `poetry update` followed by `poetry install`.

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

## 6. Running the Pipeline
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

**Note:** Currently, only OpenAI models are fully supported.

### Execute the Pipeline
Make sure you're in the Poetry virtual environment (see Installation Step 4), then run:

```sh
# Navigate to the src directory
cd src

# Run the main module
python -m transcriptanalysis.main
```

Alternatively, you can use Poetry's run script (from the project root):

```sh
poetry run transcriptanalysis.main:run
```

If you encounter any errors, check the logs for details.

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

