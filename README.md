## 1. Overview
LLMQualitativeCoder is a tool for automated qualitative coding using Large Language Models (LLMs). It provides a modern web interface for easy interaction with the qualitative coding pipeline, as well as a programmable API for advanced users.

Key features include:

- **Web Interface:** A modern React-based UI for uploading files, configuring the pipeline, running analyses, and managing results.
- **Flexible Data Handling:** Supports diverse JSON input data formats with customizable fields and basic filtering.
- **Automated Parsing:** Breaks down large texts into smaller meaning units for coding based on specifications in LLM user prompt.
- **Deductive and Inductive Coding:** Offers both predefined (deductive) and emergent (inductive) coding approaches.
- **Customizable Configuration:** Configure batch size, context size, and LLM provider/model directly through the UI.

## 2. Quick Start (Using the Web Interface)

The easiest way to use LLMQualitativeCoder is through its web interface, which requires minimal setup.

### Starting the Application

1. **Clone the Repository:**
   ```
   git clone https://github.com/iggygraceful/LLMQualitativeCoder.git
   cd LLMQualitativeCoder
   ```

2. **Install Dependencies:**
   Follow the installation steps in section 3 below to set up the required dependencies.

3. **Start the Backend Server:**
   ```sh
   # Activate your virtual environment if not already active
   source /path/to/your/virtualenv/bin/activate
   
   # Start the FastAPI server
   cd src
   uvicorn transcriptanalysis.new_api:app --reload --host 0.0.0.0 --port 8000
   ```

4. **Start the Frontend Server:**
   ```sh
   # Navigate to the frontend directory
   cd frontend
   
   # Start the development server
   npm run dev
   ```

5. **Access the UI:**
   Open your browser and navigate to:
   ```
   http://localhost:5173
   ```
   (Or the port displayed in your terminal after starting the frontend server)

### Using the Web Interface

The web interface provides an intuitive workflow for qualitative coding:

#### 1. File Management
- **View Available Files:** The home screen displays both default example files and your uploaded files.
- **Upload New Files:** Click the "Upload JSON File" button to add your own JSON files for analysis.
- **Preview Files:** Click on a file to see its content before processing.
- **Delete Files:** Remove user-uploaded files when no longer needed.

#### 2. Configure Analysis
- **Select a File:** Click the "Configure" button next to a file to set up analysis parameters.
- **Map Fields:** Specify which fields in your JSON contain the text content, contextual information, etc.
- **Choose Coding Mode:** Select between inductive (generate new codes) or deductive (use predefined codes) coding.
- **Configure Segmentation:** Choose between LLM-based or sentence-based segmentation of your text.
- **Select Model:** Choose which OpenAI model to use for analysis.
- **Advanced Options:** Configure batch sizes, context window, and thread count.

#### 3. Manage Codebases (for Deductive Coding)
- **View Codebases:** See available code schemes under the "Codebases" tab.
- **Create New Codebases:** Build your own coding scheme or modify existing ones.
- **Add Codes:** Add new codes to your codebase with descriptions and metadata.

#### 4. Customize Prompts
- **Edit Prompts:** Customize the instructions sent to the LLM for inductive, deductive, and parsing tasks.
- **Preview Prompts:** See how your configuration will appear in the prompt sent to the LLM.
- **Reset to Default:** Easily reset prompts to their default templates if needed.

#### 5. Run Analysis and View Results
- **Start Jobs:** Begin the analysis process with your selected configuration.
- **Monitor Progress:** See the status of running and completed jobs.
- **Download Results:** Get output and validation files from completed jobs.
- **View Reports:** Analyze validation reports to check for quality issues.

#### 6. API Key Management
- **Set OpenAI API Key:** Enter your OpenAI API key in the settings panel to authenticate API requests.
- **Secure Storage:** Your key is stored securely in memory for the duration of your session.

## 3. Installation & Setup
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
  **Note:** To make the `poetry` command available in all terminal sessions, add the relevant `export PATH=...` line to your shell's startup file (e.g., `~/.zshrc`, `~/.bash_profile`, or `~/.config/fish/config.fish`) and then source it or open a new terminal.
  
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
   Run the `source` command that `poetry env activate` outputs.

   ### Verify Environment Activation (Optional)
   After running the `source .../activate` command, you can verify the environment is active by:
   1. Checking your command prompt: It should now be prefixed with the environment name (e.g., `(transcriptanalysis-py3.9)`).
   2. Checking the Python interpreter path:
      ```sh
      which python
      # Should point to the python executable within your virtual environment
      ```
   3. Listing installed packages:
      ```sh
      pip list
      # You should see the project's dependencies installed here
      ```

5. **Install Additional Required Packages:**
   ```
   pip install python-multipart
   ```
   This package is required for file uploads in the web interface.

6. **Set Environment Variables:**
   Configure API keys before running the pipeline:
   - On Linux/macOS:
     ```
     export OPENAI_API_KEY='your-openai-api-key'
     # export HUGGINGFACE_API_KEY='your-huggingface-api-key' # If HuggingFace is supported later
     ```
   - On Windows CMD:
     ```
     set OPENAI_API_KEY=your-openai-api-key
     # set HUGGINGFACE_API_KEY=your-huggingface-api-key # If HuggingFace is supported later
     ```
   **Note:** Currently, only OpenAI models are fully supported.
   
   **Alternative:** You can also set your API key through the web interface, eliminating the need to set environment variables.

### Troubleshooting
- **Poetry Not Found:** If you get a "command not found" error, ensure Poetry's bin directory is added to your PATH.
- **Python Version Mismatch:** If you see "Python version X is not supported by the project", modify the `pyproject.toml` file as described above.
- **SSL/OpenSSL Warnings:** You may see warnings about OpenSSL versions with urllib3. These are usually harmless and can be ignored.
- **Missing Dependencies:** If you encounter errors about missing modules, try running `poetry update` followed by `poetry install`.
- **File Upload Issues:** If file uploads are not working, ensure you have installed the `python-multipart` package.
- **No Files Showing in UI:** Make sure the backend server is running and check the terminal for any error messages.

## 4. Advanced: Running the Pipeline Programmatically

For advanced users who prefer to run the pipeline programmatically without the web interface:

### Setting Up Environment Variables
Set the necessary API keys:
- On Linux/macOS:
  ```
  export OPENAI_API_KEY='your-openai-api-key'
  ```
- On Windows CMD:
  ```
  set OPENAI_API_KEY=your-openai-api-key
  ```

**Important:** Ensure you have set your `OPENAI_API_KEY` environment variable with a valid key **before** proceeding to execute the pipeline. The application will not function correctly without it.

### Execute the Pipeline
Make sure you're in the Poetry virtual environment (see Installation Step 4 and verify its activation), then run:

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
A successful run will show API communication logs (HTTP 200) and messages indicating that the output and validation report JSON files have been saved to the `outputs/` directory.

## 5. Configuration
### Pipeline Configuration (`config.json`)
Defines the pipeline behavior, including coding modes, model selections, paths, and logging settings.

**Note on Configuration File Paths:** The application expects `config.json` and `data_format_config.json` (described below) to be in the `src/transcriptanalysis/configs/` directory by default. Ensure your input data files (e.g., `teacher_transcript.json`) and prompt files (e.g., `parse_prompt.txt`) are correctly pathed within your `config.json` relative to the project structure and the paths specified in `config.json` itself (e.g., `paths.json_folder`, `paths.prompts_folder`).

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
    "config_folder": "transcriptanalysis/configs",
    "user_uploads_folder": "data/user_uploads"
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
    "provider": "openai",
    "model_name": "gpt-4o-mini",
    "temperature": 0.6,
    "max_tokens": 1500,
    "api_key": "YOUR_OPENAI_API_KEY"
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

## 6. Key Components
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

### `new_api.py`
**Purpose:** Provides a FastAPI server for asynchronous pipeline execution and web interface support.
**Key Endpoints:**
- `POST /run-pipeline` - Start a new coding job
- `GET /jobs/{job_id}` - Get job status
- `GET /jobs/{job_id}/output` - Download output file
- `GET /jobs/{job_id}/validation` - Download validation report
- `POST /files/upload` - Upload a JSON file
- `GET /files/list` - List available files
- `DELETE /files/{filename}` - Delete a user-uploaded file

## 7. Validation Process
Validation ensures the final meaning units accurately represent the original segments. Discrepancies are reported in `validation_report.json`.

## 8. Logging
Logs capture pipeline operations and are saved in the specified `log_file_path`.

## 9. Input and Output
### Input
- **Preliminary Segments:** JSON files containing raw data.
- **Codebase Files:** JSONL files for deductive coding.
- **Prompts:** Text files with LLM instructions.

### Output
- **Coded JSON Files:** Contain meaning units with assigned codes.
- **Validation Reports:** Detail discrepancies between input and output.
- **Logs:** Available in the console and specified files.

