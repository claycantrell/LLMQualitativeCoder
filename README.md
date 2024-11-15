= Qualitative Coding Application

== Introduction

The *Qualitative Coding Application* is a tool designed to assist researchers in analyzing qualitative data. Leveraging LLMs, this application automates the process of breaking down textual data into smaller units for analysis and assigning relevant codes.

== Features

* *Dynamic Configuration*: Driven by a `config.json` file, allowing easy customization without modifying the codebase.
* *Deductive and Inductive Coding*: Supports both deductive (using predefined codes) and inductive (generating codes based on guidelines) coding approaches.
* *Parsing Transcripts*: Breaks down speaking turns into smaller meaning units based on customizable prompts.
* *Code Retrieval with FAISS*: Utilizes FAISS for efficient retrieval of relevant codes, enhancing the relevance and accuracy of code assignments.
* *Integration with OpenAI*: Leverages OpenAI's language models for advanced natural language processing tasks.
* *Flexible Output Formats*: Exports coded data in JSON or CSV formats.
* *Comprehensive Logging*: Detailed logging with options to log to console and/or files, aiding in monitoring and debugging.

== Architecture

The application is structured into several key components, each responsible for specific functionalities:

* *`main.py`*: Orchestrates the entire workflow, from loading configurations to exporting results.
* *`utils.py`*: Contains utility functions for loading configurations, prompts, and initializing resources.
* *`data_handlers.py`*: Handles data loading, validation, and transformation of data into units for analysis.
* *`qual_functions.py`*: Implements core functionalities such as parsing transcripts and assigning codes to meaning units.

== Installation

=== Prerequisites

- Python 3.8 or above
- `pip` (Python package manager)
- An API key for OpenAI (set as an environment variable)

=== Steps

. *Clone the Repository*

git clone  cd 

. *Create and Activate a Virtual Environment*

python3 -m venv .venv source .venv/bin/activate // On Windows use ".venv\Scripts\activate"

. *Install Dependencies*

pip install -r requirements.txt

. *Set Up Environment Variables*

Ensure you have your OpenAI API key available as an environment variable:

export OPENAI_API_KEY='your_openai_api_key_here'

== Configuration

The application relies on a `config.json` file for all configurable settings. 

=== Configuration Parameters

* *coding_mode*: `"deductive"` or `"inductive"` depending on the approach you want to use (open or closed coding).
* *use_parsing*: Boolean to enable or disable parsing of texual data into smaller units.
* *use_rag*: Boolean to use Retrieval-Augmented Generation (RAG) for storage and retrieval of codebase (limits the size of context window in prompts, reducing costs with larger codebases. It could also improve performance with larger codebases but I have not tested this).
* Below are the default models used for this tool, better performance can be achived with larger and more costly models.
* *parse_model*: The model to use for parsing transcripts (e.g., `gpt-4o-mini`).
* *assign_model*: The model used for assigning codes (e.g., `gpt-4o-mini`).
* *initialize_embedding_model*: Model used for embedding (e.g., `text-embedding-3-small`).
* *retrieve_embedding_model*: Model used for code retrieval embeddings.
* *data_format*: Defines the format of the input data (e.g., `interview`).
* new data formats must be defined in (`data_format_config.json`) Here you will assign your `content_field` to the JSON value associated with your textual data for analysis. You will also list all other fields in your JSON file. 
* *paths*: Specifies folder and file paths to be used during the process.
** *prompts_folder*: Path to the folder containing prompt files.
** *codebase_folder*: Path to the folder containing the qualitative codebase files.
** *json_folder*: Path to the folder with JSON transcripts.
** *config_folder*: Path where configuration files are located.
** *parse_prompt_file*: File containing the prompt for parsing.
** *deductive_coding_prompt_file*: File for deductive coding prompts.
** *inductive_coding_prompt_file*: File for inductive coding prompts.
** *codebase_file*: JSONL file with the codebase definitions.
** *data_file*: JSON file to be processed.

== Usage

=== Running the Main Pipeline

To run the main pipeline, execute the `main.py` script:

python main.py --config_path=configs/config.json

This will initiate the entire process, from loading the data, parsing, code assignment, and finally exporting the results.

=== Configurable Modes

You can select between *Deductive* and *Inductive* coding modes by modifying the `"coding_mode"` parameter in the `config.json` file.

- *Deductive*: Predefined codes are used from the codebase.
- *Inductive*: Codes are generated during the analysis based on the content of the transcripts.

== Folder Structure

Here is a typical folder structure for the project:

qualitative-coding-app/ ├── configs/ │ ├── config.json │ └── data_format_config.json ├── prompts/ │ ├── parse_prompt.txt │ ├── deductive_coding_prompt.txt │ └── inductive_coding_prompt.txt ├── qual_codebase/ │ └── new_schema.jsonl ├── json_transcripts/ │ └── output_cues.json ├── main.py ├── utils.py ├── data_handlers.py ├── qual_functions.py ├── requirements.txt └── README.adoc

== Logging

The application generates logs to aid in debugging and understanding the workflow. Logging is configured with different levels (INFO, DEBUG, ERROR) to capture varying degrees of detail. 

Logs are printed to the console by default. You can configure log files or use different logging handlers as needed.

== Troubleshooting

=== Common Errors

* *Missing OpenAI API Key*: Make sure the environment variable `OPENAI_API_KEY` is set correctly.
* *File Not Found Errors*: Ensure all paths defined in the `config.json` file are correct.

=== Debugging Tips

* Enable DEBUG level logging to view detailed trace messages:
  
export LOG_LEVEL=DEBUG
