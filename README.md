Transcript Analyzer is a comprehensive tool designed to facilitate the analysis of qualitative data through automated coding processes. This pipeline leverages advanced language models to parse, transform, and assign codes to qualitative data.

Flexible Data Handling: Supports various JSON data formats
Automated Parsing: Breaks down large texts into smaller, manageable meaning units.
Deductive and Inductive Coding: Offers both predefined (deductive) and emergent (inductive) coding approaches.
Retrieval-Augmented Generation (RAG): Enhances code assignment accuracy by leveraging FAISS indexes.
Customizable Configuration: Easily adaptable to different datasets and coding schemas.

Main Pipeline (main.py): Orchestrates the entire workflow, from data loading to code assignment and validation.
Data Handlers (data_handlers.py): Manages data loading, transformation, and filtering based on configuration.
Qualitative Functions (qual_functions.py): Contains core functionalities like parsing transcripts and assigning codes.
Utilities (utils.py): Provides helper functions for environment setup, configuration loading, and resource initialization.
Validator (validator.py): Ensures the output's consistency and completeness through validation reports.

Configuration
The pipeline is highly configurable through two main JSON configuration files:
Pipeline Configuration (config.json)
Data Format Configuration (data_format_config.json)

1. Pipeline Configuration (config.json)
This file controls the overall behavior of the pipeline, including coding modes, model selections, paths, and logging settings.

2. Data Format Configuration (data_format_config.json)
This file defines how different data formats are handled, specifying fields for content, speaker, source IDs, and any filtering rules.

Usage
Setting Up Environment Variables
Before running the pipeline, set the OpenAI API key.
export OPENAI_API_KEY='your-openai-api-key'
On Windows:
set OPENAI_API_KEY=your-openai-api-key

Running the Pipeline
Execute the main script to start the coding process:

Modules
1. main.py
The entry point of the pipeline, responsible for orchestrating all stages from data loading to validation. It reads configurations, initializes resources, processes data, assigns codes, and generates outputs and reports.

2. data_handlers.py
Handles data operations, including loading JSON files, applying filter rules, and transforming data into MeaningUnit objects. It ensures that data conforms to the specified formats and applies necessary preprocessing steps.

3. qual_functions.py
Contains core functionalities such as parsing transcripts into meaning units and assigning codes using language models. It interfaces with OpenAI's API and manages FAISS indexes for RAG.

4. utils.py
Provides utility functions for environment setup, configuration loading, prompt file handling, and initializing resources like FAISS indexes. It ensures that all necessary resources are available and correctly configured.

5. validator.py
Validates the consistency and integrity of the coded outputs. It compares original speaking turns with concatenated meaning units, identifies inconsistencies, and generates detailed validation reports in JSON format.

Input and Output
Input
JSON Files: The pipeline accepts JSON files containing qualitative data. Depending on the data_format specified in the configuration, it expects certain fields:

Codebase Files: For deductive coding, JSONL files containing predefined codes are used. Each line should represent a JSON object with text and metadata.

Prompts: Text files containing prompts for parsing and coding instructions.

Coded JSON Files: The output includes meaning_units with assigned codes and document_metadata. These files are saved in the specified output_folder with timestamped filenames.

Validation Reports: JSON reports detailing skipped and inconsistent speaking turns to ensure data integrity.

Logs: Detailed logs are maintained in both the console and specified log files for monitoring and debugging.

Logging Levels: Configurable through config.json (DEBUG, INFO, WARNING, ERROR, CRITICAL).

Log Outputs: Logs are output to the console and can be saved to files as specified in the configuration.

Master Log: A master log file (logs/master_log.jsonl) records each run's timestamp, output file, and configuration for auditing purposes.

Validation Reports: Post-processing reports (*_validation_report.json) highlight any discrepancies between the original data and the coded outputs, ensuring reliability.