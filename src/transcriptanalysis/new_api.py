#!/usr/bin/env python3
# new_api.py

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uuid
import json
import logging
import os
import shutil
from pathlib import Path
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import our package modules
from transcriptanalysis.main import main
from transcriptanalysis.config_schemas import ConfigModel, CodingModeEnum, LoggingLevelEnum, LLMConfig, ProviderEnum
from transcriptanalysis.utils import load_environment_variables

# Setup enhanced logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("transcriptanalysis.api")

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Define user uploads directory (outside the package)
USER_UPLOADS_DIR = PROJECT_ROOT / "data" / "user_uploads"
USER_UPLOADS_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="TranscriptAnalysis API",
    description="API for running the qualitative coding pipeline",
    version="1.0.0"
)

# Add CORS middleware to allow requests from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# In-memory job store
jobs = {}

class JobStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

class JobInfo:
    def __init__(self, job_id: str):
        self.job_id = job_id
        self.status = JobStatus.PENDING
        self.started_at = datetime.now()
        self.completed_at = None
        self.error = None
        self.output_file = None
        self.validation_file = None
    
    def to_dict(self):
        return {
            "job_id": self.job_id,
            "status": self.status,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "output_file": str(self.output_file) if self.output_file else None,
            "validation_file": str(self.validation_file) if self.validation_file else None
        }

class ApiConfigModel(BaseModel):
    """API-friendly version of ConfigModel for simplified request validation"""
    coding_mode: str  # "deductive" or "inductive"
    use_parsing: bool
    preliminary_segments_per_prompt: int
    meaning_units_per_assignment_prompt: int
    context_size: int
    data_format: str = "transcript"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    thread_count: int = 2
    input_file: Optional[str] = None  # Name of input file to use (can be default or user-uploaded)

class FileInfo(BaseModel):
    """Information about a file"""
    filename: str
    size: int
    upload_date: str
    is_default: bool

@app.get("/")
def read_root():
    return {"message": "Welcome to the LLM Qualitative Coder API"}

# File handling endpoints
@app.post("/files/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a new JSON input file"""
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Only JSON files are supported")
        
        # Save file to user uploads directory
        file_path = USER_UPLOADS_DIR / file.filename
        
        # Check if file already exists
        if file_path.exists():
            # Add timestamp to filename to avoid overwriting
            filename_parts = file.filename.rsplit('.', 1)
            timestamped_filename = f"{filename_parts[0]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{filename_parts[1]}"
            file_path = USER_UPLOADS_DIR / timestamped_filename
        
        # Save the file
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Validate JSON format
        try:
            with open(file_path, "r") as f:
                json.load(f)
        except json.JSONDecodeError:
            # If not valid JSON, delete the file and raise an error
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="Invalid JSON file")
        
        logger.info(f"File uploaded successfully: {file_path}")
        return {"filename": file_path.name, "size": file_path.stat().st_size}
    
    except Exception as e:
        if not isinstance(e, HTTPException):
            logger.exception(f"Error uploading file: {e}")
            raise HTTPException(status_code=500, detail=f"Error uploading file: {str(e)}")
        raise

@app.get("/files/list")
def list_files():
    """List all available input files (both default and user-uploaded)"""
    try:
        # Get default files from the package
        default_dir = Path(__file__).resolve().parent / "json_inputs"
        default_files = []
        
        if default_dir.exists():
            for file_path in default_dir.glob("*.json"):
                if file_path.name != "__init__.py" and file_path.is_file():
                    default_files.append(FileInfo(
                        filename=file_path.name,
                        size=file_path.stat().st_size,
                        upload_date=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        is_default=True
                    ).dict())
        
        # Get user-uploaded files
        user_files = []
        for file_path in USER_UPLOADS_DIR.glob("*.json"):
            if file_path.is_file():
                user_files.append(FileInfo(
                    filename=file_path.name,
                    size=file_path.stat().st_size,
                    upload_date=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                    is_default=False
                ).dict())
        
        return {"default_files": default_files, "user_files": user_files}
    
    except Exception as e:
        logger.exception(f"Error listing files: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing files: {str(e)}")

@app.delete("/files/{filename}")
def delete_file(filename: str):
    """Delete a user-uploaded file"""
    try:
        file_path = USER_UPLOADS_DIR / filename
        
        # Check if file exists
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
        # Check if it's a default file (which shouldn't be deleted)
        default_dir = Path(__file__).resolve().parent / "json_inputs"
        default_file_path = default_dir / filename
        if default_file_path.exists():
            raise HTTPException(status_code=403, detail="Cannot delete default files")
        
        # Delete the file
        os.remove(file_path)
        logger.info(f"File deleted: {filename}")
        
        return {"message": f"File {filename} deleted successfully"}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting file: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting file: {str(e)}")

@app.get("/jobs")
def list_jobs():
    """List all jobs"""
    logger.debug(f"Returning {len(jobs)} jobs")
    return {
        "jobs": [job_info.to_dict() for job_info in jobs.values()]
    }

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        logger.warning(f"Job ID not found: {job_id}")
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id].to_dict()

@app.post("/run-pipeline")
async def run_pipeline(
    config: ApiConfigModel,
    background_tasks: BackgroundTasks
):
    """Run the qualitative coding pipeline with the provided configuration"""
    try:
        # Log received configuration
        logger.info(f"Received pipeline request with config: {config}")
        
        # Generate job ID
        job_id = str(uuid.uuid4())
        job_info = JobInfo(job_id)
        jobs[job_id] = job_info
        
        # Create job output directory
        job_dir = OUTPUTS_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        
        # Convert API config to internal config
        internal_config = create_internal_config(config, job_dir)
        
        # Start background task
        background_tasks.add_task(
            run_pipeline_task,
            job_id=job_id,
            config=internal_config,
            input_file=config.input_file
        )
        
        logger.info(f"Started pipeline job: {job_id}")
        return {
            "job_id": job_id,
            "status": job_info.status,
            "message": "Pipeline started"
        }
    except Exception as e:
        logger.exception(f"Failed to start pipeline: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/jobs/{job_id}/output")
def get_job_output(job_id: str):
    """Get job output file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = jobs[job_id]
    
    if job_info.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if not job_info.output_file or not os.path.exists(job_info.output_file):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    logger.info(f"Returning output file for job {job_id}: {job_info.output_file}")
    return FileResponse(
        path=job_info.output_file,
        filename=os.path.basename(job_info.output_file),
        media_type="application/json"
    )

@app.get("/jobs/{job_id}/validation")
def get_job_validation(job_id: str):
    """Get job validation file"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job_info = jobs[job_id]
    
    if job_info.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed")
    
    if not job_info.validation_file or not os.path.exists(job_info.validation_file):
        raise HTTPException(status_code=404, detail="Validation file not found")
    
    logger.info(f"Returning validation file for job {job_id}: {job_info.validation_file}")
    return FileResponse(
        path=job_info.validation_file,
        filename=os.path.basename(job_info.validation_file),
        media_type="application/json"
    )

def create_internal_config(api_config: ApiConfigModel, output_dir: Path) -> ConfigModel:
    """Convert API config to internal config"""
    # Set up LLM config (using same config for both parse and assign)
    llm_config = LLMConfig(
        provider=ProviderEnum.OPENAI,
        model_name=api_config.model_name,
        temperature=api_config.temperature,
        max_tokens=api_config.max_tokens,
        api_key=""  # Will be loaded from environment
    )
    
    # Input file selection - either from user uploads or default
    input_file = api_config.input_file or "teacher_transcript.json"
    
    # Create paths config - include both package directories and user directories
    paths_config = {
        "prompts_folder": "prompts",
        "codebase_folder": "qual_codebase",
        "json_folder": "json_inputs",  # Default package location
        "config_folder": "configs",
        "user_uploads_folder": str(USER_UPLOADS_DIR)  # Add user uploads folder
    }
    
    logger.debug(f"Creating internal config with output dir: {output_dir}")
    
    # Create full config
    return ConfigModel(
        coding_mode=CodingModeEnum(api_config.coding_mode),
        use_parsing=api_config.use_parsing,
        preliminary_segments_per_prompt=api_config.preliminary_segments_per_prompt,
        meaning_units_per_assignment_prompt=api_config.meaning_units_per_assignment_prompt,
        context_size=api_config.context_size,
        data_format=api_config.data_format,
        paths=paths_config,
        selected_codebase="teacher_schema.jsonl",
        selected_json_file=input_file,  # Use the selected input file
        parse_prompt_file="parse.txt",
        inductive_coding_prompt_file="inductive.txt",
        deductive_coding_prompt_file="deductive.txt",
        output_folder=str(output_dir),
        enable_logging=True,
        logging_level=LoggingLevelEnum.INFO,
        log_to_file=True,
        log_file_path=str(output_dir / "api_log.log"),
        thread_count=api_config.thread_count,
        parse_llm_config=llm_config,
        assign_llm_config=llm_config
    )

def run_pipeline_task(job_id: str, config: ConfigModel, input_file: Optional[str] = None):
    """Run the pipeline task in the background"""
    job_info = jobs[job_id]
    job_info.status = JobStatus.RUNNING
    
    try:
        logger.info(f"Starting pipeline execution for job {job_id}")
        
        # If using a user-uploaded file, we need to copy it to a temporary location
        # or modify the main.py code to look in the user uploads directory
        if input_file and (USER_UPLOADS_DIR / input_file).exists():
            logger.info(f"Using user-uploaded file: {input_file}")
            # For now, we'll modify the config to use the user uploads directory
            config.selected_json_file = str(USER_UPLOADS_DIR / input_file)
        
        # Load environment variables and update config
        env_vars = load_environment_variables()
        if env_vars.get("OPENAI_API_KEY"):
            config.parse_llm_config.api_key = env_vars["OPENAI_API_KEY"]
            config.assign_llm_config.api_key = env_vars["OPENAI_API_KEY"]
        else:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        
        # Run main pipeline
        main(config)
        
        # Update job status
        job_info.status = JobStatus.COMPLETED
        job_info.completed_at = datetime.now()
        
        # Find output files
        output_dir = Path(config.output_folder)
        json_files = list(output_dir.glob("*_output_*.json"))
        validation_files = list(output_dir.glob("*_validation_report.json"))
        
        logger.debug(f"Found output files: {json_files}")
        logger.debug(f"Found validation files: {validation_files}")
        
        if json_files:
            job_info.output_file = str(json_files[0])
        
        if validation_files:
            job_info.validation_file = str(validation_files[0])
        
        logger.info(f"Pipeline completed successfully for job {job_id}")
    except Exception as e:
        logger.exception(f"Pipeline failed for job {job_id}: {e}")
        job_info.status = JobStatus.FAILED
        job_info.error = str(e)
        job_info.completed_at = datetime.now()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 