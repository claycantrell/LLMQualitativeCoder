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
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Any, Optional, List
from datetime import datetime

# Import our package modules
from transcriptanalysis.main import main
from transcriptanalysis.config_schemas import ConfigModel, CodingModeEnum, LoggingLevelEnum, LLMConfig, ProviderEnum
from transcriptanalysis.utils import load_environment_variables

# Create logger
logger = logging.getLogger("transcriptanalysis.api")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Define project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Initialize FastAPI app
app = FastAPI(
    title="TranscriptAnalysis API",
    description="API for running the qualitative coding pipeline",
    version="1.0.0"
)

# Add CORS middleware
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

@app.get("/")
def read_root():
    return {"message": "Welcome to the LLM Qualitative Coder API"}

@app.get("/jobs")
def list_jobs():
    """List all jobs"""
    return {
        "jobs": [job_info.to_dict() for job_info in jobs.values()]
    }

@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    """Get job status"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id].to_dict()

@app.post("/run-pipeline")
async def run_pipeline(
    config: ApiConfigModel,
    background_tasks: BackgroundTasks
):
    """Run the qualitative coding pipeline with the provided configuration"""
    try:
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
            config=internal_config
        )
        
        return {
            "job_id": job_id,
            "status": job_info.status,
            "message": "Pipeline started"
        }
    except Exception as e:
        logger.exception("Failed to start pipeline")
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
    
    # Create paths config
    paths_config = {
        "prompts_folder": "prompts",
        "codebase_folder": "qual_codebase",
        "json_folder": "json_inputs",
        "config_folder": "configs"
    }
    
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
        selected_json_file="teacher_transcript.json",
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

def run_pipeline_task(job_id: str, config: ConfigModel):
    """Run the pipeline task in the background"""
    job_info = jobs[job_id]
    job_info.status = JobStatus.RUNNING
    
    try:
        # Load environment variables and update config
        env_vars = load_environment_variables()
        if env_vars.get("OPENAI_API_KEY"):
            config.parse_llm_config.api_key = env_vars["OPENAI_API_KEY"]
            config.assign_llm_config.api_key = env_vars["OPENAI_API_KEY"]
        
        # Run main pipeline
        main(config)
        
        # Update job status
        job_info.status = JobStatus.COMPLETED
        job_info.completed_at = datetime.now()
        
        # Find output files
        output_dir = Path(config.output_folder)
        json_files = list(output_dir.glob("*_output_*.json"))
        validation_files = list(output_dir.glob("*_validation_report.json"))
        
        if json_files:
            job_info.output_file = str(json_files[0])
        
        if validation_files:
            job_info.validation_file = str(validation_files[0])
        
        logger.info(f"Pipeline completed successfully for job {job_id}")
    except Exception as e:
        logger.exception(f"Pipeline failed for job {job_id}")
        job_info.status = JobStatus.FAILED
        job_info.error = str(e)
        job_info.completed_at = datetime.now()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 