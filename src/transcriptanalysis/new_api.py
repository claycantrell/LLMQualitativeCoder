#!/usr/bin/env python3
# new_api.py

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form, Request, Body
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
from transcriptanalysis.config_schemas import ConfigModel, CodingModeEnum, LoggingLevelEnum, LLMConfig, ProviderEnum, DataFormatConfigItem, PathsModel
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

# Create prompts directory in user uploads
USER_PROMPTS_DIR = USER_UPLOADS_DIR / "prompts"
USER_PROMPTS_DIR.mkdir(exist_ok=True)

# Create codebases directory in user uploads
USER_CODEBASES_DIR = USER_UPLOADS_DIR / "codebases"
USER_CODEBASES_DIR.mkdir(exist_ok=True)

# Default prompts directory (inside the package)
DEFAULT_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

# Default codebases directory (inside the package)
DEFAULT_CODEBASES_DIR = Path(__file__).resolve().parent.parent.parent / "qual_codebase"

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
    selected_codebase: Optional[str] = None

class FileInfo(BaseModel):
    """Information about a file"""
    filename: str
    size: int
    upload_date: str
    is_default: bool

class CodebaseInfo(BaseModel):
    """Information about a codebase file"""
    filename: str
    size: int
    upload_date: str
    is_default: bool
    code_count: int

class CodeEntry(BaseModel):
    """A single code entry for a codebase"""
    text: str
    metadata: Dict[str, str]

class DynamicConfigModel(BaseModel):
    """Model for dynamic configuration from frontend"""
    file_id: str
    content_field: str
    context_fields: List[str] = []
    list_field: Optional[str] = None
    filter_rules: List[Dict[str, Any]] = []
    coding_mode: str = "inductive"
    use_parsing: bool = True
    preliminary_segments_per_prompt: int = 5
    meaning_units_per_assignment_prompt: int = 10
    context_size: int = 5
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 2000
    thread_count: int = 2
    selected_codebase: Optional[str] = None

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

@app.get("/analyze-file/{file_id}")
async def analyze_file_structure(file_id: str):
    """Analyze a JSON file's structure and suggest configuration mappings"""
    try:
        # Determine if it's a user-uploaded file or default file
        file_path = USER_UPLOADS_DIR / file_id
        
        if not file_path.exists():
            # Check in the default directory
            default_dir = Path(__file__).resolve().parent / "json_inputs"
            file_path = default_dir / file_id
            
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"File {file_id} not found")
        
        # Read the file
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Analyze structure
        structure = detect_json_structure(data)
        
        logger.info(f"Successfully analyzed file structure for {file_id}")
        return {
            "file_id": file_id,
            "structure": structure,
            "suggested_mappings": structure["suggested_mappings"]
        }
    
    except json.JSONDecodeError:
        logger.exception(f"Invalid JSON file: {file_id}")
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        logger.exception(f"Error analyzing file {file_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error analyzing file: {str(e)}")

@app.post("/run-pipeline-with-config")
async def run_pipeline_with_config(
    background_tasks: BackgroundTasks,
    config: DynamicConfigModel
):
    """Run the pipeline with a dynamic user-provided configuration"""
    try:
        # Validate codebase exists if in deductive mode
        if config.coding_mode == "deductive":
            if not config.selected_codebase:
                raise HTTPException(
                    status_code=400, 
                    detail="A codebase must be selected for deductive coding mode"
                )
            
            # Check if codebase exists in either location
            default_codebase_path = DEFAULT_CODEBASES_DIR / config.selected_codebase
            user_codebase_path = USER_CODEBASES_DIR / config.selected_codebase
            
            if not default_codebase_path.exists() and not user_codebase_path.exists():
                raise HTTPException(
                    status_code=404, 
                    detail=f"Codebase '{config.selected_codebase}' not found in either default or user directories"
                )
            
            logger.info(f"Using codebase: {config.selected_codebase} for deductive coding")
        
        # Create a job ID
        job_id = str(uuid.uuid4())
        job_info = JobInfo(job_id)
        jobs[job_id] = job_info
        
        # Create job output directory
        job_dir = OUTPUTS_DIR / job_id
        job_dir.mkdir(exist_ok=True)
        
        # Create internal config
        internal_config = create_internal_config_from_user_input(config, job_dir)
        
        # Start background task
        background_tasks.add_task(
            run_pipeline_task,
            job_id=job_id,
            config=internal_config,
            input_file=config.file_id
        )
        
        logger.info(f"Started pipeline job with dynamic config: {job_id}")
        return {
            "job_id": job_id,
            "status": job_info.status,
            "message": "Pipeline started with dynamic configuration"
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Failed to start pipeline with dynamic config: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Prompt management endpoints
@app.get("/prompts/{prompt_type}")
async def get_prompt(prompt_type: str):
    """Get a prompt's content and information about whether it's customized.
    
    Args:
        prompt_type: The type of prompt ('inductive' or 'deductive')
        
    Returns:
        A JSON object containing the prompt content and whether it's customized
    """
    if prompt_type not in ["inductive", "deductive"]:
        raise HTTPException(status_code=400, detail="Invalid prompt type. Must be 'inductive' or 'deductive'")
    
    try:
        # Check for custom prompt
        custom_prompt_path = USER_PROMPTS_DIR / f"{prompt_type}.txt"
        default_prompt_path = DEFAULT_PROMPTS_DIR / f"{prompt_type}.txt"
        
        is_custom = custom_prompt_path.exists()
        prompt_path = custom_prompt_path if is_custom else default_prompt_path
        
        if not prompt_path.exists():
            raise HTTPException(status_code=404, detail=f"Prompt file {prompt_type}.txt not found")
        
        # Read the prompt content
        with open(prompt_path, 'r') as f:
            content = f.read()
        
        return {
            "content": content,
            "is_custom": is_custom
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error reading prompt {prompt_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading prompt: {str(e)}")

@app.post("/prompts/{prompt_type}")
async def save_custom_prompt(prompt_type: str, content: str = Body(...)):
    """Save a customized version of a prompt.
    
    Args:
        prompt_type: The type of prompt ('inductive' or 'deductive')
        content: The new content for the prompt (sent as raw text)
        
    Returns:
        A JSON object confirming that the prompt was saved
    """
    if prompt_type not in ["inductive", "deductive"]:
        raise HTTPException(status_code=400, detail="Invalid prompt type. Must be 'inductive' or 'deductive'")
    
    try:
        # Ensure prompts directory exists
        USER_PROMPTS_DIR.mkdir(exist_ok=True)
        
        # Save the custom prompt
        custom_prompt_path = USER_PROMPTS_DIR / f"{prompt_type}.txt"
        
        with open(custom_prompt_path, 'w') as f:
            f.write(content)
        
        logger.info(f"Custom {prompt_type} prompt saved")
        
        return {
            "message": f"Custom {prompt_type} prompt saved successfully",
            "is_custom": True
        }
    
    except Exception as e:
        logger.exception(f"Error saving custom prompt {prompt_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving custom prompt: {str(e)}")

@app.delete("/prompts/{prompt_type}")
async def reset_prompt(prompt_type: str):
    """Reset a prompt to its default version by deleting the custom version.
    
    Args:
        prompt_type: The type of prompt ('inductive' or 'deductive')
        
    Returns:
        A JSON object confirming that the prompt was reset
    """
    if prompt_type not in ["inductive", "deductive"]:
        raise HTTPException(status_code=400, detail="Invalid prompt type. Must be 'inductive' or 'deductive'")
    
    try:
        # Check if a custom prompt exists
        custom_prompt_path = USER_PROMPTS_DIR / f"{prompt_type}.txt"
        
        if not custom_prompt_path.exists():
            return {
                "message": f"No custom {prompt_type} prompt found",
                "is_custom": False
            }
        
        # Delete the custom prompt
        os.remove(custom_prompt_path)
        
        logger.info(f"Custom {prompt_type} prompt reset to default")
        
        return {
            "message": f"Custom {prompt_type} prompt reset to default",
            "is_custom": False
        }
    
    except Exception as e:
        logger.exception(f"Error resetting prompt {prompt_type}: {e}")
        raise HTTPException(status_code=500, detail=f"Error resetting prompt: {str(e)}")

@app.get("/codebases/list")
def list_codebases():
    """List all available codebases (both default and user-created)"""
    try:
        # Get default codebases
        default_codebases = []
        
        if DEFAULT_CODEBASES_DIR.exists():
            for file_path in DEFAULT_CODEBASES_DIR.glob("*.jsonl"):
                if file_path.is_file():
                    # Count codes in the file
                    code_count = 0
                    with open(file_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                code_count += 1
                    
                    default_codebases.append(CodebaseInfo(
                        filename=file_path.name,
                        size=file_path.stat().st_size,
                        upload_date=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        is_default=True,
                        code_count=code_count
                    ).dict())
        
        # Get user codebases
        user_codebases = []
        if USER_CODEBASES_DIR.exists():
            for file_path in USER_CODEBASES_DIR.glob("*.jsonl"):
                if file_path.is_file():
                    # Count codes in the file
                    code_count = 0
                    with open(file_path, 'r') as f:
                        for line in f:
                            if line.strip():
                                code_count += 1
                    
                    user_codebases.append(CodebaseInfo(
                        filename=file_path.name,
                        size=file_path.stat().st_size,
                        upload_date=datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
                        is_default=False,
                        code_count=code_count
                    ).dict())
        
        return {"default_codebases": default_codebases, "user_codebases": user_codebases}
    
    except Exception as e:
        logger.exception(f"Error listing codebases: {e}")
        raise HTTPException(status_code=500, detail=f"Error listing codebases: {str(e)}")

@app.get("/codebases/{codebase_name}")
def get_codebase(codebase_name: str):
    """Get a codebase's content and information about whether it's a default codebase.
    
    Args:
        codebase_name: The name of the codebase file (e.g. 'news_schema.jsonl')
        
    Returns:
        A JSON object containing the codebase content and whether it's a default codebase
    """
    try:
        # Check for user codebase
        user_codebase_path = USER_CODEBASES_DIR / codebase_name
        default_codebase_path = DEFAULT_CODEBASES_DIR / codebase_name
        
        is_default = False
        codebase_path = None
        
        if user_codebase_path.exists():
            codebase_path = user_codebase_path
        elif default_codebase_path.exists():
            codebase_path = default_codebase_path
            is_default = True
        else:
            raise HTTPException(status_code=404, detail=f"Codebase {codebase_name} not found")
        
        # Read the codebase content
        codes = []
        with open(codebase_path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        code_entry = json.loads(line)
                        codes.append(code_entry)
                    except json.JSONDecodeError:
                        logger.warning(f"Invalid JSON line in codebase {codebase_name}: {line}")
        
        return {
            "codebase_name": codebase_name,
            "is_default": is_default,
            "codes": codes
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error reading codebase {codebase_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading codebase: {str(e)}")

@app.post("/codebases/create")
async def create_codebase(codebase_name: str = Form(...), base_codebase: Optional[str] = Form(None)):
    """Create a new codebase, optionally based on an existing one.
    
    Args:
        codebase_name: The name for the new codebase file (will be saved as {codebase_name}.jsonl)
        base_codebase: Optional name of an existing codebase to use as a template
        
    Returns:
        A JSON object confirming that the codebase was created
    """
    try:
        # Ensure the codebase name is valid
        if not codebase_name.endswith('.jsonl'):
            codebase_name = f"{codebase_name}.jsonl"
        
        # Check if the codebase already exists
        target_path = USER_CODEBASES_DIR / codebase_name
        if target_path.exists():
            raise HTTPException(status_code=400, detail=f"A codebase with name {codebase_name} already exists")
        
        # Create the codebases directory if it doesn't exist
        USER_CODEBASES_DIR.mkdir(exist_ok=True)
        
        # If a base codebase was specified, copy it
        if base_codebase:
            # Check if the base codebase exists
            base_path = None
            user_base_path = USER_CODEBASES_DIR / base_codebase
            default_base_path = DEFAULT_CODEBASES_DIR / base_codebase
            
            if user_base_path.exists():
                base_path = user_base_path
            elif default_base_path.exists():
                base_path = default_base_path
            else:
                raise HTTPException(status_code=404, detail=f"Base codebase {base_codebase} not found")
            
            # Copy the base codebase
            shutil.copy(base_path, target_path)
        else:
            # Create an empty codebase file
            with open(target_path, 'w') as f:
                pass  # Create an empty file
        
        logger.info(f"Codebase {codebase_name} created successfully")
        
        return {
            "message": f"Codebase {codebase_name} created successfully",
            "codebase_name": codebase_name
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error creating codebase: {e}")
        raise HTTPException(status_code=500, detail=f"Error creating codebase: {str(e)}")

@app.post("/codebases/{codebase_name}/add_code")
async def add_code_to_codebase(
    codebase_name: str,
    code: CodeEntry
):
    """Add a new code to an existing codebase.
    
    Args:
        codebase_name: The name of the codebase file to add the code to
        code: The code entry to add (with text and metadata)
        
    Returns:
        A JSON object confirming that the code was added
    """
    try:
        # Check if the codebase exists
        codebase_path = USER_CODEBASES_DIR / codebase_name
        
        if not codebase_path.exists():
            # Check if it's a default codebase
            default_codebase_path = DEFAULT_CODEBASES_DIR / codebase_name
            if default_codebase_path.exists():
                # For default codebases, create a copy in the user directory first
                USER_CODEBASES_DIR.mkdir(exist_ok=True)
                shutil.copy(default_codebase_path, codebase_path)
                logger.info(f"Created user copy of default codebase {codebase_name}")
            else:
                raise HTTPException(status_code=404, detail=f"Codebase {codebase_name} not found")
        
        # Add the new code to the codebase
        with open(codebase_path, 'a') as f:
            # Check if file is empty or ends with a newline
            needs_newline = False
            if codebase_path.stat().st_size > 0:
                with open(codebase_path, 'rb') as check_file:
                    # Move to the last character of the file
                    check_file.seek(max(0, codebase_path.stat().st_size - 1))
                    last_char = check_file.read(1)
                    needs_newline = last_char != b'\n'
            
            # Add a newline only if needed
            if needs_newline:
                f.write("\n")
            
            # Write the JSON object
            json.dump(code.dict(), f)
            
            # Ensure the file ends with a newline (good practice for text files)
            f.write("\n")
        
        logger.info(f"Code '{code.text}' added to codebase {codebase_name}")
        
        return {
            "message": f"Code '{code.text}' added to codebase {codebase_name}",
            "codebase_name": codebase_name,
            "code": code
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error adding code to codebase {codebase_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error adding code to codebase: {str(e)}")

@app.delete("/codebases/{codebase_name}")
def delete_codebase(codebase_name: str):
    """Delete a user-created codebase.
    
    Args:
        codebase_name: The name of the codebase file to delete
        
    Returns:
        A JSON object confirming that the codebase was deleted
    """
    try:
        # Check if the codebase exists
        codebase_path = USER_CODEBASES_DIR / codebase_name
        
        if not codebase_path.exists():
            raise HTTPException(status_code=404, detail=f"Codebase {codebase_name} not found")
        
        # Check if it's a default codebase that was copied
        default_codebase_path = DEFAULT_CODEBASES_DIR / codebase_name
        if default_codebase_path.exists():
            logger.warning(f"Deleting user copy of default codebase {codebase_name}")
        
        # Delete the codebase
        os.remove(codebase_path)
        
        logger.info(f"Codebase {codebase_name} deleted")
        
        return {
            "message": f"Codebase {codebase_name} deleted successfully"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error deleting codebase {codebase_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Error deleting codebase: {str(e)}")

@app.get("/files/{filename}/content")
def get_file_content(filename: str):
    """Get the content of a file (either user-uploaded or default)"""
    try:
        # Check user uploads first
        file_path = USER_UPLOADS_DIR / filename
        
        if not file_path.exists():
            # Check in the default directory
            default_dir = Path(__file__).resolve().parent / "json_inputs"
            file_path = default_dir / filename
            
            if not file_path.exists():
                raise HTTPException(status_code=404, detail=f"File {filename} not found")
        
        # Read the file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Try to parse as JSON to ensure it's valid
        try:
            json_content = json.loads(content)
            # Return the parsed JSON for proper formatting
            return {"content": json_content, "filename": filename}
        except json.JSONDecodeError:
            # If not valid JSON, return as plain text
            return {"content": content, "filename": filename, "is_text": True}
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error reading file content: {e}")
        raise HTTPException(status_code=500, detail=f"Error reading file content: {str(e)}")

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
    
    # Use the selected codebase from the API config if provided, otherwise use default
    selected_codebase = api_config.selected_codebase if hasattr(api_config, 'selected_codebase') and api_config.selected_codebase else "teacher_schema.jsonl"
    
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
        selected_codebase=selected_codebase,
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

def create_internal_config_from_user_input(config: DynamicConfigModel, output_dir: Path) -> ConfigModel:
    """
    Convert user-provided configuration to internal ConfigModel format
    """
    # Set up LLM config (using same config for both parse and assign)
    llm_config = LLMConfig(
        provider=ProviderEnum.OPENAI,
        model_name=config.model_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        api_key=""  # Will be loaded from environment
    )
    
    # Create paths config
    paths_config = {
        "prompts_folder": "prompts",
        "codebase_folder": "qual_codebase",
        "json_folder": "json_inputs",
        "config_folder": "configs",
        "user_uploads_folder": str(USER_UPLOADS_DIR)
    }
    
    # Create custom format config
    custom_format_config = DataFormatConfigItem(
        content_field=config.content_field,
        context_fields=config.context_fields,
        list_field=config.list_field,
        source_id_field=None,  # We don't expose this in the UI for simplicity
        filter_rules=[]  # We don't expose this in the UI for simplicity
    )
    
    # Use the selected codebase from the UI if provided, otherwise use default
    selected_codebase = config.selected_codebase if hasattr(config, 'selected_codebase') and config.selected_codebase else "teacher_schema.jsonl"
    
    # Create full config
    return ConfigModel(
        coding_mode=CodingModeEnum(config.coding_mode),
        use_parsing=config.use_parsing,
        preliminary_segments_per_prompt=config.preliminary_segments_per_prompt,
        meaning_units_per_assignment_prompt=config.meaning_units_per_assignment_prompt,
        context_size=config.context_size,
        data_format="custom",  # Mark as custom
        paths=PathsModel(**paths_config),
        selected_codebase=selected_codebase,
        selected_json_file=config.file_id,  # Use the file ID
        parse_prompt_file="parse.txt",
        inductive_coding_prompt_file="inductive.txt",
        deductive_coding_prompt_file="deductive.txt",
        output_folder=str(output_dir),
        enable_logging=True,
        logging_level=LoggingLevelEnum.INFO,
        log_to_file=True,
        log_file_path=str(output_dir / "api_log.log"),
        thread_count=config.thread_count,
        parse_llm_config=llm_config,
        assign_llm_config=llm_config,
        custom_format_config=custom_format_config
    )

def detect_json_structure(data):
    """
    Analyze a JSON document to determine fields and structure
    Returns detailed information about the fields for configuration
    """
    structure = {
        "fields": [],
        "arrays": [],
        "objects": [],
        "suggested_mappings": {
            "content_field": None,
            "context_fields": [],
            "list_field": None
        }
    }
    
    # Handle top-level array
    if isinstance(data, list) and len(data) > 0:
        structure["is_array"] = True
        structure["suggested_mappings"]["list_field"] = "root"
        
        # Analyze the first item if it's an object
        if len(data) > 0 and isinstance(data[0], dict):
            sample_item = data[0]
            analyze_object_fields(sample_item, structure)
    
    # Handle top-level object
    elif isinstance(data, dict):
        structure["is_object"] = True
        analyze_object_fields(data, structure)
        
        # Check for arrays in the object that might contain the main data
        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0:
                structure["arrays"].append(key)
                if len(value) > 0 and isinstance(value[0], dict):
                    structure["suggested_mappings"]["list_field"] = key
                    analyze_object_fields(value[0], structure, prefix=f"{key}.")
    
    return structure

def analyze_object_fields(obj, structure, prefix=""):
    """Helper function to analyze fields in an object"""
    for key, value in obj.items():
        field_path = f"{prefix}{key}"
        field_info = {
            "name": field_path,
            "type": determine_field_type(value)
        }
        
        # Add field to the fields list
        structure["fields"].append(field_info)
        
        # Detect potential content fields (text fields with substantial content)
        if field_info["type"] == "string" and isinstance(value, str) and len(value) > 50:
            structure["suggested_mappings"]["content_field"] = field_path
        
        # Detect potential context fields (shorter text or metadata)
        elif field_info["type"] in ["string", "number", "boolean", "date"] and field_path != structure["suggested_mappings"]["content_field"]:
            structure["suggested_mappings"]["context_fields"].append(field_path)
        
        # Recursively analyze nested objects
        if isinstance(value, dict):
            structure["objects"].append(field_path)
            analyze_object_fields(value, structure, prefix=f"{field_path}.")

def determine_field_type(value):
    """Helper function to determine the data type of a field"""
    if isinstance(value, str):
        # Try to detect if it's a date
        try:
            datetime.fromisoformat(value.replace('Z', '+00:00'))
            return "date"
        except (ValueError, TypeError):
            return "string"
    elif isinstance(value, (int, float)):
        return "number"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, list):
        return "array"
    elif isinstance(value, dict):
        return "object"
    elif value is None:
        return "null"
    else:
        return "unknown"

@app.post("/preview-prompt")
async def preview_prompt(config: DynamicConfigModel):
    """
    Build and return the prompt that would be sent to the LLM without actually sending it.
    This is used for preview purposes in the frontend.
    """
    try:
        logger.info(f"Building prompt preview for config: {config}")
        
        # First, get the file content
        file_path = USER_UPLOADS_DIR / config.file_id
        if not file_path.exists():
            raise HTTPException(status_code=404, detail=f"File {config.file_id} not found")
        
        # Load the file
        with open(file_path, 'r') as f:
            file_data = json.load(f)
        
        # Get the appropriate prompt
        prompt_content = ""
        if config.coding_mode == "inductive":
            # Check if there's a custom inductive prompt
            custom_prompt_file = USER_PROMPTS_DIR / "inductive.txt"
            default_prompt_file = DEFAULT_PROMPTS_DIR / "inductive.txt"
            prompt_path = custom_prompt_file if custom_prompt_file.exists() else default_prompt_file
            
            with open(prompt_path, 'r') as f:
                prompt_content = f.read()
                
            logger.info(f"Using {'custom' if custom_prompt_file.exists() else 'default'} inductive prompt for preview")
        else:  # deductive
            # Check if there's a custom deductive prompt
            custom_prompt_file = USER_PROMPTS_DIR / "deductive.txt"
            default_prompt_file = DEFAULT_PROMPTS_DIR / "deductive.txt"
            prompt_path = custom_prompt_file if custom_prompt_file.exists() else default_prompt_file
            
            with open(prompt_path, 'r') as f:
                prompt_content = f.read()
                
            logger.info(f"Using {'custom' if custom_prompt_file.exists() else 'default'} deductive prompt for preview")
        
        # Build a sample meaning unit
        # Get the data based on configuration
        list_field = config.list_field
        content_field = config.content_field
        context_fields = config.context_fields or []
        
        # Get the number of meaning units to include in the prompt (from config)
        num_units = config.meaning_units_per_assignment_prompt
        
        # Prepare sample data with multiple items based on meaning_units_per_assignment_prompt
        all_items = []
        
        # Handle case where the data is a top-level array (list_field is "root")
        if list_field == "root" and isinstance(file_data, list) and len(file_data) > 0:
            # Get up to num_units items or all available items
            all_items = file_data[:min(num_units, len(file_data))]
        # Handle case where list_field is a property in the data
        elif list_field and list_field in file_data and isinstance(file_data[list_field], list) and file_data[list_field]:
            all_items = file_data[list_field][:min(num_units, len(file_data[list_field]))]
        else:
            # If no list field or not found, use the whole file data as one item
            all_items = [file_data]
        
        sample_data = all_items
        
        # Prepare codes block
        codes_block = ""
        if config.coding_mode == "deductive":
            codes_block = "Full Codebase (all codes with details):\n"
            if config.selected_codebase:
                # Try to load the codebase
                user_codebase_path = USER_CODEBASES_DIR / config.selected_codebase
                default_codebase_path = DEFAULT_CODEBASES_DIR / config.selected_codebase
                
                codebase_path = user_codebase_path if user_codebase_path.exists() else default_codebase_path
                
                if codebase_path.exists():
                    with open(codebase_path, 'r') as f:
                        codebase_lines = f.readlines()
                        if codebase_lines:
                            # Just include first 3 codes for preview
                            preview_lines = codebase_lines[:min(3, len(codebase_lines))]
                            codes_block += "".join(preview_lines)
                            if len(codebase_lines) > 3:
                                codes_block += "\n... (additional codes not shown in preview) ...\n"
                    codes_block += "\n\n"
                else:
                    codes_block += f"[Codebase {config.selected_codebase} would be loaded here]\n\n"
        else:
            codes_block = "Guidelines for Inductive Coding:\nNo predefined codes. Please generate codes based on the following guidelines.\n\n"
        
        # Create context block for multiple meaning units
        context_info = ""
        for idx, item in enumerate(sample_data):
            if not item:
                continue
                
            # Unit ID is 1-indexed
            unit_id = idx + 1
            
            # Implement context window size functionality
            # For each meaning unit, show context from surrounding units
            context_info += f"Contextual Excerpts for Meaning Unit ID {unit_id}:\n"
            
            # Calculate context window size
            context_start = max(0, idx - config.context_size + 1)
            context_end = min(len(sample_data), idx + 1)
            
            # Add context items
            for context_idx in range(context_start, context_end):
                context_item = sample_data[context_idx]
                
                # Skip if it's not a valid item
                if not context_item:
                    continue
                    
                # Include ID and context fields
                context_info += f"ID: {context_idx + 1}\n"
                
                for field in context_fields:
                    if field in context_item:
                        context_info += f"{field}: {context_item[field]}\n"
                
                if content_field in context_item:
                    context_info += f"{context_item[content_field]}\n\n"
                else:
                    context_info += "[No content found in specified field]\n\n"
            
            # Current excerpt section - this is the actual meaning unit to code
            context_info += f"Current Excerpt For Coding (Meaning Unit ID {unit_id}):\n"
            
            for field in context_fields:
                if field in item:
                    context_info += f"{field}: {item[field]}\n"
            
            if content_field in item:
                context_info += f"Quote: {item[content_field]}\n\n"
            else:
                context_info += "Quote: [No content found in specified field]\n\n"
        
        # Final instruction based on coding mode
        final_instruction = ""
        if config.coding_mode == "deductive":
            final_instruction = "**Apply codes exclusively to the excerpt(s) provided above.**"
        else:
            final_instruction = "**Generate codes based on the excerpt(s) provided above using the guidelines.**"
        
        # Build the complete prompt
        full_prompt = f"{prompt_content}\n\n{codes_block}\n{context_info}\n{final_instruction}"
        
        return {"prompt": full_prompt}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Error generating prompt preview: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating prompt preview: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 