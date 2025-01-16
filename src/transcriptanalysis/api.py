# src/transcriptanalysis/api.py

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from transcriptanalysis.main import main
from transcriptanalysis.utils import load_config_from_dict  # Modify to load config from dict
from transcriptanalysis.config_schemas import ConfigModel
import importlib.resources
import logging
from pathlib import Path
import json
from pydantic import ValidationError
from typing import Optional, Dict
from uuid import uuid4

# Initialize FastAPI app
app = FastAPI(
    title="TranscriptAnalysis API",
    description="API for running the qualitative coding pipeline.",
    version="2.0.0"  # Updated version
)

# Configure logging for the API
logger = logging.getLogger("transcriptanalysis.api")
logger.setLevel(logging.DEBUG)  # Set to DEBUG or INFO as needed

# Create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)  # Adjust as needed

# Create formatter and add it to the handler
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
ch.setFormatter(formatter)

# Add the handler to the logger if not already present
if not logger.handlers:
    logger.addHandler(ch)

# In-memory store for job statuses and outputs
# Structure: { job_id: { "status": str, "output_path": Path, "error": Optional[str] } }
job_store: Dict[str, Dict] = {}

@app.get("/", summary="Root Endpoint", tags=["General"])
def read_root():
    """
    Root endpoint that returns a welcome message.
    """
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the TranscriptAnalysis API!"}

@app.post("/run-pipeline", summary="Run Qualitative Coding Pipeline", tags=["Pipeline"])
def run_pipeline(config: ConfigModel, background_tasks: BackgroundTasks):
    """
    Endpoint to trigger the qualitative coding pipeline with a user-provided configuration.
    
    The pipeline runs as a background task to allow concurrent executions.
    
    Args:
        config (ConfigModel): The configuration for the pipeline provided by the user.
    
    Returns:
        JSONResponse: Contains the unique job ID assigned to the pipeline execution.
    """
    try:
        # Validate and parse the user-provided configuration
        user_config = config  # Already validated by Pydantic
        
        # Generate a unique job ID
        job_id = str(uuid4())
        logger.info(f"Received request to run pipeline. Assigned Job ID: {job_id}")
        
        # Define a unique output folder for this job
        output_folder = Path("outputs") / job_id
        output_folder.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created output folder at '{output_folder}' for Job ID: {job_id}")
        
        # Initialize job status
        job_store[job_id] = {
            "status": "pending",
            "output_path": output_folder,
            "error": None
        }
        
        # Add the pipeline task as a background task
        background_tasks.add_task(execute_pipeline, job_id, user_config, output_folder)
        
        logger.info(f"Pipeline task for Job ID: {job_id} has been started.")
        
        # Return the job ID to the user
        return JSONResponse(content={"job_id": job_id, "message": "Pipeline has been started."}, status_code=202)
    
    except ValidationError as ve:
        logger.error(f"Configuration validation error: {ve}")
        raise HTTPException(status_code=400, detail=f"Configuration validation error: {ve}")
    except Exception as e:
        logger.error(f"Failed to start the pipeline: {e}")
        raise HTTPException(status_code=500, detail="Failed to start the pipeline due to an internal error.")

def execute_pipeline(job_id: str, config: ConfigModel, output_folder: Path):
    """
    Executes the pipeline in the background.
    
    Args:
        job_id (str): The unique identifier for the job.
        config (ConfigModel): The configuration for the pipeline.
        output_folder (Path): The directory where outputs will be stored.
    """
    try:
        # Update job status to running
        job_store[job_id]["status"] = "running"
        logger.info(f"Job ID: {job_id} is now running.")
        
        # Modify the configuration to set the output folder
        config_dict = config.dict()
        config_dict["output_folder"] = str(output_folder)
        
        # Load configuration from dict (modify the utility function accordingly)
        loaded_config = load_config_from_dict(config_dict)
        
        # Execute the main pipeline function
        main(loaded_config)  # Ensure that 'main' uses 'output_folder' from config
        
        # After successful execution, locate the generated JSON script
        script_file_path = output_folder / "generated_script.json"  # Adjust filename as needed
        
        if not script_file_path.exists():
            error_msg = f"Generated script '{script_file_path}' does not exist."
            logger.error(error_msg)
            job_store[job_id]["status"] = "failed"
            job_store[job_id]["error"] = error_msg
            return
        
        # Update job status to completed
        job_store[job_id]["status"] = "completed"
        logger.info(f"Job ID: {job_id} has completed successfully.")
    
    except Exception as e:
        logger.error(f"Pipeline execution failed for Job ID: {job_id} with error: {e}")
        job_store[job_id]["status"] = "failed"
        job_store[job_id]["error"] = str(e)

@app.get("/status/{job_id}", summary="Check Pipeline Status", tags=["Pipeline"])
def check_status(job_id: str):
    """
    Endpoint to check the status of a specific pipeline job.
    
    Args:
        job_id (str): The unique identifier of the job.
    
    Returns:
        JSONResponse: Contains the current status and any error messages.
    """
    try:
        if job_id not in job_store:
            logger.warning(f"Status check requested for unknown Job ID: {job_id}")
            raise HTTPException(status_code=404, detail="Job ID not found.")
        
        status_info = {
            "job_id": job_id,
            "status": job_store[job_id]["status"],
            "error": job_store[job_id]["error"]
        }
        
        logger.info(f"Status retrieved for Job ID: {job_id}: {status_info['status']}")
        return JSONResponse(content=status_info)
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Failed to retrieve status for Job ID: {job_id} with error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve pipeline status.")

@app.get("/output/{job_id}", summary="Retrieve Generated Output", tags=["Output"])
def get_output(job_id: str):
    """
    Endpoint to retrieve the generated JSON output of a completed job.
    
    Args:
        job_id (str): The unique identifier of the job.
    
    Returns:
        JSONResponse or FileResponse: The contents of the generated JSON script or an error message.
    """
    try:
        if job_id not in job_store:
            logger.warning(f"Output retrieval requested for unknown Job ID: {job_id}")
            raise HTTPException(status_code=404, detail="Job ID not found.")
        
        job_info = job_store[job_id]
        
        if job_info["status"] != "completed":
            logger.warning(f"Output requested for Job ID: {job_id} which is not completed yet.")
            raise HTTPException(status_code=400, detail="Job is not completed yet.")
        
        # Define the script file path
        script_file_path = job_info["output_path"] / "generated_script.json"  # Adjust filename as needed
        
        if not script_file_path.exists():
            logger.error(f"Generated script '{script_file_path}' does not exist for Job ID: {job_id}.")
            raise HTTPException(status_code=500, detail="Generated script not found.")
        
        logger.info(f"Output retrieved for Job ID: {job_id}.")
        return FileResponse(path=script_file_path, filename="generated_script.json", media_type='application/json')
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Failed to retrieve output for Job ID: {job_id} with error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve the generated script.")

@app.get("/reports/{job_id}/{report_name}", summary="Retrieve Validation Report", tags=["Reports"])
def get_validation_report(job_id: str, report_name: str):
    """
    Endpoint to retrieve a specific validation report by name for a particular job.
    
    Args:
        job_id (str): The unique identifier of the job.
        report_name (str): The name of the validation report file without the '.json' extension.
    
    Returns:
        JSONResponse or FileResponse: The contents of the validation report as JSON.
    """
    try:
        if job_id not in job_store:
            logger.warning(f"Report retrieval requested for unknown Job ID: {job_id}")
            raise HTTPException(status_code=404, detail="Job ID not found.")
        
        # Access the output folder for the job
        output_folder = job_store[job_id]["output_path"]
        
        # Define the report file path
        report_file_path = output_folder / f"{report_name}.json"
        
        if not report_file_path.exists():
            logger.warning(f"Validation report '{report_name}.json' not found for Job ID: {job_id}.")
            raise HTTPException(status_code=404, detail=f"Report '{report_name}.json' does not exist.")
        
        logger.info(f"Validation report '{report_name}.json' retrieved for Job ID: {job_id}.")
        return FileResponse(path=report_file_path, filename=f"{report_name}.json", media_type='application/json')
    
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Failed to retrieve validation report for Job ID: {job_id} with error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve the validation report.")

