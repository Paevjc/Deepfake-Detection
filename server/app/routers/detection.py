from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import uuid
import shutil
import subprocess
from tempfile import NamedTemporaryFile
from pathlib import Path
from typing import Optional
from app.services.detector import DeepfakeDetector
from app.models.response import DetectionResponse

# Create router for deepfake detection endpoints
router = APIRouter(tags=["Deepfake Detection"])
detector = DeepfakeDetector()

# Create temp directories if they don't exist
os.makedirs("app/tmp/uploads", exist_ok=True)
os.makedirs("app/tmp/processed", exist_ok=True)

# Rename the endpoint to avoid ad blockers
@router.post("/analyze-media", response_model=DetectionResponse)
async def detect_deepfake(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Detect if a video is a deepfake
    """
    # Validate file type
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")

    # Generate a unique filename to avoid collisions
    temp_file_id = str(uuid.uuid4())
    
    # Save uploaded file to temporary location
    with NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
        try:
            # Copy content from the uploaded file to our temp file
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name
            
            # Log the file save
            print(f"Saved uploaded file to {temp_file_path}, size: {os.path.getsize(temp_file_path)} bytes")
            
            # Perform detection on the video file
            try:
                result = detector.analyze_video(temp_file_path)
                
                # Clean up the temp file in the background after sending the response
                background_tasks.add_task(os.unlink, temp_file_path)
                
                return DetectionResponse(
                    filename=file.filename,
                    is_deepfake=result["is_deepfake"],
                    confidence=result["confidence"],
                    processing_time=result["processing_time"],
                    frames_analyzed=result["frames_analyzed"],
                    detection_areas=result["detection_areas"]
                )
            except Exception as e:
                # Log the specific error
                print(f"Error processing video: {str(e)}")
                import traceback
                traceback.print_exc()
                raise HTTPException(status_code=500, detail=f"Error processing video: {str(e)}")
        except Exception as e:
            # Make sure to clean up if there's an error
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
            raise HTTPException(status_code=500, detail=f"Error saving video: {str(e)}")
        finally:
            # Close the file
            file.file.close()

# Add a video conversion endpoint
@router.post("/convert-video")
async def convert_video(file: UploadFile = File(...)):
    """Convert video to web-compatible format"""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    # Generate unique filenames
    file_id = str(uuid.uuid4())
    input_filename = f"input_{file_id}{os.path.splitext(file.filename)[1]}"
    output_filename = f"output_{file_id}.mp4"
    
    input_path = os.path.join("/app/tmp/uploads", input_filename)
    output_path = os.path.join("/app/tmp/processed", output_filename)
    
    # Save the uploaded file
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")
    
    # Convert using FFmpeg
    try:
        cmd = [
            "ffmpeg", "-i", input_path,
            "-c:v", "libx264", "-preset", "fast", 
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            "-profile:v", "baseline", "-level", "3.0",
            "-b:v", "800k", "-c:a", "aac", "-bufsize", "1600k",
            output_path
        ]
        
        process = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            check=True
        )
    except subprocess.CalledProcessError as e:
        # Clean up input file
        os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"FFmpeg conversion failed: {e.stderr}")
    except Exception as e:
        # Clean up input file
        os.remove(input_path)
        raise HTTPException(status_code=500, detail=f"Error during conversion: {str(e)}")
    
    # Clean up input file if conversion succeeded
    os.remove(input_path)
    
    # Return the converted file
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"{os.path.splitext(file.filename)[0]}.mp4",
        background=FileResponse.background_task_to_remove_file(output_path)
    )

# Simple heartbeat endpoint to test connectivity
@router.get("/heartbeat")
async def heartbeat():
    return {"status": "ok"}

# For backward compatibility, keep the old endpoint but make it point to the new one
@router.post("/detect", response_model=DetectionResponse)
async def legacy_detect_deepfake(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Legacy endpoint that redirects to the new analyzer endpoint"""
    return await detect_deepfake(background_tasks=background_tasks, file=file)