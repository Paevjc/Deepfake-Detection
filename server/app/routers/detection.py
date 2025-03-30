# Deepfake detection endpoints

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import os
import uuid
import shutil
from tempfile import NamedTemporaryFile
from app.services.detector import DeepfakeDetector
from app.models.response import DetectionResponse

router = APIRouter(tags=["Deepfake Detection"])

detector = DeepfakeDetector()

@router.post("/detect", response_model=DetectionResponse)
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