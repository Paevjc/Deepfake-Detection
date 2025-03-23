# Response data models

from pydantic import BaseModel
from typing import List, Dict, Optional

class DetectionArea(BaseModel):
    '''
    Area in the video where deepfake artifacts were detected
    '''
    frame_number: int
    coordinates: Dict[str, int]
    confidence: float

class DetectionResponse(BaseModel):
    '''
    Response model for deepfake detection results
    '''
    filename: str
    is_deepfake: bool
    confidence: float
    processing_time: float
    frames_analysed: int
    detection_areas: Optional[List[DetectionArea]] = None