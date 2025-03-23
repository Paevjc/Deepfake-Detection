import os
import time
import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Any

class DeepfakeDetector:
    """Service for detecting deepfakes in video files using the loaded Swin Transformer model"""
    
    def __init__(self, model_path: str = "model/best_swin_transformer_model.pth", device: str = None):
        """
        Initialize the deepfake detector with the PyTorch Swin Transformer model
        
        Args:
            model_path: Path to the PyTorch model file (.pth)
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.model_path = model_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        self.model = self._load_model()
    
    def _load_model(self) -> Any:
        """Load the PyTorch Swin Transformer model from disk"""
        try:
            # Import required libraries
            import timm
            import torch.nn as nn
            
            # Define Swin Transformer model architecture
            class SwinTransformerClassifier(nn.Module):
                def __init__(self, num_classes=2):
                    super(SwinTransformerClassifier, self).__init__()
                    self.model = timm.create_model("swin_base_patch4_window7_224", pretrained=True)

                    # Freeze initial layers
                    for name, param in self.model.named_parameters():
                        if "layers.0" in name or "layers.1" in name:
                            param.requires_grad = False

                    # Unfreeze last few layers
                    for param in self.model.head.parameters():
                        param.requires_grad = True

                    # Get correct input size for classifier head
                    in_features = self.model.num_features
                    
                    # Fix: Ensure correct feature processing
                    self.pooling = nn.AdaptiveAvgPool1d(1)
                    self.dropout = nn.Dropout(0.3)
                    self.fc = nn.Linear(in_features, num_classes)

                def forward(self, x):
                    x = self.model.forward_features(x)
                    x = x.mean(dim=[1, 2])
                    x = x.view(x.shape[0], -1)
                    x = self.dropout(x)
                    x = self.fc(x)
                    return x
            
            # Initialize the model architecture
            model = SwinTransformerClassifier(num_classes=2)
            
            # Load the trained weights
            model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            
            # Set the model to evaluation mode
            model.eval()
            model = model.to(self.device)
            
            print(f"Swin Transformer model loaded successfully from {self.model_path}")
            return model
            
        except Exception as e:
            print(f"Error loading Swin Transformer model: {e}")
            # Return a placeholder model for development/testing
            return self._get_placeholder_model()
    
    def _get_placeholder_model(self):
        """
        Create a placeholder model for testing when the real model isn't available
        This is just for development - replace with actual model loading logic
        """
        try:
            import timm
            import torch.nn as nn
            
            # Create a simple classifier using timm's base model but with minimal setup
            class PlaceholderSwinModel(nn.Module):
                def __init__(self):
                    super().__init__()
                    # Use the smallest Swin model for faster performance during testing
                    self.backbone = timm.create_model("swin_tiny_patch4_window7_224", pretrained=False)
                    self.fc = nn.Linear(768, 2)  # Swin tiny has 768 features
                    self.softmax = nn.Softmax(dim=1)
                
                def forward(self, x):
                    with torch.no_grad():
                        # Extract features
                        features = self.backbone.forward_features(x)
                        # Global pooling
                        features = features.mean(dim=[1, 2])
                        # Classification
                        logits = self.fc(features)
                        # Return probabilities
                        return self.softmax(logits)
            
            model = PlaceholderSwinModel().to(self.device)
            model.eval()
            return model
            
        except Exception as e:
            print(f"Error creating placeholder model: {e}")
            # Very basic fallback
            # return BasicPlaceholderModel()
    
    def _extract_frames(self, video_path: str, max_frames: int = 30) -> List[np.ndarray]:
        """
        Extract frames from a video file
        
        Args:
            video_path: Path to the video file
            max_frames: Maximum number of frames to extract
            
        Returns:
            List of frames as numpy arrays
        """
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame sampling interval
        interval = max(1, total_frames // max_frames)
        
        frame_count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                # Preprocess frame for the model
                # This preprocessing depends on your model requirements
                processed_frame = self._preprocess_frame(frame)
                frames.append(processed_frame)
                
            frame_count += 1
                
        cap.release()
        return frames
    
    def _preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocess a frame for PyTorch model input
        
        Args:
            frame: Raw frame as numpy array
            
        Returns:
            Processed frame as PyTorch tensor ready for model input
        """
        # Resize to expected input size (adjust based on your model)
        resized = cv2.resize(frame, (224, 224))
        
        # Convert to RGB if needed (OpenCV uses BGR by default)
        if len(resized.shape) == 3 and resized.shape[2] == 3:
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
        # Normalize pixel values (typical PyTorch normalization)
        normalized = resized / 255.0
        
        # Convert to PyTorch tensor and add batch dimension
        # PyTorch expects: [batch_size, channels, height, width]
        tensor = torch.from_numpy(normalized).float()
        
        # Transpose from (H, W, C) to (C, H, W) format for PyTorch
        tensor = tensor.permute(2, 0, 1)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        # Move tensor to the right device (GPU or CPU)
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _detect_face_regions(self, frame: np.ndarray) -> List[Dict[str, int]]:
        """
        Detect face regions in a frame
        This is a placeholder - replace with actual face detection logic
        
        Args:
            frame: Video frame as numpy array
            
        Returns:
            List of face bounding boxes as dictionaries
        """
        # This is a placeholder - in a real implementation you would use
        # a face detection model like dlib, OpenCV's face detector, or
        # a more advanced deep learning model
        
        # For this example, we'll return a dummy face region
        # in the center of the frame
        h, w = frame.shape[0], frame.shape[1]
        center_x, center_y = w // 2, h // 2
        
        return [
            {
                "x1": max(0, center_x - w//4),
                "y1": max(0, center_y - h//4),
                "x2": min(w, center_x + w//4),
                "y2": min(h, center_y + h//4)
            }
        ]
    
    def analyse_video(self, video_path: str) -> Dict:
        """
        Analyze a video file to detect if it's a deepfake
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with detection results
        """
        start_time = time.time()
        
        # Extract frames from the video
        frames = self._extract_frames(video_path)
        
        if not frames:
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "frames_analyzed": 0,
                "detection_areas": []
            }
        
        # Process frames with the PyTorch model
        predictions = []
        confidence_values = []
        
        # Use torch.no_grad() to disable gradient calculation for inference
        with torch.no_grad():
            for frame in frames:
                # Convert frame to PyTorch tensor
                input_tensor = self._preprocess_frame(frame)
                
                # Get model prediction
                output = self.model(input_tensor)
                
                # Get probabilities (output is usually [real_prob, fake_prob])
                probs = output.cpu().numpy()[0]
                
                # Determine if deepfake (index 1 typically represents deepfake probability)
                pred = np.argmax(probs) == 1
                
                # Store the prediction and confidence
                predictions.append(pred)
                confidence_values.append(probs[1])  # Deepfake probability
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_values) / len(confidence_values)
        
        # Determine if the video is a deepfake
        # You may need to adjust the threshold based on your model
        is_deepfake = avg_confidence > 0.5
        
        # Identify regions in frames where deepfake artifacts were detected
        detection_areas = []
        for i, (pred, conf) in enumerate(zip(predictions, confidence_values)):
            if pred and conf > 0.6:  # Only include higher confidence detections
                # In a real implementation, you'd use face or manipulation detection
                # to determine the exact areas affected
                face_regions = self._detect_face_regions(frames[i])
                
                for region in face_regions:
                    detection_areas.append({
                        "frame_number": i,
                        "coordinates": region,
                        "confidence": conf
                    })
        
        processing_time = time.time() - start_time
        
        return {
            "is_deepfake": is_deepfake,
            "confidence": avg_confidence,
            "processing_time": processing_time,
            "frames_analyzed": len(frames),
            "detection_areas": detection_areas
        }