import os
import time
import cv2
import numpy as np
import torch
from typing import Dict, List, Optional, Any

class DeepfakeDetector:
    """Service for detecting deepfakes in video files using the loaded Swin Transformer model"""
    
    def __init__(self, model_path: str = "app/model/best_swin_transformer_model.pth", device: str = None):
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
            
            # print(f"Swin Transformer model loaded successfully from {self.model_path}")

            dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
            try:
                with torch.no_grad():
                    output = model(dummy_input)
                    print(f"Model test output: {output}")
            except Exception as e:
                print(f"Model test failed: {e}")

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
    
    def _extract_frames(self, video_path, max_frames=30):
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
        
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return frames
        
        # Get total frame count
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames <= 0:
            print(f"Error: Video has no frames or frame count couldn't be determined")
            return frames
        
        # Calculate frame sampling interval
        interval = max(1, total_frames // max_frames)
        
        frame_count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % interval == 0:
                # Verify frame integrity
                if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                    frames.append(frame)
                else:
                    print(f"Warning: Invalid frame at position {frame_count}")
            
            frame_count += 1
        
        cap.release()
        print(f"Extracted {len(frames)} frames from video")
        return frames
    
    def _preprocess_frame(self, frame):
        """
        Preprocess a frame for model input
        
        Args:
            frame: Raw frame as numpy array
            
        Returns:
            Processed frame as PyTorch tensor
        """
        try:
            # Ensure frame is a valid numpy array
            if frame is None or not isinstance(frame, np.ndarray):
                print(f"Invalid frame type: {type(frame)}")
                return None
                
            # Check if frame has valid dimensions
            if frame.ndim != 3 or frame.shape[2] != 3:
                print(f"Invalid frame shape: {frame.shape}")
                return None
                
            # Resize to expected input size
            resized = cv2.resize(frame, (224, 224))
            
            # Convert to RGB if needed
            if frame.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                
            # Normalize pixel values
            normalized = resized / 255.0
            
            # Convert to PyTorch tensor and add batch dimension
            tensor = torch.from_numpy(normalized).float()
            tensor = tensor.permute(2, 0, 1)  # HWC to CHW format
            tensor = tensor.unsqueeze(0)  # Add batch dimension
            tensor = tensor.to(self.device)
            
            return tensor
        except Exception as e:
            print(f"Error preprocessing frame: {e}")
            return None
    
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
    
    def analyze_video(self, video_path):
        """
        Analyze a video file to detect if it's a deepfake
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Dictionary with detection results including confidence score
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
        
        # Verify the model is valid
        if not self.model or not hasattr(self.model, 'forward'):
            print(f"Error: Model not properly initialized")
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "frames_analyzed": len(frames),
                "detection_areas": []
            }
        
        # Process frames with the model
        predictions = []
        confidence_values = []
        fake_probs = []  # Store fake probabilities for debugging
        real_probs = []  # Store real probabilities for debugging
        valid_frames = 0
        
        # Use torch.no_grad() to disable gradient calculation for inference
        with torch.no_grad():
            for frame_idx, frame in enumerate(frames):
                # Convert frame to PyTorch tensor
                input_tensor = self._preprocess_frame(frame)
                
                # Skip invalid frames
                if input_tensor is None:
                    continue
                
                # Get model prediction
                try:
                    output = self.model(input_tensor)
                    
                    # Detailed logging of raw model outputs
                    probs = output.cpu().numpy()[0]
                    print(f"Frame {frame_idx} - Raw model output: {output}")
                    print(f"Frame {frame_idx} - Probabilities - Fake: {probs[0]:.6f}, Real: {probs[1]:.6f}")
                    
                    # Check for extremely small values that might be rounded to zero
                    if probs[1] < 0.0001:
                        print(f"WARNING: Frame {frame_idx} - Extremely low real probability: {probs[1]:.10f}")
                    
                    # Force softmax if needed (if your model doesn't already apply it)
                    # This helps ensure outputs are proper probabilities
                    import torch.nn.functional as F
                    softmax_probs = F.softmax(output, dim=1).cpu().numpy()[0]
                    print(f"Frame {frame_idx} - After softmax - Fake: {softmax_probs[0]:.6f}, Real: {softmax_probs[1]:.6f}")
                    
                    # Use the softmax probabilities for better calibration
                    real_prob = softmax_probs[1]  # Probability of being real
                    fake_prob = softmax_probs[0]  # Probability of being fake
                    
                    # Store both probabilities for debugging
                    real_probs.append(real_prob)
                    fake_probs.append(fake_prob)
                    
                    # Determine if deepfake (higher fake probability means it's more likely a deepfake)
                    pred = fake_prob > real_prob
                    
                    # Store the prediction and confidence
                    predictions.append(pred)
                    confidence_values.append(real_prob)  # Store real probability as our confidence
                    valid_frames += 1
                    
                except Exception as e:
                    print(f"Error during model inference: {e}")
                    continue
        
        # If no valid frames were processed, return default result
        if not valid_frames:
            return {
                "is_deepfake": False,
                "confidence": 0.0,
                "processing_time": time.time() - start_time,
                "frames_analyzed": len(frames),
                "detection_areas": []
            }
        
        # Calculate overall confidence
        avg_confidence = sum(confidence_values) / len(confidence_values)
        
        # Print detailed statistics about the frames analyzed
        print(f"========= Video Analysis Results =========")
        print(f"Frames analyzed: {valid_frames}")
        print(f"Average real probability: {avg_confidence:.6f}")
        print(f"Average fake probability: {sum(fake_probs) / len(fake_probs):.6f}")
        print(f"Real probability range: {min(real_probs):.6f} to {max(real_probs):.6f}")
        print(f"Fake probability range: {min(fake_probs):.6f} to {max(fake_probs):.6f}")
        print(f"Frames classified as fake: {sum(predictions)}/{len(predictions)}")
        print(f"=======================================")
        
        # Ensure confidence is within proper range (0-1)
        avg_confidence = max(0.0, min(1.0, avg_confidence))
        
        # Determine if the video is a deepfake based on authenticity confidence
        # Lower authentic confidence (< 0.5) means it's more likely a deepfake
        is_deepfake = avg_confidence < 0.5
        
        # Report the confidence of our prediction, not the authenticity score
        # If is_deepfake is True, we should report (1 - avg_confidence) as our confidence
        # If is_deepfake is False, we should report avg_confidence as our confidence
        prediction_confidence = 1.0 - avg_confidence if is_deepfake else avg_confidence
        
        # Calculate the processing time
        processing_time = time.time() - start_time
        
        # Create detection areas
        detection_areas = []
        for i, frame in enumerate(frames):
            if i < len(predictions) and predictions[i]:
                # If this frame is predicted as deepfake
                face_regions = self._detect_face_regions(frame)
                for region in face_regions:
                    detection_areas.append({
                        "frame_number": i,
                        "coordinates": region,
                        "confidence": float(1.0 - confidence_values[i])  # Deep fake confidence = 1 - authenticity
                    })
        
        # Final result with detailed information
        result = {
            "is_deepfake": is_deepfake,
            "confidence": float(prediction_confidence),  # Now represents prediction confidence
            "processing_time": processing_time,
            "frames_analyzed": valid_frames,
            "detection_areas": detection_areas
        }
        
        # Print the final detection result
        print(f"Final detection: {'DEEPFAKE' if is_deepfake else 'REAL'} with confidence {prediction_confidence:.6f}")
        
        return result