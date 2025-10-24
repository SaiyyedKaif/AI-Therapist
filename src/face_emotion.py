import cv2
import numpy as np
from deepface import DeepFace
import logging
from typing import Dict, List, Optional, Tuple

class FaceEmotionDetector:
    """
    Facial emotion detection using DeepFace library
    Detects emotions from camera feed in real-time
    """
    
    def __init__(self, model_name: str = 'emotion'):
        """
        Initialize the face emotion detector
        
        Args:
            model_name: DeepFace model to use for emotion detection
        """
        self.model_name = model_name
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect faces in the frame using Haar cascade
        
        Args:
            frame: Input video frame
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return faces
        except Exception as e:
            self.logger.error(f"Face detection error: {str(e)}")
            return []
    
    def detect_emotions(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect emotions from faces in the frame
        
        Args:
            frame: Input video frame
            
        Returns:
            List of emotion detection results for each face
        """
        try:
            # First detect faces
            faces = self.detect_faces(frame)
            
            if len(faces) == 0:
                return []
            
            results = []
            
            for (x, y, w, h) in faces:
                # Extract face region
                face_roi = frame[y:y+h, x:x+w]
                
                # Skip if face is too small
                if face_roi.shape[0] < 48 or face_roi.shape[1] < 48:
                    continue
                
                try:
                    # Use DeepFace for emotion detection
                    analysis = DeepFace.analyze(
                        face_roi,
                        actions=['emotion'],
                        enforce_detection=False,
                        silent=True
                    )
                    
                    if isinstance(analysis, list):
                        analysis = analysis[0]
                    
                    emotions = analysis.get('emotion', {})
                    
                    # Normalize emotion scores
                    total = sum(emotions.values())
                    if total > 0:
                        emotions = {k: v/total for k, v in emotions.items()}
                    
                    results.append({
                        'bbox': (x, y, w, h),
                        'emotions': emotions,
                        'dominant_emotion': analysis.get('dominant_emotion', 'neutral')
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Emotion analysis failed for face: {str(e)}")
                    # Fallback to neutral emotion
                    emotions = {emotion: 0.0 for emotion in self.emotion_labels}
                    emotions['neutral'] = 1.0
                    
                    results.append({
                        'bbox': (x, y, w, h),
                        'emotions': emotions,
                        'dominant_emotion': 'neutral'
                    })
            
            return results
            
        except Exception as e:
            self.logger.error(f"Emotion detection error: {str(e)}")
            return []
    
    def draw_emotions(self, frame: np.ndarray, face_emotions: List[Dict]) -> np.ndarray:
        """
        Draw emotion detection results on the frame
        
        Args:
            frame: Input video frame
            face_emotions: Emotion detection results
            
        Returns:
            Frame with drawn emotion annotations
        """
        try:
            frame_copy = frame.copy()
            
            for face_data in face_emotions:
                x, y, w, h = face_data['bbox']
                emotions = face_data['emotions']
                dominant_emotion = face_data['dominant_emotion']
                
                # Draw bounding box
                cv2.rectangle(frame_copy, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Draw dominant emotion label
                label_text = f"{dominant_emotion}: {emotions.get(dominant_emotion, 0):.2f}"
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                
                # Background rectangle for label
                cv2.rectangle(frame_copy, (x, y - 25), (x + label_size[0], y), (0, 255, 0), -1)
                cv2.putText(frame_copy, label_text, (x, y - 8), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
                
                # Draw emotion confidence bars
                bar_width = w // len(emotions)
                for i, (emotion, confidence) in enumerate(emotions.items()):
                    if confidence > 0.1:  # Only show significant emotions
                        bar_x = x + i * bar_width
                        bar_height = int(confidence * 50)  # Scale to pixel height
                        cv2.rectangle(frame_copy, (bar_x, y + h + 5), 
                                    (bar_x + bar_width - 2, y + h + 5 + bar_height), 
                                    (255, 255, 0), -1)
                        
                        # Emotion label
                        cv2.putText(frame_copy, emotion[:3].upper(), 
                                   (bar_x, y + h + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
            
            return frame_copy
            
        except Exception as e:
            self.logger.error(f"Error drawing emotions: {str(e)}")
            return frame
    
    def get_dominant_emotion(self, face_emotions: List[Dict]) -> Optional[Dict]:
        """
        Get the dominant emotion across all detected faces
        
        Args:
            face_emotions: Emotion detection results
            
        Returns:
            Dictionary with dominant emotion and confidence
        """
        if not face_emotions:
            return None
        
        try:
            # Average emotions across all faces
            combined_emotions = {}
            
            for face_data in face_emotions:
                emotions = face_data['emotions']
                for emotion, confidence in emotions.items():
                    if emotion in combined_emotions:
                        combined_emotions[emotion] += confidence
                    else:
                        combined_emotions[emotion] = confidence
            
            # Normalize by number of faces
            num_faces = len(face_emotions)
            combined_emotions = {k: v/num_faces for k, v in combined_emotions.items()}
            
            # Find dominant emotion
            dominant_emotion = max(combined_emotions.items(), key=lambda x: x[1])
            
            return {
                'emotion': dominant_emotion[0],
                'confidence': dominant_emotion[1],
                'all_emotions': combined_emotions
            }
            
        except Exception as e:
            self.logger.error(f"Error getting dominant emotion: {str(e)}")
            return None
    
    def cleanup(self):
        """Clean up resources"""
        # No specific cleanup needed for DeepFace
        pass
