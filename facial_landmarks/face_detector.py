"""
Fast face detection using MediaPipe and OpenCV.
Provides high-performance face detection optimized for real-time applications.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional, NamedTuple
from dataclasses import dataclass


@dataclass
class BoundingBox:
    """Bounding box with convenient attribute access."""
    x: float
    y: float
    width: float
    height: float


@dataclass
class FaceDetection:
    """Data class representing a detected face."""
    bbox: BoundingBox
    confidence: float
    keypoints: Optional[List[Tuple[int, int]]] = None


class FaceDetector:
    """Fast face detector using MediaPipe."""
    
    def __init__(self, 
                 min_detection_confidence: float = 0.7,
                 model_selection: int = 0):
        """
        Initialize the face detector.
        
        Args:
            min_detection_confidence: Minimum confidence threshold for detection
            model_selection: 0 for short-range model (< 2m), 1 for full-range model
        """
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )
    
    def detect_faces(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Detect faces in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of FaceDetection objects
        """
        # Validate input
        if image is None or image.size == 0:
            raise ValueError("Invalid image: image is None or empty")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Invalid image: expected 3-channel BGR image")
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform detection
        results = self.face_detection.process(rgb_image)
        
        detections = []
        if results.detections:
            for detection in results.detections:
                # Get normalized bounding box (keep as normalized coordinates)
                bbox = detection.location_data.relative_bounding_box
                
                # Get confidence score
                confidence = detection.score[0]
                
                # Get key points (6 points: right eye, left eye, nose tip, mouth center, right ear, left ear)
                keypoints = []
                if detection.location_data.relative_keypoints:
                    h, w, _ = image.shape
                    for keypoint in detection.location_data.relative_keypoints:
                        kp_x = int(keypoint.x * w)
                        kp_y = int(keypoint.y * h)
                        keypoints.append((kp_x, kp_y))
                
                detections.append(FaceDetection(
                    bbox=BoundingBox(x=bbox.xmin, y=bbox.ymin, width=bbox.width, height=bbox.height),
                    confidence=confidence,
                    keypoints=keypoints if keypoints else None
                ))
        
        return detections
    
    def draw_detections(self, 
                       image: np.ndarray, 
                       detections: List[FaceDetection],
                       draw_keypoints: bool = True) -> np.ndarray:
        """
        Draw face detections on image.
        
        Args:
            image: Input image
            detections: List of face detections
            draw_keypoints: Whether to draw facial keypoints
            
        Returns:
            Image with drawn detections
        """
        result_image = image.copy()
        
        for detection in detections:
            x, y, w, h = detection.bbox.x, detection.bbox.y, detection.bbox.width, detection.bbox.height
            
            # Draw bounding box
            cv2.rectangle(result_image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
            
            # Draw confidence score
            confidence_text = f"{detection.confidence:.2f}"
            cv2.putText(result_image, confidence_text, (int(x), int(y) - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw keypoints
            if draw_keypoints and detection.keypoints:
                for kp_x, kp_y in detection.keypoints:
                    cv2.circle(result_image, (kp_x, kp_y), 3, (255, 0, 0), -1)
        
        return result_image
    
    def get_face_crops(self, 
                      image: np.ndarray, 
                      detections: List[FaceDetection],
                      padding: float = 0.2) -> List[Tuple[np.ndarray, Tuple[int, int, int, int]]]:
        """
        Extract face crops from image based on detections.
        
        Args:
            image: Input image
            detections: List of face detections
            padding: Padding factor to add around face bounding box
            
        Returns:
            List of (cropped_face, original_bbox) tuples
        """
        crops = []
        h, w = image.shape[:2]
        
        for detection in detections:
            x, y, face_w, face_h = detection.bbox.x, detection.bbox.y, detection.bbox.width, detection.bbox.height
            
            # Add padding
            pad_w = int(face_w * padding)
            pad_h = int(face_h * padding)
            
            # Calculate padded coordinates
            x1 = max(0, int(x) - pad_w)
            y1 = max(0, int(y) - pad_h)
            x2 = min(w, int(x + face_w) + pad_w)
            y2 = min(h, int(y + face_h) + pad_h)
            
            # Crop face
            face_crop = image[y1:y2, x1:x2]
            crops.append((face_crop, (x1, y1, x2 - x1, y2 - y1)))
        
        return crops
    
    def bbox_to_pixels(self, bbox: BoundingBox, image_shape: Tuple[int, int]) -> BoundingBox:
        """
        Convert normalized bbox to pixel coordinates.
        
        Args:
            bbox: Normalized bounding box
            image_shape: (height, width) of image
            
        Returns:
            BoundingBox in pixel coordinates
        """
        h, w = image_shape
        return BoundingBox(
            x=bbox.x * w,
            y=bbox.y * h,
            width=bbox.width * w,
            height=bbox.height * h
        )
    
    def draw_faces(self, image: np.ndarray, faces: List[FaceDetection], 
                   bbox_color: Tuple[int, int, int] = (0, 255, 0),
                   text_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
        """
        Draw face bounding boxes on image.
        
        Args:
            image: Input image
            faces: List of face detections
            bbox_color: Color for bounding box
            text_color: Color for text
            
        Returns:
            Image with drawn faces
        """
        result_image = image.copy()
        
        for face in faces:
            x, y, w, h = int(face.bbox.x), int(face.bbox.y), int(face.bbox.width), int(face.bbox.height)
            
            # Draw bounding box
            cv2.rectangle(result_image, (x, y), (x + w, y + h), bbox_color, 2)
            
            # Draw confidence text
            confidence_text = f"{face.confidence:.2f}"
            cv2.putText(result_image, confidence_text, (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)
        
        return result_image
    
    def get_face_region(self, image: np.ndarray, face: FaceDetection) -> np.ndarray:
        """
        Extract face region from image.
        
        Args:
            image: Input image
            face: Face detection
            
        Returns:
            Cropped face region
        """
        h, w = image.shape[:2]
        x, y, face_w, face_h = int(face.bbox.x), int(face.bbox.y), int(face.bbox.width), int(face.bbox.height)
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w, x + face_w)
        y2 = min(h, y + face_h)
        
        # Return cropped region
        return image[y1:y2, x1:x2]
    
    def filter_faces_by_confidence(self, faces: List[FaceDetection], 
                                  threshold: float) -> List[FaceDetection]:
        """
        Filter faces by confidence threshold.
        
        Args:
            faces: List of face detections
            threshold: Minimum confidence threshold
            
        Returns:
            Filtered list of faces
        """
        return [face for face in faces if face.confidence >= threshold]
    
    def get_largest_face(self, faces: List[FaceDetection]) -> Optional[FaceDetection]:
        """
        Get the largest detected face.
        
        Args:
            faces: List of face detections
            
        Returns:
            Largest face or None if no faces
        """
        if not faces:
            return None
            
        return max(faces, key=lambda f: f.bbox.width * f.bbox.height)
    
    def get_face_area(self, face: FaceDetection) -> float:
        """
        Calculate area of face bounding box.
        
        Args:
            face: Face detection
            
        Returns:
            Area of face bounding box
        """
        return face.bbox.width * face.bbox.height
    
    def get_detection_statistics(self, faces: List[FaceDetection]) -> dict:
        """
        Get statistics about face detections.
        
        Args:
            faces: List of face detections
            
        Returns:
            Dictionary with detection statistics
        """
        if not faces:
            return {
                'count': 0,
                'avg_confidence': 0.0,
                'max_confidence': 0.0,
                'min_confidence': 0.0,
                'avg_area': 0.0
            }
            
        confidences = [f.confidence for f in faces]
        areas = [self.get_face_area(f) for f in faces]
        
        return {
            'count': len(faces),
            'avg_confidence': sum(confidences) / len(confidences),
            'max_confidence': max(confidences),
            'min_confidence': min(confidences),
            'avg_area': sum(areas) / len(areas)
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
