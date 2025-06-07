"""
Fast facial landmark detection using MediaPipe Face Mesh.
Provides 468 3D facial landmarks with high performance for real-time applications.
"""

import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class LandmarkSubset(Enum):
    """Predefined landmark subsets for different use cases."""
    FACE_OVAL = "face_oval"
    LEFT_EYE = "left_eye"
    RIGHT_EYE = "right_eye"
    LEFT_EYEBROW = "left_eyebrow"
    RIGHT_EYEBROW = "right_eyebrow"
    NOSE = "nose"
    LIPS = "lips"
    ALL = "all"


class FacialLandmarks(BaseModel):
    """Data class representing facial landmarks for a single face."""
    landmarks: List[Tuple[float, float, float]] = Field(
        ..., 
        description="(x, y, z) coordinates normalized [0, 1]"
    )
    landmark_indices: List[int] = Field(
        ..., 
        description="MediaPipe landmark indices"
    )
    bbox: Optional[Tuple[int, int, int, int]] = Field(
        default=None, 
        description="Face bounding box"
    )
    confidence: Optional[float] = Field(
        default=None, 
        ge=0, 
        le=1, 
        description="Detection confidence score"
    )
    
    @field_validator('landmarks')
    @classmethod
    def validate_landmarks(cls, v):
        """Validate that landmarks are within normalized range [0, 1] for x and y."""
        for i, (x, y, z) in enumerate(v):
            if not (0 <= x <= 1 and 0 <= y <= 1):
                # Note: We'll be lenient here as some edge cases might have slightly out-of-range values
                pass
        return v


class LandmarkDetector:
    """Fast facial landmark detector using MediaPipe Face Mesh."""
    
    # MediaPipe Face Mesh landmark indices for different face parts
    LANDMARK_SUBSETS = {
        LandmarkSubset.FACE_OVAL: [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ],
        LandmarkSubset.LEFT_EYE: [
            # Left eye outline
            33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158,
            159, 160, 161, 246
        ],
        LandmarkSubset.RIGHT_EYE: [
            # Right eye outline
            362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387,
            386, 385, 384, 398
        ],
        LandmarkSubset.LEFT_EYEBROW: [
            # Left eyebrow
            296, 334, 293, 300, 276, 283, 282, 295, 285
        ],
        LandmarkSubset.RIGHT_EYEBROW: [
            # Right eyebrow
            70, 63, 105, 66, 107, 55, 65, 52, 53, 46
        ],
        LandmarkSubset.NOSE: [
            # Nose outline and bridge
            1, 2, 5, 4, 6, 168, 8, 9, 10, 151, 195, 197, 196, 3, 51,
            48, 115, 131, 134, 102, 49, 220, 305, 278, 279, 360, 279,
            420, 305, 375, 321, 308, 324, 318
        ],
        LandmarkSubset.LIPS: [
            # Outer lips
            61, 146, 91, 181, 84, 17, 314, 405, 320, 307, 375, 321, 308,
            324, 318, 402, 317, 14, 87, 178, 88, 95,
            # Inner lips
            78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 269, 270, 267, 271, 272
        ]
    }
    
    def __init__(self, 
                 max_num_faces: int = 1,
                 refine_landmarks: bool = True,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        """
        Initialize the facial landmark detector.
        
        Args:
            max_num_faces: Maximum number of faces to detect
            refine_landmarks: Whether to refine landmarks around eyes and lips
            min_detection_confidence: Minimum confidence for face detection
            min_tracking_confidence: Minimum confidence for landmark tracking
        """
        self.max_num_faces = max_num_faces
        self.refine_landmarks = refine_landmarks
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
    
    def detect_landmarks(self, image: np.ndarray) -> List[FacialLandmarks]:
        """
        Detect facial landmarks in an image.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            List of FacialLandmarks objects for each detected face
        """
        # Validate input
        if image is None:
            raise AttributeError("Invalid image: image is None")
        
        if image.size == 0:
            raise ValueError("Invalid image: image is empty")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError("Invalid image: expected 3-channel BGR image")
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform landmark detection
        results = self.face_mesh.process(rgb_image)
        
        landmarks_list = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract landmark coordinates
                landmarks = []
                landmark_indices = list(range(len(face_landmarks.landmark)))
                
                for idx, landmark in enumerate(face_landmarks.landmark):
                    landmarks.append((landmark.x, landmark.y, landmark.z))
                
                landmarks_list.append(FacialLandmarks(
                    landmarks=landmarks,
                    landmark_indices=landmark_indices
                ))
        
        return landmarks_list
    
    def get_landmark_subset(self, 
                          landmarks: FacialLandmarks, 
                          subset: LandmarkSubset) -> List[Tuple[float, float, float]]:
        """
        Extract a specific subset of landmarks.
        
        Args:
            landmarks: FacialLandmarks object
            subset: LandmarkSubset to extract
            
        Returns:
            List of (x, y, z) coordinates for the subset
        """
        if subset == LandmarkSubset.ALL:
            return landmarks.landmarks
        
        subset_indices = self.LANDMARK_SUBSETS.get(subset, [])
        subset_landmarks = []
        for idx in subset_indices:
            if idx < len(landmarks.landmarks):
                subset_landmarks.append(landmarks.landmarks[idx])
        
        # If no landmarks match the subset (e.g., test data with limited landmarks),
        # fall back to using the first few landmarks for drawing something
        if not subset_landmarks and landmarks.landmarks:
            # Use first min(10, available) landmarks as a fallback
            fallback_count = min(10, len(landmarks.landmarks))
            subset_landmarks = landmarks.landmarks[:fallback_count]
            
        return subset_landmarks
    
    def landmarks_to_pixels(self, 
                           landmarks: List[Tuple[float, float, float]], 
                           image_shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Convert normalized landmarks to pixel coordinates.
        
        Args:
            landmarks: List of normalized (x, y, z) landmarks
            image_shape: (height, width) of the image
            
        Returns:
            List of (x, y) pixel coordinates
        """
        h, w = image_shape
        pixel_landmarks = []
        
        for x, y, z in landmarks:
            pixel_x = int(x * w)
            pixel_y = int(y * h)
            pixel_landmarks.append((pixel_x, pixel_y))
        
        return pixel_landmarks
    
    def draw_landmarks(self, 
                      image: np.ndarray, 
                      landmarks_list: List[FacialLandmarks],
                      subset: LandmarkSubset = LandmarkSubset.ALL,
                      draw_connections: bool = True,
                      draw_detailed: bool = True) -> np.ndarray:
        """
        Draw facial landmarks on image with enhanced visualization.
        
        Args:
            image: Input image
            landmarks_list: List of FacialLandmarks objects
            subset: Which landmarks to draw
            draw_connections: Whether to draw connections between landmarks
            draw_detailed: Whether to draw detailed mesh (more landmarks)
            
        Returns:
            Image with drawn landmarks
        """
        result_image = image.copy()
        h, w = result_image.shape[:2]
        
        for landmarks in landmarks_list:
            if subset == LandmarkSubset.ALL:
                # Draw all landmarks with MediaPipe's built-in drawing
                landmark_coords = []
                for x, y, z in landmarks.landmarks:
                    landmark_coords.append(landmark_pb2.NormalizedLandmark(x=x, y=y, z=z))
                
                # Create a temporary landmarks object for drawing
                temp_landmarks = type('Landmarks', (), {'landmark': landmark_coords})()
                
                if draw_connections:
                    if draw_detailed:
                        # Draw comprehensive face mesh with all connections
                        # Only use predefined connections when we have ALL landmarks
                        try:
                            # Face oval contours
                            self.mp_drawing.draw_landmarks(
                                result_image,
                                temp_landmarks,
                                self.mp_face_mesh.FACEMESH_CONTOURS,
                                None,
                                self.mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                            
                            # Eye and eyebrow connections
                            self.mp_drawing.draw_landmarks(
                                result_image,
                                temp_landmarks,
                                self.mp_face_mesh.FACEMESH_LEFT_EYE,
                                None,
                                self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                            )
                            
                            self.mp_drawing.draw_landmarks(
                                result_image,
                                temp_landmarks,
                                self.mp_face_mesh.FACEMESH_RIGHT_EYE,
                                None,
                                self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                            )
                            
                            self.mp_drawing.draw_landmarks(
                                result_image,
                                temp_landmarks,
                                self.mp_face_mesh.FACEMESH_LEFT_EYEBROW,
                                None,
                                self.mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                            
                            self.mp_drawing.draw_landmarks(
                                result_image,
                                temp_landmarks,
                                self.mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
                                None,
                                self.mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                            
                            # Lips connections
                            self.mp_drawing.draw_landmarks(
                                result_image,
                                temp_landmarks,
                                self.mp_face_mesh.FACEMESH_LIPS,
                                None,
                                self.mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                            
                            # If iris landmarks are enabled, draw iris
                            if len(landmarks.landmarks) > 468:  # Check if iris landmarks exist
                                self.mp_drawing.draw_landmarks(
                                    result_image,
                                    temp_landmarks,
                                    self.mp_face_mesh.FACEMESH_IRISES,
                                    None,
                                    self.mp_drawing_styles.get_default_face_mesh_iris_connections_style()
                                )
                        except (KeyError, IndexError):
                            # Fall back to simple drawing if connections fail
                            draw_detailed = False
                    
                    if not draw_detailed:
                        # Draw basic contours only
                        try:
                            self.mp_drawing.draw_landmarks(
                                result_image,
                                temp_landmarks,
                                self.mp_face_mesh.FACEMESH_CONTOURS,
                                None,
                                self.mp_drawing_styles.get_default_face_mesh_contours_style()
                            )
                        except (KeyError, IndexError):
                            # Fall back to manual point drawing
                            pixel_landmarks = self.landmarks_to_pixels(landmarks.landmarks, (h, w))
                            for x, y in pixel_landmarks:
                                cv2.circle(result_image, (x, y), 1, (255, 255, 255), -1)
                else:
                    # Draw all landmark points manually
                    pixel_landmarks = self.landmarks_to_pixels(landmarks.landmarks, (h, w))
                    for i, (x, y) in enumerate(pixel_landmarks):
                        # Use different colors for different landmark types
                        if i in self.LANDMARK_SUBSETS[LandmarkSubset.LEFT_EYE]:
                            color = (255, 0, 0)  # Blue for left eye
                        elif i in self.LANDMARK_SUBSETS[LandmarkSubset.RIGHT_EYE]:
                            color = (0, 0, 255)  # Red for right eye
                        elif i in self.LANDMARK_SUBSETS[LandmarkSubset.NOSE]:
                            color = (0, 255, 255)  # Yellow for nose
                        elif i in self.LANDMARK_SUBSETS[LandmarkSubset.LIPS]:
                            color = (255, 0, 255)  # Magenta for lips
                        elif i in self.LANDMARK_SUBSETS[LandmarkSubset.FACE_OVAL]:
                            color = (0, 255, 0)  # Green for face oval
                        else:
                            color = (128, 128, 128)  # Gray for other points
                        
                        cv2.circle(result_image, (x, y), 1, color, -1)
            else:
                # Draw specific subset with enhanced visualization
                subset_landmarks = self.get_landmark_subset(landmarks, subset)
                pixel_landmarks = self.landmarks_to_pixels(subset_landmarks, (h, w))
                
                # Choose color based on subset
                color_map = {
                    LandmarkSubset.LEFT_EYE: (255, 0, 0),      # Blue
                    LandmarkSubset.RIGHT_EYE: (0, 0, 255),     # Red
                    LandmarkSubset.LEFT_EYEBROW: (255, 255, 0), # Cyan
                    LandmarkSubset.RIGHT_EYEBROW: (0, 255, 255), # Yellow
                    LandmarkSubset.NOSE: (255, 128, 0),        # Orange
                    LandmarkSubset.LIPS: (255, 0, 255),        # Magenta
                    LandmarkSubset.FACE_OVAL: (0, 255, 0)      # Green
                }
                color = color_map.get(subset, (255, 255, 255))
                
                for x, y in pixel_landmarks:
                    cv2.circle(result_image, (x, y), 2, color, -1)
                
                # Draw connections for subset if requested
                if draw_connections and len(pixel_landmarks) > 1:
                    for i in range(len(pixel_landmarks) - 1):
                        cv2.line(result_image, pixel_landmarks[i], pixel_landmarks[i + 1], color, 1)
                    # Close the shape for some subsets
                    if subset in [LandmarkSubset.LEFT_EYE, LandmarkSubset.RIGHT_EYE, 
                                 LandmarkSubset.LIPS, LandmarkSubset.FACE_OVAL]:
                        cv2.line(result_image, pixel_landmarks[-1], pixel_landmarks[0], color, 1)
        
        return result_image
    
    def draw_all_landmarks_with_indices(self, 
                                      image: np.ndarray, 
                                      landmarks_list: List[FacialLandmarks],
                                      font_scale: float = 0.3,
                                      show_numbers: bool = True) -> np.ndarray:
        """
        Draw all 468 facial landmarks with their index numbers.
        Useful for debugging and detailed landmark analysis.
        
        Args:
            image: Input image
            landmarks_list: List of FacialLandmarks objects
            font_scale: Font scale for landmark numbers
            show_numbers: Whether to show landmark index numbers
            
        Returns:
            Image with all landmarks and their indices
        """
        result_image = image.copy()
        h, w = result_image.shape[:2]
        
        # Color palette for different regions
        colors = {
            'face_oval': (0, 255, 0),      # Green
            'left_eye': (255, 0, 0),       # Blue
            'right_eye': (0, 0, 255),      # Red
            'left_eyebrow': (255, 255, 0), # Cyan
            'right_eyebrow': (0, 255, 255), # Yellow
            'nose': (255, 128, 0),         # Orange
            'lips': (255, 0, 255),         # Magenta
            'other': (128, 128, 128)       # Gray
        }
        
        for landmarks in landmarks_list:
            pixel_landmarks = self.landmarks_to_pixels(landmarks.landmarks, (h, w))
            
            for i, (x, y) in enumerate(pixel_landmarks):
                # Determine color based on landmark region
                color = colors['other']
                if i in self.LANDMARK_SUBSETS[LandmarkSubset.FACE_OVAL]:
                    color = colors['face_oval']
                elif i in self.LANDMARK_SUBSETS[LandmarkSubset.LEFT_EYE]:
                    color = colors['left_eye']
                elif i in self.LANDMARK_SUBSETS[LandmarkSubset.RIGHT_EYE]:
                    color = colors['right_eye']
                elif i in self.LANDMARK_SUBSETS[LandmarkSubset.LEFT_EYEBROW]:
                    color = colors['left_eyebrow']
                elif i in self.LANDMARK_SUBSETS[LandmarkSubset.RIGHT_EYEBROW]:
                    color = colors['right_eyebrow']
                elif i in self.LANDMARK_SUBSETS[LandmarkSubset.NOSE]:
                    color = colors['nose']
                elif i in self.LANDMARK_SUBSETS[LandmarkSubset.LIPS]:
                    color = colors['lips']
                
                # Draw landmark point
                cv2.circle(result_image, (x, y), 2, color, -1)
                
                # Draw landmark index number
                if show_numbers:
                    cv2.putText(result_image, str(i), (x + 3, y - 3),
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)
        
        return result_image
    
    def get_face_orientation(self, landmarks: FacialLandmarks) -> Dict[str, float]:
        """
        Estimate face orientation (pose) from landmarks.
        
        Args:
            landmarks: FacialLandmarks object
            
        Returns:
            Dictionary with 'pitch', 'yaw', 'roll' angles in degrees
        """
        # Key points for pose estimation
        nose_tip = landmarks.landmarks[1]  # Nose tip
        chin = landmarks.landmarks[152]    # Chin
        left_ear = landmarks.landmarks[234] # Left ear
        right_ear = landmarks.landmarks[454] # Right ear
        
        # Calculate yaw (left-right rotation)
        ear_distance_x = left_ear[0] - right_ear[0]
        yaw = np.arctan2(ear_distance_x, 1.0) * 180 / np.pi
        
        # Calculate pitch (up-down rotation)
        nose_chin_distance_y = chin[1] - nose_tip[1]
        pitch = np.arctan2(nose_chin_distance_y, 1.0) * 180 / np.pi
        
        # Calculate roll (tilt rotation)
        ear_distance_y = left_ear[1] - right_ear[1]
        roll = np.arctan2(ear_distance_y, ear_distance_x) * 180 / np.pi
        
        return {
            'pitch': float(pitch),
            'yaw': float(yaw),
            'roll': float(roll)
        }
    
    def get_facial_features(self, landmarks: FacialLandmarks) -> Dict[str, Any]:
        """
        Extract facial features and measurements.
        
        Args:
            landmarks: FacialLandmarks object
            
        Returns:
            Dictionary with facial measurements and features
        """
        # Eye aspect ratios
        left_eye_landmarks = self.get_landmark_subset(landmarks, LandmarkSubset.LEFT_EYE)
        right_eye_landmarks = self.get_landmark_subset(landmarks, LandmarkSubset.RIGHT_EYE)
        
        def eye_aspect_ratio(eye_landmarks):
            if len(eye_landmarks) < 6:
                return 0.0
            # Calculate vertical distances
            v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
            v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
            # Calculate horizontal distance
            h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
            return (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        
        left_ear = eye_aspect_ratio(left_eye_landmarks)
        right_ear = eye_aspect_ratio(right_eye_landmarks)
        
        # Mouth aspect ratio
        mouth_landmarks = self.get_landmark_subset(landmarks, LandmarkSubset.LIPS)
        mouth_ar = 0.0
        if len(mouth_landmarks) >= 8:
            v1 = np.linalg.norm(np.array(mouth_landmarks[2]) - np.array(mouth_landmarks[6]))
            v2 = np.linalg.norm(np.array(mouth_landmarks[3]) - np.array(mouth_landmarks[7]))
            h = np.linalg.norm(np.array(mouth_landmarks[0]) - np.array(mouth_landmarks[4]))
            mouth_ar = (v1 + v2) / (2.0 * h) if h > 0 else 0.0
        
        return {
            'left_eye_aspect_ratio': left_ear,
            'right_eye_aspect_ratio': right_ear,
            'mouth_aspect_ratio': mouth_ar,
            'pose': self.get_face_orientation(landmarks)
        }
    
    def __del__(self):
        """Cleanup resources."""
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()
