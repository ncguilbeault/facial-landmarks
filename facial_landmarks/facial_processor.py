"""
Combined facial detection and landmark processing system.
Provides a unified interface for face detection, landmark detection, and real-time processing.
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import threading

from face_detector import FaceDetector, FaceDetection
from landmark_detector import LandmarkDetector, FacialLandmarks, LandmarkSubset


@dataclass
class ProcessingResult:
    """Combined result from face detection and landmark detection."""
    faces: List[FaceDetection] = field(default_factory=list)
    landmarks: List[FacialLandmarks] = field(default_factory=list)
    processing_time: float = 0.0
    fps: float = 0.0
    image_shape: Tuple[int, int] = (0, 0)
    processed_image: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __getitem__(self, key):
        """Allow dictionary-like access for backward compatibility."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"'{key}' not found in ProcessingResult")
    
    def __contains__(self, key):
        """Allow 'in' operator for backward compatibility."""
        return hasattr(self, key)


@dataclass
class ProcessingConfig:
    """Configuration for facial processing pipeline."""
    # Face detection settings
    face_detection_confidence: float = 0.5
    face_model_selection: int = 0  # 0 for short-range, 1 for full-range
    
    # Landmark detection settings
    max_num_faces: int = 1
    refine_landmarks: bool = True
    landmark_detection_confidence: float = 0.5
    landmark_tracking_confidence: float = 0.5
    
    # Processing settings
    enable_face_detection: bool = True
    enable_landmark_detection: bool = True
    enable_parallel_processing: bool = False
    
    # Visualization settings
    draw_face_boxes: bool = False
    draw_face_keypoints: bool = True
    draw_landmarks: bool = True
    landmark_subset: LandmarkSubset = LandmarkSubset.ALL
    draw_landmark_connections: bool = True
    draw_detailed_landmarks: bool = True
    show_landmark_indices: bool = False
    
    # Backward compatibility aliases for tests
    min_detection_confidence: Optional[float] = None
    min_tracking_confidence: Optional[float] = None
    
    # Output settings
    output_path: Optional[str] = None
    save_video: bool = False
    show_fps: bool = True
    
    def __post_init__(self):
        """Set up backward compatibility and validate settings."""
        # Handle backward compatibility
        if self.min_detection_confidence is not None:
            self.face_detection_confidence = self.min_detection_confidence
        else:
            self.min_detection_confidence = self.face_detection_confidence
            
        if self.min_tracking_confidence is not None:
            self.landmark_tracking_confidence = self.min_tracking_confidence
        else:
            self.min_tracking_confidence = self.landmark_tracking_confidence


class FacialLandmarkProcessor:
    """Combined facial detection and landmark processing system."""
    
    def __init__(self, config: Optional[ProcessingConfig] = None):
        """
        Initialize the facial landmark processor.
        
        Args:
            config: Processing configuration
        """
        self.config = config or ProcessingConfig()
        self._lock = threading.Lock()
        self._executor = None
        
        # Initialize detectors
        if self.config.enable_face_detection:
            self.face_detector = FaceDetector(
                min_detection_confidence=self.config.face_detection_confidence,
                model_selection=self.config.face_model_selection
            )
        else:
            self.face_detector = None
            
        if self.config.enable_landmark_detection:
            self.landmark_detector = LandmarkDetector(
                max_num_faces=self.config.max_num_faces,
                refine_landmarks=self.config.refine_landmarks,
                min_detection_confidence=self.config.landmark_detection_confidence,
                min_tracking_confidence=self.config.landmark_tracking_confidence
            )
        else:
            self.landmark_detector = None
        
        # Performance tracking
        self._frame_times = []
        self._last_time = time.time()
    
    def process_image(self, image: np.ndarray) -> ProcessingResult:
        """
        Process a single image for faces and landmarks.
        
        Args:
            image: Input image as numpy array (BGR format)
            
        Returns:
            ProcessingResult with detected faces and landmarks
        """
        start_time = time.time()
        
        faces = []
        landmarks = []
        
        if self.config.enable_parallel_processing and self.face_detector and self.landmark_detector:
            # Parallel processing
            with ThreadPoolExecutor(max_workers=2) as executor:
                face_future = executor.submit(self.face_detector.detect_faces, image)
                landmark_future = executor.submit(self.landmark_detector.detect_landmarks, image)
                
                faces = face_future.result()
                landmarks = landmark_future.result()
        else:
            # Sequential processing
            if self.face_detector:
                faces = self.face_detector.detect_faces(image)
            
            if self.landmark_detector:
                landmarks = self.landmark_detector.detect_landmarks(image)
        
        processing_time = time.time() - start_time
        
        # Calculate FPS
        current_time = time.time()
        frame_time = current_time - self._last_time
        self._last_time = current_time
        
        with self._lock:
            self._frame_times.append(frame_time)
            if len(self._frame_times) > 30:  # Keep last 30 frames
                self._frame_times.pop(0)
            
            avg_frame_time = sum(self._frame_times) / len(self._frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0
        
        return ProcessingResult(
            faces=faces,
            landmarks=landmarks,
            processing_time=processing_time,
            fps=fps,
            image_shape=image.shape[:2],
            processed_image=self.draw_results(image, ProcessingResult(faces=faces, landmarks=landmarks))
        )
    
    def process_video_stream(self, 
                           source: Union[int, str] = 0,
                           output_path: Optional[str] = None,
                           display: bool = True) -> None:
        """
        Process video stream in real-time.
        
        Args:
            source: Video source (camera index or video file path)
            output_path: Optional path to save output video
            display: Whether to display the video stream
        """
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Starting video processing...")
        print(f"Resolution: {width}x{height}")
        print(f"Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("End of video stream")
                    break
                
                # Process frame
                result = self.process_image(frame)
                
                # Draw visualizations
                annotated_frame = self.draw_results(frame, result)
                
                # Add performance info
                self._draw_performance_info(annotated_frame, result)
                
                # Save frame if writer is available
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame
                if display:
                    cv2.imshow('Facial Landmarks', annotated_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        screenshot_path = f"screenshot_{frame_count:06d}.jpg"
                        cv2.imwrite(screenshot_path, annotated_frame)
                        print(f"Screenshot saved: {screenshot_path}")
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\\nInterrupted by user")
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
            
            print(f"\\nProcessed {frame_count} frames")
    
    def draw_results(self, image: np.ndarray, result: ProcessingResult) -> np.ndarray:
        """
        Draw processing results on image.
        
        Args:
            image: Input image
            result: Processing results
            
        Returns:
            Image with drawn results
        """
        annotated_image = image.copy()
        
        # Draw face detections
        if self.config.draw_face_boxes and result.faces and self.face_detector:
            annotated_image = self.face_detector.draw_detections(
                annotated_image, 
                result.faces, 
                draw_keypoints=self.config.draw_face_keypoints
            )
        
        # Draw landmarks
        if self.config.draw_landmarks and result.landmarks and self.landmark_detector:
            if self.config.show_landmark_indices:
                # Use detailed debug drawing with indices
                annotated_image = self.landmark_detector.draw_all_landmarks_with_indices(
                    annotated_image,
                    result.landmarks,
                    show_numbers=True
                )
            else:
                # Use standard landmark drawing
                annotated_image = self.landmark_detector.draw_landmarks(
                    annotated_image,
                    result.landmarks,
                    subset=self.config.landmark_subset,
                    draw_connections=self.config.draw_landmark_connections,
                    draw_detailed=self.config.draw_detailed_landmarks
                )
        
        return annotated_image
    
    def _draw_performance_info(self, image: np.ndarray, result: ProcessingResult) -> None:
        """Draw performance information on image."""
        h, w = image.shape[:2]
        
        # Performance text
        fps_text = f"FPS: {result.fps:.1f}"
        time_text = f"Process: {result.processing_time*1000:.1f}ms"
        faces_text = f"Faces: {len(result.faces)}"
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)
        thickness = 1
        
        # Draw background for text
        texts = [fps_text, time_text, faces_text]
        y_start = 30
        for i, text in enumerate(texts):
            y = y_start + i * 25
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            cv2.rectangle(image, (10, y - text_h - 5), (10 + text_w + 10, y + 5), (0, 0, 0), -1)
            cv2.putText(image, text, (15, y), font, font_scale, color, thickness)
    
    def get_facial_analysis(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Get comprehensive facial analysis.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with detailed facial analysis
        """
        result = self.process_image(image)
        
        analysis = {
            'num_faces': len(result.faces),
            'processing_time': result.processing_time,
            'faces': [],
            'landmarks': []
        }
        
        # Face analysis
        for i, face in enumerate(result.faces):
            # Convert BoundingBox to pixel coordinates for JSON serialization
            bbox_pixels = self.face_detector.bbox_to_pixels(face.bbox, image.shape[:2])
            face_info = {
                'id': i,
                'bbox': list(bbox_pixels),  # Convert to list for JSON serialization
                'confidence': face.confidence,
                'keypoints': face.keypoints
            }
            analysis['faces'].append(face_info)
        
        # Landmark analysis
        if self.landmark_detector:
            for i, landmarks in enumerate(result.landmarks):
                features = self.landmark_detector.get_facial_features(landmarks)
                landmark_info = {
                    'id': i,
                    'num_landmarks': len(landmarks.landmarks),
                    'features': features
                }
                analysis['landmarks'].append(landmark_info)
        
        return analysis
    
    def save_results(self, *args, **kwargs) -> Dict[str, str]:
        """
        Save processing results to files.
        
        Supports two calling patterns:
        1. save_results(image, result, output_dir="output")
        2. save_results(results_dict, output_path)
        
        Returns:
            Dictionary with saved file paths
        """
        import os
        import json
        from datetime import datetime
        
        # Handle different calling patterns
        if len(args) >= 2 and isinstance(args[1], (ProcessingResult, dict)) and isinstance(args[0], np.ndarray):
            # Pattern 1: save_results(image, result, output_dir)
            image = args[0]
            result = args[1]
            output_dir = args[2] if len(args) > 2 else kwargs.get('output_dir', 'output')
            
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            saved_files = {}
            
            # Save annotated image
            annotated_image = self.draw_results(image, result)
            image_path = os.path.join(output_dir, f"result_{timestamp}.jpg")
            cv2.imwrite(image_path, annotated_image)
            saved_files['image'] = image_path
            
            # Save analysis data
            analysis = self.get_facial_analysis(image)
            json_path = os.path.join(output_dir, f"analysis_{timestamp}.json")
            with open(json_path, 'w') as f:
                json.dump(analysis, f, indent=2)
            saved_files['analysis'] = json_path
            
            return saved_files
            
        elif len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], str):
            # Pattern 2: save_results(results_dict, output_path)
            results = args[0]
            output_path = args[1]
            
            # Save results dictionary directly to JSON file
            with open(output_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            return {'results': output_path}
        
        else:
            raise ValueError("Invalid arguments for save_results. Expected patterns: "
                           "save_results(image, result, output_dir) or "
                           "save_results(results_dict, output_path)")
    
    def update_config(self, **kwargs) -> None:
        """Update processing configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                print(f"Warning: Unknown config parameter: {key}")
    
    def __del__(self):
        """Cleanup resources."""
        if self._executor:
            self._executor.shutdown(wait=True)
    
    def process_webcam_frame(self, image: np.ndarray) -> np.ndarray:
        """
        Process a single webcam frame and return annotated image.
        
        Args:
            image: Input frame from webcam
            
        Returns:
            Annotated image with detections drawn
        """
        result = self.process_image(image)
        return self.draw_results(image, result)
    
    def process_webcam(self, device_id: int = 0) -> None:
        """
        Process webcam feed in real-time.
        
        Args:
            device_id: Camera device ID (default: 0)
        """
        cap = cv2.VideoCapture(device_id)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open webcam with device ID: {device_id}")
        
        print("Starting webcam processing. Press 'q' to quit.")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read from webcam")
                    break
                
                # Process frame
                annotated_frame = self.process_webcam_frame(frame)
                
                # Display
                cv2.imshow('Facial Landmarks - Webcam', annotated_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            cv2.destroyAllWindows()
    
    def process_video(self, video_path: str, output_path: Optional[str] = None) -> None:
        """
        Process video file.
        
        Args:
            video_path: Path to input video file
            output_path: Optional path to save output video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Warning: Failed to open video file: {video_path}")
            return None
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # Setup video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        print(f"Processing video: {video_path}")
        frame_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame = self.process_webcam_frame(frame)
                
                # Save frame if writer is available
                if writer:
                    writer.write(annotated_frame)
                
                # Display frame (can be disabled for batch processing)
                cv2.imshow('Video Processing', annotated_frame)
                
                # Check for quit key
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"Processed {frame_count} frames")
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """
        Get information about a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Dictionary with video information
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {video_path}")
        
        try:
            info = {
                'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': float(cap.get(cv2.CAP_PROP_FPS)),
                'frame_count': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                'duration': float(cap.get(cv2.CAP_PROP_FRAME_COUNT)) / float(cap.get(cv2.CAP_PROP_FPS))
            }
        finally:
            cap.release()
        
        return info
    
    def calculate_fps(self, frame_times: List[float]) -> float:
        """
        Calculate FPS from frame times.
        
        Args:
            frame_times: List of frame timestamps
            
        Returns:
            Calculated FPS
        """
        if len(frame_times) < 2:
            return 0.0
        
        # Calculate time differences
        time_diffs = [frame_times[i] - frame_times[i-1] for i in range(1, len(frame_times))]
        avg_time_diff = sum(time_diffs) / len(time_diffs)
        
        return 1.0 / avg_time_diff if avg_time_diff > 0 else 0.0
    
    def add_fps_text(self, image: np.ndarray, fps: float) -> np.ndarray:
        """
        Add FPS text to image.
        
        Args:
            image: Input image
            fps: FPS value to display
            
        Returns:
            Image with FPS text
        """
        result_image = image.copy()
        
        # Add FPS text
        fps_text = f"FPS: {fps:.1f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        color = (0, 255, 0)
        thickness = 1
        
        # Get text size for background
        (text_w, text_h), _ = cv2.getTextSize(fps_text, font, font_scale, thickness)
        
        # Draw background
        cv2.rectangle(result_image, (10, 10), (10 + text_w + 10, 10 + text_h + 10), (0, 0, 0), -1)
        
        # Draw text
        cv2.putText(result_image, fps_text, (15, 10 + text_h), font, font_scale, color, thickness)
        
        return result_image
    
    def add_info_text(self, image: np.ndarray, info: Dict[str, Any]) -> np.ndarray:
        """
        Add information text to image.
        
        Args:
            image: Input image
            info: Dictionary with information to display
            
        Returns:
            Image with information text
        """
        result_image = image.copy()
        
        # Prepare text lines
        lines = []
        if 'faces_detected' in info:
            lines.append(f"Faces: {info['faces_detected']}")
        if 'landmarks_detected' in info:
            lines.append(f"Landmarks: {info['landmarks_detected']}")
        if 'processing_time' in info:
            lines.append(f"Time: {info['processing_time']*1000:.1f}ms")
        
        # Text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0)
        thickness = 1
        
        # Draw each line
        y_start = 30
        for i, line in enumerate(lines):
            y = y_start + i * 20
            (text_w, text_h), _ = cv2.getTextSize(line, font, font_scale, thickness)
            
            # Draw background
            cv2.rectangle(result_image, (10, y - text_h - 2), (10 + text_w + 5, y + 2), (0, 0, 0), -1)
            
            # Draw text
            cv2.putText(result_image, line, (12, y), font, font_scale, color, thickness)
        
        return result_image
    
    def resize_image(self, image: np.ndarray, width: Optional[int] = None, 
                    height: Optional[int] = None) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            width: Target width (optional)
            height: Target height (optional)
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        
        if width is None and height is None:
            return image.copy()
        
        if width is None:
            # Calculate width based on height
            aspect_ratio = w / h
            width = int(height * aspect_ratio)
        elif height is None:
            # Calculate height based on width
            aspect_ratio = h / w
            height = int(width * aspect_ratio)
        
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    
    def process_batch(self, images: List[np.ndarray]) -> List[ProcessingResult]:
        """
        Process a batch of images.
        
        Args:
            images: List of input images
            
        Returns:
            List of processing results
        """
        results = []
        
        for image in images:
            result = self.process_image(image)
            results.append(result)
        
        return results
