"""
Tests for the FaceDetector class.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'facial_landmarks'))

from face_detector import FaceDetector, FaceDetection, BoundingBox
from conftest import create_test_image, create_face_like_image, create_noisy_image


class TestFaceDetector:
    """Test cases for FaceDetector class."""
    
    def test_init_default_params(self):
        """Test FaceDetector initialization with default parameters."""
        detector = FaceDetector()
        
        assert detector.model_selection == 0
        assert detector.min_detection_confidence == 0.7
        assert detector.face_detection is not None
    
    def test_init_custom_params(self):
        """Test FaceDetector initialization with custom parameters."""
        detector = FaceDetector(
            model_selection=1,
            min_detection_confidence=0.8
        )
        
        assert detector.model_selection == 1
        assert detector.min_detection_confidence == 0.8
    
    def test_detect_faces_no_face(self, test_image):
        """Test face detection on image with no faces."""
        detector = FaceDetector()
        faces = detector.detect_faces(test_image)
        
        assert isinstance(faces, list)
        assert len(faces) == 0
    
    def test_detect_faces_mock_face(self):
        """Test face detection with mocked MediaPipe results."""
        detector = FaceDetector()
        
        # Create mock detection
        mock_detection = MagicMock()
        mock_detection.location_data.relative_bounding_box.xmin = 0.2
        mock_detection.location_data.relative_bounding_box.ymin = 0.3
        mock_detection.location_data.relative_bounding_box.width = 0.4
        mock_detection.location_data.relative_bounding_box.height = 0.5
        mock_detection.score = [0.9]
        
        # Create mock results
        mock_results = MagicMock()
        mock_results.detections = [mock_detection]
        
        with patch.object(detector.face_detection, 'process', return_value=mock_results):
            test_image = create_face_like_image()
            faces = detector.detect_faces(test_image)
            
            assert len(faces) == 1
            assert isinstance(faces[0], FaceDetection)
            
            face = faces[0]
            assert face.confidence == 0.9
            assert face.bbox.x == 0.2
            assert face.bbox.y == 0.3
            assert face.bbox.width == 0.4
            assert face.bbox.height == 0.5
    
    def test_bbox_to_pixels(self):
        """Test bounding box conversion to pixel coordinates."""
        detector = FaceDetector()
        
        # Normalized bbox
        bbox = BoundingBox(x=0.25, y=0.3, width=0.4, height=0.5)
        
        image_shape = (480, 640)  # height, width
        pixel_bbox = detector.bbox_to_pixels(bbox, image_shape)
        
        expected_x = int(0.25 * 640)  # 160
        expected_y = int(0.3 * 480)   # 144
        expected_w = int(0.4 * 640)   # 256
        expected_h = int(0.5 * 480)   # 240
        
        assert pixel_bbox == (expected_x, expected_y, expected_w, expected_h)
    
    def test_bbox_to_pixels_edge_cases(self):
        """Test bbox conversion with edge cases."""
        detector = FaceDetector()
        
        # Full image bbox
        bbox = BoundingBox(x=0.0, y=0.0, width=1.0, height=1.0)
        
        image_shape = (100, 200)
        pixel_bbox = detector.bbox_to_pixels(bbox, image_shape)
        assert pixel_bbox == (0, 0, 200, 100)
        
        # Small bbox
        bbox = BoundingBox(x=0.9, y=0.9, width=0.1, height=0.1)
        
        pixel_bbox = detector.bbox_to_pixels(bbox, image_shape)
        assert pixel_bbox == (180, 90, 20, 10)
    
    def test_draw_faces(self, face_like_image):
        """Test drawing face bounding boxes."""
        detector = FaceDetector()
        
        # Create sample face detection
        faces = [FaceDetection(
            bbox=BoundingBox(x=0.2, y=0.3, width=0.4, height=0.5),
            confidence=0.85
        )]
        
        result_image = detector.draw_faces(face_like_image, faces)
        
        assert result_image.shape == face_like_image.shape
        assert not np.array_equal(result_image, face_like_image)  # Should be modified
    
    def test_draw_faces_empty_list(self, face_like_image):
        """Test drawing with empty face list."""
        detector = FaceDetector()
        
        result_image = detector.draw_faces(face_like_image, [])
        
        # Should return unchanged image
        assert np.array_equal(result_image, face_like_image)
    
    def test_draw_faces_custom_colors(self, face_like_image):
        """Test drawing faces with custom colors."""
        detector = FaceDetector()
        
        faces = [FaceDetection(
            bbox=BoundingBox(x=0.2, y=0.3, width=0.4, height=0.5),
            confidence=0.85
        )]
        
        result_image = detector.draw_faces(
            face_like_image, 
            faces,
            bbox_color=(255, 0, 0),  # Red
            text_color=(0, 255, 0)   # Green
        )
        
        assert result_image.shape == face_like_image.shape
        assert not np.array_equal(result_image, face_like_image)
    
    def test_get_face_region(self):
        """Test extracting face region from image."""
        detector = FaceDetector()
        
        # Create test image
        test_image = create_face_like_image()
        
        # Create face detection
        face = FaceDetection(
            bbox=BoundingBox(x=0.2, y=0.3, width=0.4, height=0.4),
            confidence=0.9
        )
        
        face_region = detector.get_face_region(test_image, face)
        
        # Should return cropped region
        assert face_region.shape[0] > 0  # height
        assert face_region.shape[1] > 0  # width
        assert len(face_region.shape) == 3  # channels
        
        # Should be smaller than original image
        assert face_region.shape[0] <= test_image.shape[0]
        assert face_region.shape[1] <= test_image.shape[1]
    
    def test_get_face_region_edge_cases(self):
        """Test face region extraction edge cases."""
        detector = FaceDetector()
        test_image = create_test_image(width=100, height=100)
        
        # Face bbox extending beyond image boundaries
        face = FaceDetection(
            bbox=BoundingBox(x=0.8, y=0.8, width=0.5, height=0.5),
            confidence=0.9
        )
        
        face_region = detector.get_face_region(test_image, face)
        
        # Should handle gracefully and return valid region
        assert face_region.shape[0] > 0
        assert face_region.shape[1] > 0
    
    def test_filter_faces_by_confidence(self):
        """Test filtering faces by confidence threshold."""
        detector = FaceDetector()
        
        faces = [
            FaceDetection(bbox=BoundingBox(x=0.1, y=0.1, width=0.2, height=0.2), confidence=0.9),
            FaceDetection(bbox=BoundingBox(x=0.2, y=0.2, width=0.3, height=0.3), confidence=0.4),
            FaceDetection(bbox=BoundingBox(x=0.3, y=0.3, width=0.4, height=0.4), confidence=0.7),
            FaceDetection(bbox=BoundingBox(x=0.4, y=0.4, width=0.5, height=0.5), confidence=0.3)
        ]
        
        # Filter with threshold 0.5
        filtered = detector.filter_faces_by_confidence(faces, 0.5)
        assert len(filtered) == 2
        assert all(face.confidence >= 0.5 for face in filtered)
        
        # Filter with threshold 0.8
        filtered = detector.filter_faces_by_confidence(faces, 0.8)
        assert len(filtered) == 1
        assert filtered[0].confidence == 0.9
        
        # Filter with threshold 1.0 (none should pass)
        filtered = detector.filter_faces_by_confidence(faces, 1.0)
        assert len(filtered) == 0
    
    def test_get_largest_face(self):
        """Test getting the largest detected face."""
        detector = FaceDetector()
        
        faces = [
            FaceDetection(bbox=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.3), confidence=0.8),
            FaceDetection(bbox=BoundingBox(x=0.2, y=0.2, width=0.5, height=0.4), confidence=0.7),  # Largest
            FaceDetection(bbox=BoundingBox(x=0.3, y=0.3, width=0.2, height=0.2), confidence=0.9)
        ]
        
        largest = detector.get_largest_face(faces)
        assert largest is not None
        assert largest.bbox.width == 0.5
        assert largest.bbox.height == 0.4
        
        # Test with empty list
        largest = detector.get_largest_face([])
        assert largest is None
    
    def test_face_area_calculation(self):
        """Test face area calculation."""
        detector = FaceDetector()
        
        face = FaceDetection(
            bbox=BoundingBox(x=0.1, y=0.1, width=0.4, height=0.3),
            confidence=0.8
        )
        
        area = detector.get_face_area(face)
        assert area == 0.4 * 0.3  # width * height
    
    def test_detector_cleanup(self):
        """Test detector cleanup."""
        detector = FaceDetector()
        assert hasattr(detector, 'face_detection')
        
        # Test cleanup
        detector.__del__()
        # Should not raise any exceptions
    
    def test_invalid_image_input(self):
        """Test handling of invalid image inputs."""
        detector = FaceDetector()
        
        # Test with None
        with pytest.raises((AttributeError, TypeError)):
            detector.detect_faces(None)
        
        # Test with invalid shape
        invalid_image = np.ones((100, 100), dtype=np.uint8)  # Missing channel dimension
        try:
            result = detector.detect_faces(invalid_image)
            assert isinstance(result, list)
        except (ValueError, cv2.error):
            pass  # Expected for invalid input
    
    def test_multiple_faces_detection(self):
        """Test detection of multiple faces."""
        detector = FaceDetector()
        
        # Create mock detections for multiple faces
        mock_detection1 = MagicMock()
        mock_detection1.location_data.relative_bounding_box.xmin = 0.1
        mock_detection1.location_data.relative_bounding_box.ymin = 0.1
        mock_detection1.location_data.relative_bounding_box.width = 0.3
        mock_detection1.location_data.relative_bounding_box.height = 0.4
        mock_detection1.score = [0.9]
        
        mock_detection2 = MagicMock()
        mock_detection2.location_data.relative_bounding_box.xmin = 0.6
        mock_detection2.location_data.relative_bounding_box.ymin = 0.2
        mock_detection2.location_data.relative_bounding_box.width = 0.3
        mock_detection2.location_data.relative_bounding_box.height = 0.4
        mock_detection2.score = [0.8]
        
        mock_results = MagicMock()
        mock_results.detections = [mock_detection1, mock_detection2]
        
        with patch.object(detector.face_detection, 'process', return_value=mock_results):
            test_image = create_face_like_image()
            faces = detector.detect_faces(test_image)
            
            assert len(faces) == 2
            assert all(isinstance(face, FaceDetection) for face in faces)
            assert faces[0].confidence == 0.9
            assert faces[1].confidence == 0.8
    
    @pytest.mark.parametrize("confidence_threshold", [0.3, 0.5, 0.7, 0.9])
    def test_confidence_thresholds(self, confidence_threshold):
        """Test different confidence thresholds."""
        detector = FaceDetector(min_detection_confidence=confidence_threshold)
        assert detector.min_detection_confidence == confidence_threshold
    
    def test_face_statistics(self):
        """Test face detection statistics."""
        detector = FaceDetector()
        
        faces = [
            FaceDetection(bbox=BoundingBox(x=0.1, y=0.1, width=0.3, height=0.3), confidence=0.8),
            FaceDetection(bbox=BoundingBox(x=0.2, y=0.2, width=0.5, height=0.4), confidence=0.7),
            FaceDetection(bbox=BoundingBox(x=0.3, y=0.3, width=0.2, height=0.2), confidence=0.9)
        ]
        
        stats = detector.get_detection_statistics(faces)
        
        assert 'count' in stats
        assert 'avg_confidence' in stats
        assert 'max_confidence' in stats
        assert 'min_confidence' in stats
        assert 'avg_area' in stats
        
        assert stats['count'] == 3
        assert stats['max_confidence'] == 0.9
        assert stats['min_confidence'] == 0.7
        assert abs(stats['avg_confidence'] - 0.8) < 0.01
