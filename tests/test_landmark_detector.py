"""
Tests for the LandmarkDetector class.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'facial_landmarks'))

from landmark_detector import LandmarkDetector, LandmarkSubset, FacialLandmarks
from conftest import create_test_image, create_face_like_image, create_noisy_image


class TestLandmarkDetector:
    """Test cases for LandmarkDetector class."""
    
    def test_init_default_params(self):
        """Test LandmarkDetector initialization with default parameters."""
        detector = LandmarkDetector()
        
        assert detector.max_num_faces == 1
        assert detector.refine_landmarks == True
        assert detector.min_detection_confidence == 0.5
        assert detector.min_tracking_confidence == 0.5
        assert detector.face_mesh is not None
    
    def test_init_custom_params(self):
        """Test LandmarkDetector initialization with custom parameters."""
        detector = LandmarkDetector(
            max_num_faces=3,
            refine_landmarks=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7
        )
        
        assert detector.max_num_faces == 3
        assert detector.refine_landmarks == False
        assert detector.min_detection_confidence == 0.8
        assert detector.min_tracking_confidence == 0.7
    
    def test_detect_landmarks_no_face(self, test_image):
        """Test landmark detection on image with no faces."""
        detector = LandmarkDetector()
        landmarks = detector.detect_landmarks(test_image)
        
        assert isinstance(landmarks, list)
        assert len(landmarks) == 0
    
    def test_detect_landmarks_mock_face(self):
        """Test landmark detection with mocked MediaPipe results."""
        detector = LandmarkDetector()
        
        # Create mock landmark
        mock_landmark = MagicMock()
        mock_landmark.x = 0.5
        mock_landmark.y = 0.5
        mock_landmark.z = 0.1
        
        # Create mock face landmarks
        mock_face_landmarks = MagicMock()
        mock_face_landmarks.landmark = [mock_landmark] * 468  # 468 landmarks
        
        # Create mock results
        mock_results = MagicMock()
        mock_results.multi_face_landmarks = [mock_face_landmarks]
        
        with patch.object(detector.face_mesh, 'process', return_value=mock_results):
            test_image = create_face_like_image()
            landmarks = detector.detect_landmarks(test_image)
            
            assert len(landmarks) == 1
            assert isinstance(landmarks[0], FacialLandmarks)
            assert len(landmarks[0].landmarks) == 468
            assert len(landmarks[0].landmark_indices) == 468
            
            # Check first landmark
            first_landmark = landmarks[0].landmarks[0]
            assert first_landmark == (0.5, 0.5, 0.1)
    
    def test_get_landmark_subset_all(self):
        """Test getting all landmarks subset."""
        detector = LandmarkDetector()
        
        # Create sample landmarks
        landmarks = FacialLandmarks(
            landmarks=[(i*0.01, i*0.01, i*0.001) for i in range(468)],
            landmark_indices=list(range(468))
        )
        
        subset_landmarks = detector.get_landmark_subset(landmarks, LandmarkSubset.ALL)
        assert len(subset_landmarks) == 468
        assert subset_landmarks == landmarks.landmarks
    
    def test_get_landmark_subset_face_oval(self):
        """Test getting face oval subset."""
        detector = LandmarkDetector()
        
        # Create sample landmarks
        landmarks = FacialLandmarks(
            landmarks=[(i*0.01, i*0.01, i*0.001) for i in range(468)],
            landmark_indices=list(range(468))
        )
        
        subset_landmarks = detector.get_landmark_subset(landmarks, LandmarkSubset.FACE_OVAL)
        expected_count = len(detector.LANDMARK_SUBSETS[LandmarkSubset.FACE_OVAL])
        assert len(subset_landmarks) == expected_count
    
    def test_get_landmark_subset_eyes(self):
        """Test getting eye subsets."""
        detector = LandmarkDetector()
        
        landmarks = FacialLandmarks(
            landmarks=[(i*0.01, i*0.01, i*0.001) for i in range(468)],
            landmark_indices=list(range(468))
        )
        
        left_eye = detector.get_landmark_subset(landmarks, LandmarkSubset.LEFT_EYE)
        right_eye = detector.get_landmark_subset(landmarks, LandmarkSubset.RIGHT_EYE)
        
        assert len(left_eye) == len(detector.LANDMARK_SUBSETS[LandmarkSubset.LEFT_EYE])
        assert len(right_eye) == len(detector.LANDMARK_SUBSETS[LandmarkSubset.RIGHT_EYE])
    
    def test_landmarks_to_pixels(self):
        """Test conversion of normalized landmarks to pixel coordinates."""
        detector = LandmarkDetector()
        
        # Test landmarks (normalized 0-1)
        landmarks = [(0.0, 0.0, 0.0), (0.5, 0.5, 0.1), (1.0, 1.0, 0.2)]
        image_shape = (480, 640)  # height, width
        
        pixel_landmarks = detector.landmarks_to_pixels(landmarks, image_shape)
        
        expected = [(0, 0), (320, 240), (640, 480)]
        assert pixel_landmarks == expected
    
    def test_landmarks_to_pixels_edge_cases(self):
        """Test pixel conversion with edge cases."""
        detector = LandmarkDetector()
        
        # Empty landmarks
        assert detector.landmarks_to_pixels([], (480, 640)) == []
        
        # Single landmark
        landmarks = [(0.25, 0.75, 0.0)]
        image_shape = (100, 200)
        pixel_landmarks = detector.landmarks_to_pixels(landmarks, image_shape)
        assert pixel_landmarks == [(50, 75)]
    
    def test_draw_landmarks_basic(self, face_like_image):
        """Test basic landmark drawing."""
        detector = LandmarkDetector()
        
        # Create sample landmarks
        landmarks = [FacialLandmarks(
            landmarks=[(0.3, 0.3, 0.0), (0.7, 0.3, 0.0), (0.5, 0.7, 0.0)],
            landmark_indices=[0, 1, 2]
        )]
        
        result_image = detector.draw_landmarks(
            face_like_image, 
            landmarks, 
            subset=LandmarkSubset.FACE_OVAL,
            draw_connections=False
        )
        
        assert result_image.shape == face_like_image.shape
        assert not np.array_equal(result_image, face_like_image)  # Should be modified
    
    def test_draw_all_landmarks_with_indices(self, face_like_image):
        """Test drawing all landmarks with indices."""
        detector = LandmarkDetector()
        
        # Create sample landmarks
        landmarks = [FacialLandmarks(
            landmarks=[(i*0.002, i*0.002, 0.0) for i in range(50)],  # Smaller set for testing
            landmark_indices=list(range(50))
        )]
        
        result_image = detector.draw_all_landmarks_with_indices(
            face_like_image,
            landmarks,
            show_numbers=True
        )
        
        assert result_image.shape == face_like_image.shape
        assert not np.array_equal(result_image, face_like_image)
    
    def test_get_face_orientation(self):
        """Test face orientation estimation."""
        detector = LandmarkDetector()
        
        # Create landmarks with specific positions for known orientation
        landmarks = FacialLandmarks(
            landmarks=[(i*0.002, i*0.002, i*0.0001) for i in range(468)],
            landmark_indices=list(range(468))
        )
        
        # Override specific landmarks for pose calculation
        landmarks.landmarks[1] = (0.5, 0.4, 0.0)    # nose tip
        landmarks.landmarks[152] = (0.5, 0.6, 0.0)  # chin
        landmarks.landmarks[234] = (0.3, 0.5, 0.0)  # left ear
        landmarks.landmarks[454] = (0.7, 0.5, 0.0)  # right ear
        
        orientation = detector.get_face_orientation(landmarks)
        
        assert 'pitch' in orientation
        assert 'yaw' in orientation
        assert 'roll' in orientation
        assert isinstance(orientation['pitch'], float)
        assert isinstance(orientation['yaw'], float)
        assert isinstance(orientation['roll'], float)
    
    def test_get_facial_features(self):
        """Test facial features extraction."""
        detector = LandmarkDetector()
        
        # Create landmarks
        landmarks = FacialLandmarks(
            landmarks=[(i*0.002, i*0.002, i*0.0001) for i in range(468)],
            landmark_indices=list(range(468))
        )
        
        features = detector.get_facial_features(landmarks)
        
        assert 'left_eye_aspect_ratio' in features
        assert 'right_eye_aspect_ratio' in features
        assert 'mouth_aspect_ratio' in features
        assert 'pose' in features
        
        assert isinstance(features['left_eye_aspect_ratio'], float)
        assert isinstance(features['right_eye_aspect_ratio'], float)
        assert isinstance(features['mouth_aspect_ratio'], float)
        assert isinstance(features['pose'], dict)
    
    def test_landmark_subsets_coverage(self):
        """Test that all landmark subsets are properly defined."""
        detector = LandmarkDetector()
        
        # Check that all subset enums have corresponding indices
        for subset in LandmarkSubset:
            if subset != LandmarkSubset.ALL:
                assert subset in detector.LANDMARK_SUBSETS
                indices = detector.LANDMARK_SUBSETS[subset]
                assert isinstance(indices, list)
                assert len(indices) > 0
                assert all(isinstance(idx, int) and 0 <= idx < 468 for idx in indices)
    
    def test_detector_cleanup(self):
        """Test detector cleanup."""
        detector = LandmarkDetector()
        assert hasattr(detector, 'face_mesh')
        
        # Test cleanup
        detector.__del__()
        # Should not raise any exceptions
    
    def test_invalid_image_input(self):
        """Test handling of invalid image inputs."""
        detector = LandmarkDetector()
        
        # Test with None
        with pytest.raises((AttributeError, TypeError)):
            detector.detect_landmarks(None)
        
        # Test with invalid shape
        invalid_image = np.ones((100, 100), dtype=np.uint8)  # Missing channel dimension
        # Should handle gracefully or raise appropriate error
        try:
            result = detector.detect_landmarks(invalid_image)
            assert isinstance(result, list)
        except (ValueError, cv2.error):
            pass  # Expected for invalid input
    
    def test_empty_landmarks_handling(self):
        """Test handling of empty landmarks lists."""
        detector = LandmarkDetector()
        test_image = create_test_image()
        
        # Test drawing with empty landmarks
        result = detector.draw_landmarks(test_image, [], LandmarkSubset.ALL)
        assert np.array_equal(result, test_image)  # Should return unchanged image
        
        # Test debug drawing with empty landmarks
        result = detector.draw_all_landmarks_with_indices(test_image, [])
        assert np.array_equal(result, test_image)  # Should return unchanged image
    
    @pytest.mark.parametrize("subset", [
        LandmarkSubset.FACE_OVAL,
        LandmarkSubset.LEFT_EYE,
        LandmarkSubset.RIGHT_EYE,
        LandmarkSubset.NOSE,
        LandmarkSubset.LIPS
    ])
    def test_landmark_subsets_individual(self, subset, face_like_image):
        """Test individual landmark subsets."""
        detector = LandmarkDetector()
        
        # Create sample landmarks
        landmarks = [FacialLandmarks(
            landmarks=[(i*0.002, i*0.002, 0.0) for i in range(468)],
            landmark_indices=list(range(468))
        )]
        
        result = detector.draw_landmarks(
            face_like_image,
            landmarks,
            subset=subset,
            draw_connections=True
        )
        
        assert result.shape == face_like_image.shape
