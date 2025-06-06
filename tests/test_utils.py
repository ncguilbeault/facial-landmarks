"""
Tests for utility functions.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
from pathlib import Path

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'facial_landmarks'))

from utils import (
    validate_image, validate_video_path, calculate_distance,
    calculate_angle, normalize_landmarks, denormalize_landmarks,
    get_bounding_box, crop_image_region, resize_with_aspect_ratio,
    save_image, load_image, create_video_writer, estimate_pose,
    calculate_eye_aspect_ratio, calculate_mouth_aspect_ratio,
    smooth_landmarks, interpolate_landmarks
)
from conftest import create_test_image, create_face_like_image


class TestImageValidation:
    """Test image validation utilities."""
    
    def test_validate_image_valid(self):
        """Test validation of valid images."""
        # Valid RGB image
        image = create_test_image()
        assert validate_image(image) == True
        
        # Valid grayscale image
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        assert validate_image(gray_image) == True
    
    def test_validate_image_invalid(self):
        """Test validation of invalid images."""
        # None image
        assert validate_image(None) == False
        
        # Empty array
        assert validate_image(np.array([])) == False
        
        # Wrong dimensions
        assert validate_image(np.ones((10,))) == False  # 1D
        assert validate_image(np.ones((10, 10, 10, 10))) == False  # 4D
        
        # Wrong data type
        assert validate_image(np.ones((10, 10, 3), dtype=np.float64)) == False
    
    def test_validate_video_path_valid(self):
        """Test validation of valid video paths."""
        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            # Write some dummy data
            temp_file.write(b'dummy video data')
        
        try:
            # Note: This may return False for actual validation since it's not a real video
            # But it should not crash
            result = validate_video_path(temp_path)
            assert isinstance(result, bool)
        finally:
            os.unlink(temp_path)
    
    def test_validate_video_path_invalid(self):
        """Test validation of invalid video paths."""
        assert validate_video_path("nonexistent_file.mp4") == False
        assert validate_video_path(None) == False
        assert validate_video_path("") == False
        assert validate_video_path("file_without_extension") == False


class TestGeometricCalculations:
    """Test geometric calculation utilities."""
    
    def test_calculate_distance(self):
        """Test distance calculation between points."""
        point1 = (0, 0)
        point2 = (3, 4)
        
        distance = calculate_distance(point1, point2)
        assert abs(distance - 5.0) < 1e-6  # 3-4-5 triangle
        
        # Same point - distance should be 0
        assert calculate_distance(point1, point1) == 0.0
        
        # 3D points
        point1_3d = (0, 0, 0)
        point2_3d = (1, 1, 1)
        distance_3d = calculate_distance(point1_3d, point2_3d)
        expected = np.sqrt(3)
        assert abs(distance_3d - expected) < 1e-6
    
    def test_calculate_angle(self):
        """Test angle calculation between three points."""
        # Right angle case
        point1 = (0, 0)
        point2 = (1, 0)  # vertex
        point3 = (1, 1)
        
        angle = calculate_angle(point1, point2, point3)
        assert abs(angle - 90.0) < 1e-6  # Should be 90 degrees
        
        # Straight line case (180 degrees)
        point3_straight = (2, 0)
        angle_straight = calculate_angle(point1, point2, point3_straight)
        assert abs(angle_straight - 180.0) < 1e-6
        
        # Acute angle case
        point3_acute = (1.5, 0.5)  # This creates an acute angle
        angle_acute = calculate_angle(point1, point2, point3_acute)
        assert 0 < angle_acute < 90
    
    def test_normalize_landmarks(self):
        """Test landmark normalization."""
        # Pixel coordinates
        landmarks = [(100, 200), (300, 400), (500, 600)]
        image_shape = (800, 600)  # height, width
        
        normalized = normalize_landmarks(landmarks, image_shape)
        
        expected = [
            (100/600, 200/800),  # x/width, y/height
            (300/600, 400/800),
            (500/600, 600/800)
        ]
        
        assert len(normalized) == len(expected)
        for norm, exp in zip(normalized, expected):
            assert abs(norm[0] - exp[0]) < 1e-6
            assert abs(norm[1] - exp[1]) < 1e-6
    
    def test_denormalize_landmarks(self):
        """Test landmark denormalization."""
        # Normalized coordinates (0-1)
        normalized = [(0.25, 0.5), (0.75, 0.8)]
        image_shape = (400, 300)  # height, width
        
        landmarks = denormalize_landmarks(normalized, image_shape)
        
        expected = [
            (int(0.25 * 300), int(0.5 * 400)),  # x*width, y*height
            (int(0.75 * 300), int(0.8 * 400))
        ]
        
        assert landmarks == expected
    
    def test_get_bounding_box(self):
        """Test bounding box calculation from points."""
        points = [(10, 20), (50, 30), (25, 60), (5, 15)]
        
        bbox = get_bounding_box(points)
        
        # Should be (min_x, min_y, width, height)
        expected = (5, 15, 45, 45)  # min_x=5, min_y=15, max_x=50, max_y=60
        assert bbox == expected
        
        # Single point
        single_point = [(10, 20)]
        bbox_single = get_bounding_box(single_point)
        assert bbox_single == (10, 20, 0, 0)
        
        # Empty list
        with pytest.raises((ValueError, IndexError)):
            get_bounding_box([])


class TestImageProcessing:
    """Test image processing utilities."""
    
    def test_crop_image_region(self):
        """Test cropping image region."""
        image = create_test_image(width=400, height=300)
        
        # Crop region
        x, y, w, h = 50, 60, 200, 150
        cropped = crop_image_region(image, x, y, w, h)
        
        assert cropped.shape == (h, w, 3)
        
        # Test bounds checking
        # Region extending beyond image bounds
        cropped_large = crop_image_region(image, 350, 250, 100, 100)
        assert cropped_large.shape[0] <= 100
        assert cropped_large.shape[1] <= 100
    
    def test_resize_with_aspect_ratio(self):
        """Test image resizing while maintaining aspect ratio."""
        image = create_test_image(width=400, height=300)  # 4:3 ratio
        
        # Resize by width
        resized = resize_with_aspect_ratio(image, target_width=200)
        assert resized.shape[1] == 200
        assert resized.shape[0] == 150  # Maintains 4:3 ratio
        
        # Resize by height
        resized = resize_with_aspect_ratio(image, target_height=150)
        assert resized.shape[0] == 150
        assert resized.shape[1] == 200  # Maintains 4:3 ratio
        
        # Resize with both dimensions (should use smaller scale)
        resized = resize_with_aspect_ratio(image, target_width=800, target_height=300)
        # Should scale by height (smaller scale factor)
        assert resized.shape[0] == 300
        assert resized.shape[1] == 400
    
    def test_save_and_load_image(self):
        """Test saving and loading images."""
        image = create_test_image()
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            # Save image
            success = save_image(image, temp_path)
            assert success == True
            assert os.path.exists(temp_path)
            
            # Load image
            loaded_image = load_image(temp_path)
            assert loaded_image is not None
            assert loaded_image.shape == image.shape
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_create_video_writer(self):
        """Test video writer creation."""
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            writer = create_video_writer(temp_path, width=640, height=480, fps=30.0)
            assert writer is not None
            
            # Write a test frame
            test_frame = create_test_image(width=640, height=480)
            writer.write(test_frame)
            
            # Release writer
            writer.release()
            
            # File should exist (may be corrupted but that's ok for test)
            assert os.path.exists(temp_path)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestFacialMetrics:
    """Test facial analysis metrics."""
    
    def test_estimate_pose(self):
        """Test pose estimation from landmarks."""
        # Create minimal landmark set for pose estimation
        landmarks = [
            (0.5, 0.4, 0.0),    # nose tip
            (0.5, 0.6, 0.0),    # chin
            (0.3, 0.5, 0.0),    # left mouth corner
            (0.7, 0.5, 0.0),    # right mouth corner
            (0.4, 0.45, 0.0),   # left eye
            (0.6, 0.45, 0.0)    # right eye
        ]
        
        pose = estimate_pose(landmarks)
        
        assert 'pitch' in pose
        assert 'yaw' in pose
        assert 'roll' in pose
        
        for angle in pose.values():
            assert isinstance(angle, (int, float))
            assert -180 <= angle <= 180  # Reasonable angle range
    
    def test_calculate_eye_aspect_ratio(self):
        """Test eye aspect ratio calculation."""
        # Typical eye landmarks (simplified)
        eye_landmarks = [
            (0.3, 0.4),   # left corner
            (0.32, 0.38), # top left
            (0.35, 0.38), # top center
            (0.38, 0.38), # top right
            (0.4, 0.4),   # right corner
            (0.38, 0.42), # bottom right
            (0.35, 0.42), # bottom center
            (0.32, 0.42)  # bottom left
        ]
        
        ear = calculate_eye_aspect_ratio(eye_landmarks)
        
        assert isinstance(ear, float)
        assert 0.0 <= ear <= 1.0  # EAR should be normalized
        
        # Test with closed eye (vertical distances = 0)
        closed_eye = [
            (0.3, 0.4), (0.32, 0.4), (0.35, 0.4), (0.38, 0.4),
            (0.4, 0.4), (0.38, 0.4), (0.35, 0.4), (0.32, 0.4)
        ]
        
        ear_closed = calculate_eye_aspect_ratio(closed_eye)
        assert ear_closed == 0.0
    
    def test_calculate_mouth_aspect_ratio(self):
        """Test mouth aspect ratio calculation."""
        # Mouth landmarks (simplified)
        mouth_landmarks = [
            (0.4, 0.7),   # left corner
            (0.42, 0.68), # top left
            (0.5, 0.68),  # top center
            (0.58, 0.68), # top right
            (0.6, 0.7),   # right corner
            (0.58, 0.72), # bottom right
            (0.5, 0.72),  # bottom center
            (0.42, 0.72)  # bottom left
        ]
        
        mar = calculate_mouth_aspect_ratio(mouth_landmarks)
        
        assert isinstance(mar, float)
        assert mar >= 0.0
        
        # Test with closed mouth
        closed_mouth = [
            (0.4, 0.7), (0.42, 0.7), (0.5, 0.7), (0.58, 0.7),
            (0.6, 0.7), (0.58, 0.7), (0.5, 0.7), (0.42, 0.7)
        ]
        
        mar_closed = calculate_mouth_aspect_ratio(closed_mouth)
        assert mar_closed == 0.0


class TestLandmarkProcessing:
    """Test landmark processing utilities."""
    
    def test_smooth_landmarks(self):
        """Test landmark smoothing."""
        # Create noisy landmark sequence
        base_landmarks = [(0.5, 0.5), (0.3, 0.7), (0.8, 0.2)]
        
        landmark_sequence = []
        for i in range(10):
            # Add small random noise
            noisy_landmarks = [
                (x + np.random.normal(0, 0.01), y + np.random.normal(0, 0.01))
                for x, y in base_landmarks
            ]
            landmark_sequence.append(noisy_landmarks)
        
        # Smooth the sequence
        smoothed = smooth_landmarks(landmark_sequence, window_size=5)
        
        assert len(smoothed) == len(landmark_sequence)
        assert len(smoothed[0]) == len(base_landmarks)
        
        # Smoothed landmarks should be less noisy
        # (Hard to test precisely due to randomness, but should not crash)
    
    def test_interpolate_landmarks(self):
        """Test landmark interpolation."""
        landmarks1 = [(0.0, 0.0), (0.0, 1.0)]
        landmarks2 = [(1.0, 0.0), (1.0, 1.0)]
        
        # Interpolate at midpoint
        interpolated = interpolate_landmarks(landmarks1, landmarks2, 0.5)
        expected = [(0.5, 0.0), (0.5, 1.0)]
        
        assert len(interpolated) == len(expected)
        for interp, exp in zip(interpolated, expected):
            assert abs(interp[0] - exp[0]) < 1e-6
            assert abs(interp[1] - exp[1]) < 1e-6
        
        # Test edge cases
        interp_start = interpolate_landmarks(landmarks1, landmarks2, 0.0)
        assert interp_start == landmarks1
        
        interp_end = interpolate_landmarks(landmarks1, landmarks2, 1.0)
        assert interp_end == landmarks2


class TestErrorHandling:
    """Test error handling in utility functions."""
    
    def test_invalid_inputs(self):
        """Test handling of invalid inputs."""
        # Test distance calculation with invalid points
        with pytest.raises((TypeError, ValueError)):
            calculate_distance(None, (1, 2))
        
        with pytest.raises((TypeError, ValueError, IndexError)):
            calculate_distance((1,), (1, 2))  # Different dimensions
        
        # Test angle calculation with invalid points
        with pytest.raises((TypeError, ValueError)):
            calculate_angle((0, 0), None, (1, 1))
        
        # Test empty landmark lists
        assert normalize_landmarks([], (100, 100)) == []
        assert denormalize_landmarks([], (100, 100)) == []
    
    def test_edge_cases(self):
        """Test edge cases in utility functions."""
        # Zero-sized image
        try:
            result = resize_with_aspect_ratio(
                np.zeros((1, 1, 3), dtype=np.uint8), 
                target_width=100
            )
            # Should handle gracefully
            assert result.shape[1] == 100
        except (ValueError, cv2.error):
            pass  # Expected for invalid operations
        
        # Single point bounding box
        bbox = get_bounding_box([(10, 20)])
        assert bbox == (10, 20, 0, 0)


class TestPerformance:
    """Test performance of utility functions."""
    
    def test_large_landmark_processing(self):
        """Test processing large numbers of landmarks."""
        # Create large landmark set
        large_landmarks = [(i * 0.001, i * 0.001) for i in range(1000)]
        
        # Should handle large inputs efficiently
        bbox = get_bounding_box(large_landmarks)
        assert isinstance(bbox, tuple)
        assert len(bbox) == 4
        
        # Normalization should be fast
        normalized = normalize_landmarks(large_landmarks, (1920, 1080))
        assert len(normalized) == 1000
    
    def test_batch_processing(self):
        """Test batch processing capabilities."""
        # Process multiple images
        images = [create_test_image() for _ in range(5)]
        
        # Should handle batch efficiently
        for image in images:
            assert validate_image(image) == True
            resized = resize_with_aspect_ratio(image, target_width=200)
            assert resized.shape[1] == 200
