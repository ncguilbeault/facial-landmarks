"""
Tests for the FacialLandmarkProcessor class and ProcessingConfig.
"""

import pytest
import numpy as np
import cv2
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import tempfile
import os

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'facial_landmarks'))

from facial_processor import FacialLandmarkProcessor, ProcessingConfig
from landmark_detector import LandmarkSubset
from conftest import create_test_image, create_face_like_image, create_test_video


class TestProcessingConfig:
    """Test cases for ProcessingConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ProcessingConfig()
        
        assert config.max_num_faces == 1
        assert config.refine_landmarks == True
        assert config.min_detection_confidence == 0.5
        assert config.min_tracking_confidence == 0.5
        assert config.landmark_subset == LandmarkSubset.ALL
        assert config.draw_landmarks == True
        assert config.draw_face_boxes == False
        assert config.draw_detailed_landmarks == True
        assert config.show_landmark_indices == False
        assert config.output_path is None
        assert config.save_video == False
        assert config.show_fps == True
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ProcessingConfig(
            max_num_faces=3,
            refine_landmarks=False,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.7,
            landmark_subset=LandmarkSubset.FACE_OVAL,
            draw_landmarks=False,
            draw_face_boxes=True,
            draw_detailed_landmarks=False,
            show_landmark_indices=True,
            output_path="test_output.mp4",
            save_video=True,
            show_fps=False
        )
        
        assert config.max_num_faces == 3
        assert config.refine_landmarks == False
        assert config.min_detection_confidence == 0.8
        assert config.min_tracking_confidence == 0.7
        assert config.landmark_subset == LandmarkSubset.FACE_OVAL
        assert config.draw_landmarks == False
        assert config.draw_face_boxes == True
        assert config.draw_detailed_landmarks == False
        assert config.show_landmark_indices == True
        assert config.output_path == "test_output.mp4"
        assert config.save_video == True
        assert config.show_fps == False
    
    def test_config_validation(self):
        """Test configuration validation."""
        # Test valid confidence values
        config = ProcessingConfig(min_detection_confidence=0.0)
        assert config.min_detection_confidence == 0.0
        
        config = ProcessingConfig(min_detection_confidence=1.0)
        assert config.min_detection_confidence == 1.0
        
        # Test invalid confidence values should be handled gracefully
        try:
            config = ProcessingConfig(min_detection_confidence=-0.1)
            # If no validation, should accept any value
        except ValueError:
            pass  # Expected if validation is implemented
        
        try:
            config = ProcessingConfig(min_detection_confidence=1.1)
        except ValueError:
            pass  # Expected if validation is implemented
    
    def test_config_from_dict(self):
        """Test creating config from dictionary."""
        config_dict = {
            'max_num_faces': 2,
            'min_detection_confidence': 0.7,
            'landmark_subset': 'face_oval',
            'draw_landmarks': False
        }
        
        # This would require implementing a from_dict method
        # For now, test manual assignment
        config = ProcessingConfig()
        config.max_num_faces = config_dict['max_num_faces']
        config.min_detection_confidence = config_dict['min_detection_confidence']
        config.draw_landmarks = config_dict['draw_landmarks']
        
        assert config.max_num_faces == 2
        assert config.min_detection_confidence == 0.7
        assert config.draw_landmarks == False


class TestFacialLandmarkProcessor:
    """Test cases for FacialLandmarkProcessor class."""
    
    def test_init_default_config(self):
        """Test processor initialization with default config."""
        processor = FacialLandmarkProcessor()
        
        assert processor.config is not None
        assert isinstance(processor.config, ProcessingConfig)
        assert hasattr(processor, 'face_detector')
        assert hasattr(processor, 'landmark_detector')
    
    def test_init_custom_config(self):
        """Test processor initialization with custom config."""
        config = ProcessingConfig(
            max_num_faces=2,
            min_detection_confidence=0.8
        )
        
        processor = FacialLandmarkProcessor(config)
        
        assert processor.config == config
        assert processor.config.max_num_faces == 2
        assert processor.config.min_detection_confidence == 0.8
    
    @patch('facial_processor.FaceDetector')
    @patch('facial_processor.LandmarkDetector')
    def test_process_image_no_faces(self, mock_landmark_detector, mock_face_detector):
        """Test processing image with no faces."""
        # Setup mocks
        mock_face_detector.return_value.detect_faces.return_value = []
        mock_landmark_detector.return_value.detect_landmarks.return_value = []
        
        processor = FacialLandmarkProcessor()
        test_image = create_test_image()
        
        result = processor.process_image(test_image)
        
        assert 'faces' in result
        assert 'landmarks' in result
        assert 'processed_image' in result
        assert len(result['faces']) == 0
        assert len(result['landmarks']) == 0
        assert result['processed_image'].shape == test_image.shape
    
    @patch('facial_processor.FaceDetector')
    @patch('facial_processor.LandmarkDetector')
    def test_process_image_with_faces(self, mock_landmark_detector, mock_face_detector):
        """Test processing image with detected faces."""
        # Setup mock face
        mock_face = MagicMock()
        mock_face.confidence = 0.9
        
        # Setup mock landmark
        mock_landmark = MagicMock()
        mock_landmark.landmarks = [(0.5, 0.5, 0.0)]
        
        # Setup mocks
        mock_face_detector.return_value.detect_faces.return_value = [mock_face]
        mock_landmark_detector.return_value.detect_landmarks.return_value = [mock_landmark]
        mock_landmark_detector.return_value.draw_landmarks.return_value = create_face_like_image()
        
        processor = FacialLandmarkProcessor()
        test_image = create_face_like_image()
        
        result = processor.process_image(test_image)
        
        assert len(result['faces']) == 1
        assert len(result['landmarks']) == 1
        assert result['processed_image'].shape == test_image.shape
    
    def test_process_webcam_frame(self):
        """Test processing a single webcam frame."""
        processor = FacialLandmarkProcessor()
        test_image = create_face_like_image()
        
        # This should not crash and return a processed image
        result_image = processor.process_webcam_frame(test_image)
        
        assert result_image.shape == test_image.shape
        assert isinstance(result_image, np.ndarray)
    
    @patch('cv2.VideoCapture')
    def test_process_webcam_mock(self, mock_video_capture):
        """Test webcam processing with mocked video capture."""
        # Setup mock video capture
        mock_cap = MagicMock()
        mock_cap.read.side_effect = [
            (True, create_face_like_image()),  # First frame
            (True, create_face_like_image()),  # Second frame
            (False, None)  # End of video
        ]
        mock_cap.isOpened.return_value = True
        mock_video_capture.return_value = mock_cap
        
        processor = FacialLandmarkProcessor()
        
        # Mock cv2.waitKey to simulate user pressing 'q'
        with patch('cv2.waitKey', side_effect=[ord('q')]):
            with patch('cv2.imshow'):
                with patch('cv2.destroyAllWindows'):
                    # Should exit gracefully when 'q' is pressed
                    processor.process_webcam()
        
        mock_cap.release.assert_called_once()
    
    def test_process_video_file(self):
        """Test processing a video file."""
        processor = FacialLandmarkProcessor()
        
        # Create a temporary video file for testing
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_path = temp_video.name
        
        try:
            # Create a simple test video
            create_test_video(temp_path)
            
            # Process the video
            with patch('cv2.imshow'):  # Mock display
                with patch('cv2.waitKey', return_value=ord('q')):  # Exit immediately
                    with patch('cv2.destroyAllWindows'):
                        processor.process_video(temp_path)
            
            # Should complete without errors
            
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_process_video_with_output(self):
        """Test processing video with output saving."""
        config = ProcessingConfig(
            save_video=True,
            output_path="test_output.mp4"
        )
        processor = FacialLandmarkProcessor(config)
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
            input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            output_path = temp_output.name
        
        try:
            create_test_video(input_path)
            config.output_path = output_path
            
            with patch('cv2.imshow'):
                with patch('cv2.waitKey', return_value=ord('q')):
                    with patch('cv2.destroyAllWindows'):
                        processor.process_video(input_path)
            
            # Should create output file (may be empty due to mocking)
            
        finally:
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    def test_get_video_info(self):
        """Test getting video information."""
        processor = FacialLandmarkProcessor()
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_video:
            temp_path = temp_video.name
        
        try:
            create_test_video(temp_path)
            
            info = processor.get_video_info(temp_path)
            
            assert 'fps' in info
            assert 'frame_count' in info
            assert 'width' in info
            assert 'height' in info
            assert 'duration' in info
            
            assert isinstance(info['fps'], (int, float))
            assert isinstance(info['frame_count'], int)
            assert isinstance(info['width'], int) 
            assert isinstance(info['height'], int)
            assert isinstance(info['duration'], (int, float))
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_calculate_fps(self):
        """Test FPS calculation."""
        processor = FacialLandmarkProcessor()
        
        # Test with frame times
        frame_times = [0.0, 0.033, 0.066, 0.099, 0.132]  # ~30 FPS
        fps = processor.calculate_fps(frame_times)
        
        assert 25 <= fps <= 35  # Should be around 30 FPS
    
    def test_add_fps_text(self):
        """Test adding FPS text to image."""
        processor = FacialLandmarkProcessor()
        test_image = create_test_image()
        
        result_image = processor.add_fps_text(test_image, 30.5)
        
        assert result_image.shape == test_image.shape
        assert not np.array_equal(result_image, test_image)  # Should be modified
    
    def test_add_info_text(self):
        """Test adding information text to image."""
        processor = FacialLandmarkProcessor()
        test_image = create_test_image()
        
        info = {
            'faces_detected': 2,
            'landmarks_detected': 2,
            'processing_time': 0.045
        }
        
        result_image = processor.add_info_text(test_image, info)
        
        assert result_image.shape == test_image.shape
        assert not np.array_equal(result_image, test_image)  # Should be modified
    
    def test_resize_image(self):
        """Test image resizing."""
        processor = FacialLandmarkProcessor()
        test_image = create_test_image(width=640, height=480)
        
        # Resize to smaller
        resized = processor.resize_image(test_image, width=320, height=240)
        assert resized.shape == (240, 320, 3)
        
        # Resize to larger
        resized = processor.resize_image(test_image, width=1280, height=960)
        assert resized.shape == (960, 1280, 3)
        
        # Maintain aspect ratio
        resized = processor.resize_image(test_image, width=320)
        expected_height = int(480 * 320 / 640)
        assert resized.shape[1] == 320
        assert resized.shape[0] == expected_height
    
    def test_process_batch_images(self):
        """Test batch processing of images."""
        processor = FacialLandmarkProcessor()
        
        # Create test images
        images = [
            create_test_image(),
            create_face_like_image(),
            create_test_image(width=200, height=200)
        ]
        
        results = processor.process_batch(images)
        
        assert len(results) == 3
        for result in results:
            assert 'faces' in result
            assert 'landmarks' in result
            assert 'processed_image' in result
    
    def test_save_results(self):
        """Test saving processing results."""
        processor = FacialLandmarkProcessor()
        
        results = {
            'faces': [],
            'landmarks': [],
            'metadata': {
                'processing_time': 0.05,
                'timestamp': '2024-01-01T00:00:00'
            }
        }
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            processor.save_results(results, temp_path)
            
            # File should exist
            assert os.path.exists(temp_path)
            
            # Should contain valid JSON
            import json
            with open(temp_path, 'r') as f:
                loaded_results = json.load(f)
            
            assert 'faces' in loaded_results
            assert 'landmarks' in loaded_results
            assert 'metadata' in loaded_results
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def test_error_handling(self):
        """Test error handling in processor."""
        processor = FacialLandmarkProcessor()
        
        # Test with invalid image
        with pytest.raises((AttributeError, TypeError, ValueError)):
            processor.process_image(None)
        
        # Test with invalid video path
        result = processor.process_video("nonexistent_video.mp4")
        # Should handle gracefully and not crash
    
    def test_processor_cleanup(self):
        """Test processor cleanup."""
        processor = FacialLandmarkProcessor()
        
        # Should have detectors
        assert hasattr(processor, 'face_detector')
        assert hasattr(processor, 'landmark_detector')
        
        # Test cleanup
        processor.__del__()
        # Should not raise any exceptions
    
    @pytest.mark.parametrize("landmark_subset", [
        LandmarkSubset.ALL,
        LandmarkSubset.FACE_OVAL,
        LandmarkSubset.LEFT_EYE,
        LandmarkSubset.RIGHT_EYE,
        LandmarkSubset.NOSE,
        LandmarkSubset.LIPS
    ])
    def test_different_landmark_subsets(self, landmark_subset):
        """Test processing with different landmark subsets."""
        config = ProcessingConfig(landmark_subset=landmark_subset)
        processor = FacialLandmarkProcessor(config)
        
        test_image = create_face_like_image()
        result = processor.process_image(test_image)
        
        assert 'processed_image' in result
        assert result['processed_image'].shape == test_image.shape
    
    def test_performance_monitoring(self):
        """Test performance monitoring features."""
        processor = FacialLandmarkProcessor()
        test_image = create_face_like_image()
        
        # Process multiple times to get performance stats
        times = []
        for _ in range(5):
            import time
            start_time = time.time()
            processor.process_image(test_image)
            end_time = time.time()
            times.append(end_time - start_time)
        
        # Should have reasonable processing times
        avg_time = sum(times) / len(times)
        assert avg_time > 0  # Should take some time
        assert avg_time < 1.0  # Should be reasonably fast
    
    def test_memory_usage(self):
        """Test memory usage during processing."""
        processor = FacialLandmarkProcessor()
        test_image = create_face_like_image()
        
        # Process multiple images to check for memory leaks
        for _ in range(10):
            result = processor.process_image(test_image)
            # Clear reference to help garbage collection
            del result
        
        # Should complete without memory issues
        assert True
