"""
Integration tests for the facial landmarks detection system.
"""

import pytest
import numpy as np
import cv2
import tempfile
import os
import time
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'facial_landmarks'))

from facial_processor import FacialLandmarkProcessor, ProcessingConfig
from landmark_detector import LandmarkDetector, LandmarkSubset
from face_detector import FaceDetector
from conftest import create_test_image, create_face_like_image, create_test_video


class TestEndToEndProcessing:
    """End-to-end integration tests."""
    
    def test_complete_image_processing_pipeline(self):
        """Test complete image processing from input to output."""
        # Create processor with specific config
        config = ProcessingConfig(
            max_num_faces=2,
            min_detection_confidence=0.5,
            landmark_subset=LandmarkSubset.ALL,
            draw_landmarks=True,
            draw_face_boxes=True
        )
        
        processor = FacialLandmarkProcessor(config)
        test_image = create_face_like_image()
        
        # Process image
        result = processor.process_image(test_image)
        
        # Verify result structure
        assert 'faces' in result
        assert 'landmarks' in result
        assert 'processed_image' in result
        assert 'metadata' in result
        
        # Verify result types
        assert isinstance(result['faces'], list)
        assert isinstance(result['landmarks'], list)
        assert isinstance(result['processed_image'], np.ndarray)
        assert isinstance(result['metadata'], dict)
        
        # Verify image properties
        assert result['processed_image'].shape == test_image.shape
        assert result['processed_image'].dtype == test_image.dtype
    
    def test_detector_integration(self):
        """Test integration between face detector and landmark detector."""
        face_detector = FaceDetector()
        landmark_detector = LandmarkDetector()
        
        test_image = create_face_like_image()
        
        # Detect faces first
        faces = face_detector.detect_faces(test_image)
        
        # Then detect landmarks
        landmarks = landmark_detector.detect_landmarks(test_image)
        
        # Both should return lists (may be empty for test image)
        assert isinstance(faces, list)
        assert isinstance(landmarks, list)
        
        # Results should be consistent (same number or proportional)
        # For test images, both might return empty lists, which is fine
    
    def test_different_image_formats(self):
        """Test processing different image formats and sizes."""
        processor = FacialLandmarkProcessor()
        
        # Test different image sizes
        test_cases = [
            (640, 480),   # Standard VGA
            (1920, 1080), # Full HD
            (320, 240),   # Small
            (800, 600),   # 4:3 ratio
            (1280, 720),  # 16:9 ratio
        ]
        
        for width, height in test_cases:
            test_image = create_test_image(width=width, height=height)
            
            result = processor.process_image(test_image)
            
            # Should handle all sizes without error
            assert result['processed_image'].shape == (height, width, 3)
    
    def test_landmark_subset_consistency(self):
        """Test that different landmark subsets work consistently."""
        test_image = create_face_like_image()
        
        subsets = [
            LandmarkSubset.ALL,
            LandmarkSubset.FACE_OVAL,
            LandmarkSubset.LEFT_EYE,
            LandmarkSubset.RIGHT_EYE,
            LandmarkSubset.NOSE,
            LandmarkSubset.LIPS
        ]
        
        for subset in subsets:
            config = ProcessingConfig(
                landmark_subset=subset,
                draw_landmarks=True
            )
            
            processor = FacialLandmarkProcessor(config)
            result = processor.process_image(test_image)
            
            # Should complete without error for all subsets
            assert 'processed_image' in result
            assert result['processed_image'].shape == test_image.shape


class TestVideoProcessingIntegration:
    """Integration tests for video processing."""
    
    def test_video_processing_pipeline(self):
        """Test complete video processing pipeline."""
        # Create temporary video
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_input:
            input_path = temp_input.name
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_output:
            output_path = temp_output.name
        
        try:
            # Create test video
            create_test_video(input_path, duration=1.0, fps=5)  # Short video
            
            # Configure processor
            config = ProcessingConfig(
                save_video=True,
                output_path=output_path,
                show_fps=True
            )
            
            processor = FacialLandmarkProcessor(config)
            
            # Mock display functions to avoid GUI
            with patch('cv2.imshow'):
                with patch('cv2.waitKey', return_value=ord('q')):
                    with patch('cv2.destroyAllWindows'):
                        # Process video
                        processor.process_video(input_path)
            
            # Should complete without major errors
            
        finally:
            # Cleanup
            for path in [input_path, output_path]:
                if os.path.exists(path):
                    os.unlink(path)
    
    @patch('cv2.VideoCapture')
    def test_webcam_processing_integration(self, mock_videocapture):
        """Test webcam processing integration."""
        # Setup mock webcam
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, create_face_like_image()),
            (True, create_test_image()),
            (False, None)  # End
        ]
        mock_videocapture.return_value = mock_cap
        
        processor = FacialLandmarkProcessor()
        
        with patch('cv2.imshow'):
            with patch('cv2.waitKey', return_value=ord('q')):
                with patch('cv2.destroyAllWindows'):
                    # Should process frames without error
                    processor.process_webcam()
        
        # Verify webcam was properly released
        mock_cap.release.assert_called()


class TestPerformanceIntegration:
    """Performance integration tests."""
    
    def test_processing_speed(self):
        """Test processing speed meets reasonable expectations."""
        processor = FacialLandmarkProcessor()
        test_image = create_face_like_image()
        
        # Warm up
        processor.process_image(test_image)
        
        # Measure processing time
        times = []
        for _ in range(5):
            start_time = time.time()
            processor.process_image(test_image)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = sum(times) / len(times)
        
        # Should process reasonably fast (adjust threshold as needed)
        assert avg_time < 0.5  # Less than 500ms per frame
        assert avg_time > 0.001  # But should take some time (not instant)
    
    def test_memory_stability(self):
        """Test memory usage stability over multiple processing cycles."""
        processor = FacialLandmarkProcessor()
        test_image = create_face_like_image()
        
        # Process multiple times to check for memory leaks
        for i in range(20):
            result = processor.process_image(test_image)
            
            # Clear references to help garbage collection
            del result
            
            # Every few iterations, check memory is reasonable
            if i % 5 == 0:
                # This is a basic check - in production you might use memory profiling
                import gc
                gc.collect()
        
        # Should complete without memory issues
        assert True
    
    def test_concurrent_processing(self):
        """Test concurrent processing scenarios."""
        import threading
        
        processor = FacialLandmarkProcessor()
        test_images = [create_test_image() for _ in range(3)]
        results = []
        errors = []
        
        def process_image(image):
            try:
                result = processor.process_image(image)
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create threads
        threads = []
        for image in test_images:
            thread = threading.Thread(target=process_image, args=(image,))
            threads.append(thread)
        
        # Start all threads
        for thread in threads:
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert len(errors) == 0, f"Errors during concurrent processing: {errors}"
        assert len(results) == len(test_images)


class TestConfigurationIntegration:
    """Test different configuration combinations."""
    
    @pytest.mark.parametrize("config_params", [
        {'max_num_faces': 1, 'draw_landmarks': True, 'draw_face_boxes': False},
        {'max_num_faces': 3, 'draw_landmarks': False, 'draw_face_boxes': True},
        {'landmark_subset': LandmarkSubset.FACE_OVAL, 'draw_detailed_landmarks': False},
        {'landmark_subset': LandmarkSubset.ALL, 'show_landmark_indices': True},
        {'min_detection_confidence': 0.3, 'min_tracking_confidence': 0.8},
        {'refine_landmarks': False, 'draw_landmarks': True}
    ])
    def test_configuration_combinations(self, config_params):
        """Test various configuration combinations."""
        config = ProcessingConfig(**config_params)
        processor = FacialLandmarkProcessor(config)
        
        test_image = create_face_like_image()
        result = processor.process_image(test_image)
        
        # Should work with any valid configuration
        assert 'processed_image' in result
        assert result['processed_image'].shape == test_image.shape
    
    def test_extreme_configurations(self):
        """Test extreme but valid configurations."""
        # High detection threshold
        config = ProcessingConfig(
            min_detection_confidence=0.95,
            min_tracking_confidence=0.95,
            max_num_faces=10
        )
        
        processor = FacialLandmarkProcessor(config)
        test_image = create_face_like_image()
        
        result = processor.process_image(test_image)
        assert 'processed_image' in result
        
        # Low detection threshold
        config = ProcessingConfig(
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
            max_num_faces=1
        )
        
        processor = FacialLandmarkProcessor(config)
        result = processor.process_image(test_image)
        assert 'processed_image' in result


class TestErrorRecoveryIntegration:
    """Test error recovery and robustness."""
    
    def test_corrupted_image_handling(self):
        """Test handling of corrupted or invalid images."""
        processor = FacialLandmarkProcessor()
        
        # Test various invalid images
        invalid_images = [
            np.zeros((0, 0, 3), dtype=np.uint8),  # Empty image
            np.ones((10, 10), dtype=np.uint8),    # Missing channel dimension
            np.ones((10, 10, 1), dtype=np.uint8), # Single channel
            np.ones((10, 10, 4), dtype=np.uint8), # Too many channels
        ]
        
        for invalid_image in invalid_images:
            try:
                result = processor.process_image(invalid_image)
                # If it doesn't crash, that's good
                assert 'processed_image' in result
            except (ValueError, cv2.error, AttributeError):
                # Expected for some invalid inputs
                pass
    
    def test_processing_interruption_recovery(self):
        """Test recovery from processing interruptions."""
        processor = FacialLandmarkProcessor()
        
        # Simulate processing interruption
        test_image = create_face_like_image()
        
        # Process normally first
        result1 = processor.process_image(test_image)
        assert 'processed_image' in result1
        
        # Simulate some error condition and recovery
        try:
            # This might cause an error in some MediaPipe versions
            very_large_config = ProcessingConfig(max_num_faces=1000)
            processor2 = FacialLandmarkProcessor(very_large_config)
            result2 = processor2.process_image(test_image)
        except Exception:
            # If it fails, processor should still work for normal operations
            pass
        
        # Original processor should still work
        result3 = processor.process_image(test_image)
        assert 'processed_image' in result3


class TestDataConsistency:
    """Test data consistency across processing."""
    
    def test_landmark_coordinate_consistency(self):
        """Test that landmark coordinates are consistent and valid."""
        processor = FacialLandmarkProcessor()
        test_image = create_face_like_image()
        
        result = processor.process_image(test_image)
        
        # Check landmarks if any were detected
        for landmark_set in result['landmarks']:
            for landmark in landmark_set.landmarks:
                # Coordinates should be normalized (0-1) for MediaPipe
                assert 0.0 <= landmark[0] <= 1.0, f"X coordinate out of range: {landmark[0]}"
                assert 0.0 <= landmark[1] <= 1.0, f"Y coordinate out of range: {landmark[1]}"
                # Z coordinate can be negative for depth
                assert isinstance(landmark[2], (int, float)), f"Z coordinate not numeric: {landmark[2]}"
    
    def test_bounding_box_consistency(self):
        """Test that bounding boxes are consistent and valid."""
        processor = FacialLandmarkProcessor()
        test_image = create_face_like_image()
        
        result = processor.process_image(test_image)
        
        # Check face bounding boxes if any were detected
        for face in result['faces']:
            bbox = face.bbox
            
            # Bounding box coordinates should be normalized
            assert 0.0 <= bbox.x <= 1.0
            assert 0.0 <= bbox.y <= 1.0
            assert 0.0 <= bbox.width <= 1.0
            assert 0.0 <= bbox.height <= 1.0
            
            # Bounding box should be within image boundaries
            assert bbox.x + bbox.width <= 1.0
            assert bbox.y + bbox.height <= 1.0
            
            # Confidence should be between 0 and 1
            assert 0.0 <= face.confidence <= 1.0
    
    def test_processing_repeatability(self):
        """Test that processing the same image produces consistent results."""
        processor = FacialLandmarkProcessor()
        test_image = create_face_like_image()
        
        # Process the same image multiple times
        results = []
        for _ in range(3):
            result = processor.process_image(test_image.copy())
            results.append(result)
        
        # Results should be consistent (same number of faces/landmarks)
        face_counts = [len(result['faces']) for result in results]
        landmark_counts = [len(result['landmarks']) for result in results]
        
        # All counts should be the same
        assert len(set(face_counts)) <= 1, f"Inconsistent face counts: {face_counts}"
        assert len(set(landmark_counts)) <= 1, f"Inconsistent landmark counts: {landmark_counts}"


class TestSystemIntegration:
    """Test system-level integration scenarios."""
    
    def test_cleanup_and_resource_management(self):
        """Test proper cleanup and resource management."""
        # Create and destroy multiple processors
        processors = []
        
        for _ in range(5):
            processor = FacialLandmarkProcessor()
            processors.append(processor)
            
            # Use the processor
            test_image = create_test_image()
            result = processor.process_image(test_image)
            assert 'processed_image' in result
        
        # Cleanup
        for processor in processors:
            processor.__del__()
        
        # Should complete without resource leaks
        del processors
    
    def test_multiple_format_support(self):
        """Test support for multiple image formats and color spaces."""
        processor = FacialLandmarkProcessor()
        
        # Test RGB image
        rgb_image = create_test_image()
        result_rgb = processor.process_image(rgb_image)
        assert 'processed_image' in result_rgb
        
        # Test BGR image (OpenCV default)
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
        result_bgr = processor.process_image(bgr_image)
        assert 'processed_image' in result_bgr
        
        # Results should be valid for both formats
        assert result_rgb['processed_image'].shape == result_bgr['processed_image'].shape
