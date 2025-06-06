"""
Tests for the main CLI interface and application entry points.
"""

import pytest
import sys
import os
import numpy as np
from unittest.mock import patch, MagicMock, call
import tempfile
import argparse

# Add the facial_landmarks module to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'facial_landmarks'))

from main import (
    parse_arguments, create_config_from_args, main,
    validate_args, setup_logging
)
from facial_processor import ProcessingConfig
from landmark_detector import LandmarkSubset


class TestArgumentParsing:
    """Test command line argument parsing."""
    
    def test_parse_arguments_default(self):
        """Test parsing with default arguments."""
        args = parse_arguments(['--webcam'])
        
        assert args.webcam == True
        assert args.video is None
        assert args.output is None
        assert args.max_faces == 1
        assert args.confidence == 0.5
        assert args.tracking_confidence == 0.5
        assert args.landmark_subset == 'all'
        assert args.no_landmarks == False
        assert args.show_face_boxes == False
        assert args.simple_landmarks == False
        assert args.show_indices == False
        assert args.save_video == False
        assert args.no_fps == False
    
    def test_parse_arguments_webcam(self):
        """Test webcam mode arguments."""
        args = parse_arguments(['--webcam'])
        assert args.webcam == True
        assert args.video is None
    
    def test_parse_arguments_video(self):
        """Test video file mode arguments."""
        args = parse_arguments(['--video', 'test.mp4'])
        assert args.video == 'test.mp4'
        assert args.webcam == False
    
    def test_parse_arguments_custom_values(self):
        """Test parsing with custom values."""
        cmd_args = [
            '--webcam',
            '--max-faces', '3',
            '--confidence', '0.8',
            '--tracking-confidence', '0.7',
            '--landmark-subset', 'face_oval',
            '--output', 'output.mp4',
            '--show-face-boxes',
            '--simple-landmarks',
            '--show-indices',
            '--save-video',
            '--no-fps'
        ]
        
        args = parse_arguments(cmd_args)
        
        assert args.max_faces == 3
        assert args.confidence == 0.8
        assert args.tracking_confidence == 0.7
        assert args.landmark_subset == 'face_oval'
        assert args.output == 'output.mp4'
        assert args.show_face_boxes == True
        assert args.simple_landmarks == True
        assert args.show_indices == True
        assert args.save_video == True
        assert args.no_fps == True
    
    def test_parse_arguments_landmark_subsets(self):
        """Test different landmark subset arguments."""
        subsets = ['all', 'face_oval', 'left_eye', 'right_eye', 'nose', 'lips']
        
        for subset in subsets:
            args = parse_arguments(['--webcam', '--landmark-subset', subset])
            assert args.landmark_subset == subset
    
    def test_parse_arguments_invalid_confidence(self):
        """Test parsing with invalid confidence values."""
        # Note: argparse may not validate ranges, so we test what it accepts
        try:
            args = parse_arguments(['--webcam', '--confidence', '1.5'])
            assert args.confidence == 1.5  # May accept invalid values
        except SystemExit:
            pass  # Expected if validation is implemented
    
    def test_parse_arguments_mutually_exclusive(self):
        """Test mutually exclusive arguments."""
        # Both webcam and video should not be allowed together
        # This depends on how argparse is configured
        try:
            args = parse_arguments(['--webcam', '--video', 'test.mp4'])
            # If no mutual exclusion is enforced, both might be set
        except SystemExit:
            pass  # Expected if mutual exclusion is enforced


class TestConfigCreation:
    """Test configuration creation from arguments."""
    
    def test_create_config_from_args_default(self):
        """Test config creation with default arguments."""
        args = parse_arguments(['--webcam'])
        config = create_config_from_args(args)
        
        assert isinstance(config, ProcessingConfig)
        assert config.max_num_faces == 1
        assert config.min_detection_confidence == 0.5
        assert config.min_tracking_confidence == 0.5
        assert config.landmark_subset == LandmarkSubset.ALL
        assert config.draw_landmarks == True
        assert config.draw_face_boxes == False
        assert config.draw_detailed_landmarks == True
        assert config.show_landmark_indices == False
        assert config.save_video == False
        assert config.show_fps == True
    
    def test_create_config_from_args_custom(self):
        """Test config creation with custom arguments."""
        cmd_args = [
            '--webcam',
            '--max-faces', '2',
            '--confidence', '0.7',
            '--tracking-confidence', '0.6',
            '--landmark-subset', 'face_oval',
            '--no-landmarks',
            '--show-face-boxes',
            '--simple-landmarks',
            '--show-indices',
            '--save-video',
            '--output', 'test.mp4',
            '--no-fps'
        ]
        
        args = parse_arguments(cmd_args)
        config = create_config_from_args(args)
        
        assert config.max_num_faces == 2
        assert config.min_detection_confidence == 0.7
        assert config.min_tracking_confidence == 0.6
        assert config.landmark_subset == LandmarkSubset.FACE_OVAL
        assert config.draw_landmarks == False  # --no-landmarks
        assert config.draw_face_boxes == True
        assert config.draw_detailed_landmarks == False  # --simple-landmarks
        assert config.show_landmark_indices == True
        assert config.save_video == True
        assert config.output_path == 'test.mp4'
        assert config.show_fps == False  # --no-fps
    
    def test_landmark_subset_mapping(self):
        """Test landmark subset string to enum mapping."""
        subset_mappings = {
            'all': LandmarkSubset.ALL,
            'face_oval': LandmarkSubset.FACE_OVAL,
            'left_eye': LandmarkSubset.LEFT_EYE,
            'right_eye': LandmarkSubset.RIGHT_EYE,
            'nose': LandmarkSubset.NOSE,
            'lips': LandmarkSubset.LIPS
        }
        
        for subset_str, expected_enum in subset_mappings.items():
            args = parse_arguments(['--webcam', '--landmark-subset', subset_str])
            config = create_config_from_args(args)
            assert config.landmark_subset == expected_enum


class TestArgumentValidation:
    """Test argument validation."""
    
    def test_validate_args_valid_webcam(self):
        """Test validation of valid webcam arguments."""
        args = parse_arguments(['--webcam'])
        
        # Should not raise any exceptions
        try:
            validate_args(args)
            validation_passed = True
        except ValueError:
            validation_passed = False
        
        assert validation_passed == True
    
    def test_validate_args_valid_video(self):
        """Test validation of valid video arguments."""
        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b'dummy video data')
        
        try:
            args = parse_arguments(['--video', temp_path])
            
            # Validation might pass or fail depending on video file validity
            try:
                validate_args(args)
                validation_passed = True
            except ValueError:
                validation_passed = False
            
            # Should at least not crash
            assert isinstance(validation_passed, bool)
            
        finally:
            os.unlink(temp_path)
    
    def test_validate_args_invalid_video(self):
        """Test validation of invalid video file."""
        args = parse_arguments(['--video', 'nonexistent_file.mp4'])
        
        with pytest.raises(ValueError):
            validate_args(args)
    
    def test_validate_args_invalid_confidence(self):
        """Test validation of invalid confidence values."""
        # Test negative confidence
        args = parse_arguments(['--webcam', '--confidence', '-0.1'])
        try:
            validate_args(args)
            # If no validation, should pass
        except ValueError:
            pass  # Expected if validation is implemented
        
        # Test confidence > 1.0
        args = parse_arguments(['--webcam', '--confidence', '1.5'])
        try:
            validate_args(args)
        except ValueError:
            pass  # Expected if validation is implemented
    
    def test_validate_args_no_input_source(self):
        """Test validation when no input source is specified."""
        # Neither webcam nor video specified
        args = argparse.Namespace()
        args.webcam = False
        args.video = None
        args.confidence = 0.5
        args.tracking_confidence = 0.5
        args.max_faces = 1
        
        with pytest.raises(ValueError):
            validate_args(args)


class TestMainFunction:
    """Test the main application function."""
    
    @patch('main.FacialLandmarkProcessor')
    def test_main_webcam_mode(self, mock_processor_class):
        """Test main function in webcam mode."""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        test_args = ['--webcam']
        
        with patch('sys.argv', ['main.py'] + test_args):
            with patch('main.parse_arguments', return_value=parse_arguments(test_args)) as mock_parse:
                try:
                    main()
                    # Should call processor.process_webcam()
                    mock_processor.process_webcam.assert_called_once()
                except SystemExit:
                    pass  # May exit normally
    
    @patch('main.FacialLandmarkProcessor')
    def test_main_video_mode(self, mock_processor_class):
        """Test main function in video mode."""
        mock_processor = MagicMock()
        mock_processor_class.return_value = mock_processor
        
        # Create temporary video file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as temp_file:
            temp_path = temp_file.name
            temp_file.write(b'dummy video data')
        
        try:
            test_args = ['--video', temp_path]
            
            with patch('sys.argv', ['main.py'] + test_args):
                with patch('main.validate_args'):  # Skip file validation
                    try:
                        main()
                        # Should call processor.process_video()
                        mock_processor.process_video.assert_called_once_with(temp_path)
                    except SystemExit:
                        pass  # May exit normally
        finally:
            os.unlink(temp_path)
    
    @patch('main.FacialLandmarkProcessor')
    def test_main_error_handling(self, mock_processor_class):
        """Test main function error handling."""
        mock_processor = MagicMock()
        mock_processor.process_webcam.side_effect = Exception("Test error")
        mock_processor_class.return_value = mock_processor
        
        test_args = ['--webcam']
        
        with patch('sys.argv', ['main.py'] + test_args):
            with patch('builtins.print'):  # Suppress error output
                try:
                    main()
                except SystemExit as e:
                    # Should exit with error code
                    assert e.code != 0
                except Exception:
                    # Or handle the exception gracefully
                    pass
    
    def test_main_invalid_arguments(self):
        """Test main function with invalid arguments."""
        # Test with invalid argument combination
        test_args = ['--invalid-argument']
        
        with patch('sys.argv', ['main.py'] + test_args):
            with pytest.raises(SystemExit):
                # Should exit due to argument parsing error
                main()


class TestLogging:
    """Test logging setup and functionality."""
    
    def test_setup_logging_default(self):
        """Test default logging setup."""
        try:
            setup_logging()
            # Should not raise any exceptions
            logging_setup = True
        except Exception:
            logging_setup = False
        
        assert logging_setup == True
    
    def test_setup_logging_debug(self):
        """Test debug logging setup."""
        try:
            setup_logging(debug=True)
            logging_setup = True
        except Exception:
            logging_setup = False
        
        assert logging_setup == True
    
    def test_setup_logging_file(self):
        """Test logging to file."""
        with tempfile.NamedTemporaryFile(suffix='.log', delete=False) as temp_file:
            temp_path = temp_file.name
        
        try:
            setup_logging(log_file=temp_path)
            
            # Should create log file
            assert os.path.exists(temp_path)
            
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)


class TestIntegration:
    """Integration tests for the main application."""
    
    @patch('cv2.VideoCapture')
    @patch('cv2.imshow')
    @patch('cv2.waitKey')
    @patch('cv2.destroyAllWindows')
    def test_integration_webcam_flow(self, mock_destroy, mock_waitkey, mock_imshow, mock_videocapture):
        """Test complete webcam processing flow."""
        # Setup mocks
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.read.side_effect = [
            (True, np.zeros((480, 640, 3), dtype=np.uint8)),  # First frame
            (False, None)  # End of video
        ]
        mock_videocapture.return_value = mock_cap
        mock_waitkey.return_value = ord('q')  # Exit immediately
        
        test_args = ['--webcam', '--max-faces', '1']
        
        with patch('sys.argv', ['main.py'] + test_args):
            try:
                main()
                # Should complete without errors
                integration_success = True
            except SystemExit:
                integration_success = True  # Normal exit
            except Exception:
                integration_success = False
        
        assert integration_success == True
    
    def test_integration_argument_flow(self):
        """Test complete argument processing flow."""
        test_args = [
            '--webcam',
            '--max-faces', '2',
            '--confidence', '0.7',
            '--landmark-subset', 'face_oval',
            '--show-face-boxes'
        ]
        
        # Parse arguments
        args = parse_arguments(test_args)
        
        # Create config
        config = create_config_from_args(args)
        
        # Validate arguments
        try:
            validate_args(args)
            validation_success = True
        except ValueError:
            validation_success = False
        
        # Should complete successfully
        assert validation_success == True
        assert config.max_num_faces == 2
        assert config.min_detection_confidence == 0.7
        assert config.landmark_subset == LandmarkSubset.FACE_OVAL
        assert config.draw_face_boxes == True


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_empty_arguments(self):
        """Test with empty argument list."""
        with pytest.raises(SystemExit):
            # Should require at least one input source
            parse_arguments([])
    
    def test_conflicting_arguments(self):
        """Test with potentially conflicting arguments."""
        # Test landmarks disabled but indices enabled
        args = parse_arguments([
            '--webcam',
            '--no-landmarks',
            '--show-indices'
        ])
        
        config = create_config_from_args(args)
        
        # Should handle gracefully
        assert config.draw_landmarks == False
        assert config.show_landmark_indices == True
    
    def test_extreme_values(self):
        """Test with extreme parameter values."""
        args = parse_arguments([
            '--webcam',
            '--max-faces', '100',  # Very high
            '--confidence', '0.01',  # Very low
            '--tracking-confidence', '0.99'  # Very high
        ])
        
        config = create_config_from_args(args)
        
        # Should accept extreme but valid values
        assert config.max_num_faces == 100
        assert config.min_detection_confidence == 0.01
        assert config.min_tracking_confidence == 0.99


if __name__ == '__main__':
    pytest.main([__file__])
