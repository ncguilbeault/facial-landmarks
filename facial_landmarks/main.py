#!/usr/bin/env python3
"""
Facial Landmarks Detection System
Fast facial detection and landmark detection with real-time inference capabilities.

This module provides a comprehensive system for facial detection and landmark detection
using MediaPipe and OpenCV, optimized for real-time applications.
"""

import cv2
import numpy as np
import argparse
import sys
import os
import logging
from typing import Optional

from facial_processor import FacialLandmarkProcessor, ProcessingConfig
from landmark_detector import LandmarkSubset
from utils import resize_image, enhance_image, create_comparison_image, calculate_face_quality


def parse_arguments(args_list: Optional[list] = None) -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Args:
        args_list: Optional list of arguments to parse (for testing)
        
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Facial Landmarks Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Webcam demo
  python main.py --webcam
  
  # Process single image
  python main.py --image path/to/image.jpg
  
  # Process video
  python main.py --video path/to/video.mp4 --output output.mp4
  
  # Benchmark performance
  python main.py --benchmark
  
  # Custom settings
  python main.py --webcam --max-faces 5 --no-landmarks
"""
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--webcam', action='store_true', 
                           help='Use webcam input')
    input_group.add_argument('--image', type=str, 
                           help='Path to input image')
    input_group.add_argument('--video', type=str, 
                           help='Path to input video')
    input_group.add_argument('--benchmark', action='store_true',
                           help='Run performance benchmark')
    
    # Output options
    parser.add_argument('--output', type=str,
                       help='Output file path (for video processing)')
    parser.add_argument('--output-dir', type=str, default='output',
                       help='Output directory for results')
    
    # Processing options
    parser.add_argument('--max-faces', type=int, default=1,
                       help='Maximum number of faces to detect')
    parser.add_argument('--face-confidence', type=float, default=0.7,
                       help='Face detection confidence threshold')
    parser.add_argument('--landmark-confidence', type=float, default=0.5,
                       help='Landmark detection confidence threshold')
    
    # For backward compatibility with tests
    parser.add_argument('--confidence', type=float, default=0.5,
                       help='Detection confidence threshold (alias for face-confidence)')
    parser.add_argument('--tracking-confidence', type=float, default=0.5,
                       help='Tracking confidence threshold')
    
    # Feature toggles
    parser.add_argument('--no-face-detection', action='store_true',
                       help='Disable face detection')
    parser.add_argument('--no-landmarks', action='store_true',
                       help='Disable landmark detection')
    parser.add_argument('--no-face-boxes', action='store_true',
                       help='Disable face bounding boxes')
    parser.add_argument('--show-face-boxes', action='store_true',
                       help='Show face bounding boxes')
    parser.add_argument('--no-keypoints', action='store_true',
                       help='Disable face keypoints')
    
    # Landmark options
    parser.add_argument('--landmark-subset', type=str, 
                       choices=['all', 'face_oval', 'left_eye', 'right_eye', 
                               'left_eyebrow', 'right_eyebrow', 'nose', 'lips'],
                       default='all',
                       help='Landmark subset to display')
    parser.add_argument('--no-connections', action='store_true',
                       help='Disable landmark connections')
    parser.add_argument('--simple-landmarks', action='store_true',
                       help='Use simple landmark drawing (less detailed)')
    parser.add_argument('--show-indices', action='store_true',
                       help='Show landmark indices (debug mode)')
    
    # Performance options
    parser.add_argument('--parallel', action='store_true',
                       help='Enable parallel processing')
    
    # Video recording options
    parser.add_argument('--save-video', action='store_true',
                       help='Save video output')
    parser.add_argument('--no-fps', action='store_true',
                       help='Disable FPS display')
    
    if args_list is not None:
        return parser.parse_args(args_list)
    else:
        return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> ProcessingConfig:
    """
    Create ProcessingConfig from parsed arguments.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        ProcessingConfig object
    """
    # Handle backward compatibility for confidence parameters
    face_confidence = getattr(args, 'face_confidence', 0.7)
    if hasattr(args, 'confidence') and args.confidence != 0.5:
        face_confidence = args.confidence
    
    landmark_confidence = getattr(args, 'landmark_confidence', 0.5)
    
    # Map landmark subset string to enum
    landmark_subset_map = {
        'all': LandmarkSubset.ALL,
        'face_oval': LandmarkSubset.FACE_OVAL,
        'left_eye': LandmarkSubset.LEFT_EYE,
        'right_eye': LandmarkSubset.RIGHT_EYE,
        'left_eyebrow': LandmarkSubset.LEFT_EYEBROW,
        'right_eyebrow': LandmarkSubset.RIGHT_EYEBROW,
        'nose': LandmarkSubset.NOSE,
        'lips': LandmarkSubset.LIPS
    }
    
    landmark_subset = landmark_subset_map.get(args.landmark_subset, LandmarkSubset.ALL)
    
    config = ProcessingConfig(
        # Face detection settings
        face_detection_confidence=face_confidence,
        face_model_selection=0,  # Short-range model for better performance
        
        # Landmark detection settings
        max_num_faces=args.max_faces,
        refine_landmarks=True,
        landmark_detection_confidence=landmark_confidence,
        landmark_tracking_confidence=getattr(args, 'tracking_confidence', 0.5),
        
        # Processing settings
        enable_face_detection=not getattr(args, 'no_face_detection', False),
        enable_landmark_detection=not args.no_landmarks,
        enable_parallel_processing=getattr(args, 'parallel', False),
        
        # Visualization settings
        draw_face_boxes=getattr(args, 'show_face_boxes', False) and not args.no_face_boxes,
        draw_face_keypoints=not getattr(args, 'no_keypoints', False),
        draw_landmarks=not args.no_landmarks,
        landmark_subset=landmark_subset,
        draw_landmark_connections=not getattr(args, 'no_connections', False),
        draw_detailed_landmarks=not getattr(args, 'simple_landmarks', False),
        show_landmark_indices=getattr(args, 'show_indices', False),
        
        # Output settings
        output_path=getattr(args, 'output', None),
        save_video=getattr(args, 'save_video', False),
        show_fps=not getattr(args, 'no_fps', False)
    )
    
    return config


def validate_args(args: argparse.Namespace) -> None:
    """
    Validate command line arguments.
    
    Args:
        args: Parsed arguments to validate
        
    Raises:
        ValueError: If arguments are invalid
    """
    # Check that at least one input source is specified
    input_sources = [args.webcam, bool(args.video), bool(getattr(args, 'image', None)), 
                    getattr(args, 'benchmark', False)]
    if not any(input_sources):
        raise ValueError("At least one input source must be specified (--webcam, --video, --image, or --benchmark)")
    
    # Validate video file exists if specified
    if hasattr(args, 'video') and args.video:
        if not os.path.exists(args.video):
            raise ValueError(f"Video file does not exist: {args.video}")
    
    # Validate image file exists if specified  
    if hasattr(args, 'image') and args.image:
        if not os.path.exists(args.image):
            raise ValueError(f"Image file does not exist: {args.image}")
    
    # Validate confidence values
    confidence_attrs = ['confidence', 'face_confidence', 'landmark_confidence', 'tracking_confidence']
    for attr in confidence_attrs:
        if hasattr(args, attr):
            value = getattr(args, attr)
            if value < 0.0 or value > 1.0:
                raise ValueError(f"{attr} must be between 0.0 and 1.0, got {value}")
    
    # Validate max_faces
    if args.max_faces < 1:
        raise ValueError(f"max_faces must be at least 1, got {args.max_faces}")


def setup_logging(debug: bool = False, log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        debug: Enable debug logging
        log_file: Optional log file path
    """
    level = logging.DEBUG if debug else logging.INFO
    
    # Configure basic logging
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add file handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logging.getLogger().addHandler(file_handler)


def demo_webcam(config: ProcessingConfig):
    """Demo with webcam feed."""
    print("Starting webcam demo...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    print("  '1' - Toggle face detection boxes")
    print("  '2' - Toggle landmarks")
    print("  '3' - Cycle landmark subsets")
    
    processor = FacialLandmarkProcessor(config)
    
    try:
        processor.process_video_stream(source=0, display=True)
    except Exception as e:
        print(f"Error during webcam processing: {e}")
        return False
    
    return True


def demo_image(image_path: str, config: ProcessingConfig, output_dir: str = "output"):
    """Demo with single image."""
    print(f"Processing image: {image_path}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image from {image_path}")
        return False
    
    # Resize if too large
    image = resize_image(image, max_width=1280, max_height=720)
    
    # Process image
    processor = FacialLandmarkProcessor(config)
    result = processor.process_image(image)
    
    # Get analysis
    analysis = processor.get_facial_analysis(image)
    
    # Print results
    print(f"\nResults:")
    print(f"  Faces detected: {len(result.faces)}")
    print(f"  Processing time: {result.processing_time*1000:.1f}ms")
    
    for i, face in enumerate(result.faces):
        print(f"  Face {i+1}: confidence={face.confidence:.3f}, bbox={face.bbox}")
    
    if analysis['landmarks']:
        for i, landmark_info in enumerate(analysis['landmarks']):
            features = landmark_info['features']
            print(f"  Face {i+1} features:")
            print(f"    Left eye AR: {features['left_eye_aspect_ratio']:.3f}")
            print(f"    Right eye AR: {features['right_eye_aspect_ratio']:.3f}")
            print(f"    Mouth AR: {features['mouth_aspect_ratio']:.3f}")
            pose = features['pose']
            print(f"    Pose - Pitch: {pose['pitch']:.1f}°, Yaw: {pose['yaw']:.1f}°, Roll: {pose['roll']:.1f}°")
    
    # Save results
    saved_files = processor.save_results(image, result, output_dir)
    print(f"\nSaved files:")
    for file_type, file_path in saved_files.items():
        print(f"  {file_type}: {file_path}")
    
    # Show image (if display available)
    try:
        annotated_image = processor.draw_results(image, result)
        cv2.imshow('Facial Landmarks Result', annotated_image)
        print("\nPress any key to close the image window...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Display not available, results saved to files.")
    
    return True


def demo_video(video_path: str, config: ProcessingConfig, output_path: Optional[str] = None):
    """Demo with video file."""
    print(f"Processing video: {video_path}")
    
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return False
    
    processor = FacialLandmarkProcessor(config)
    
    try:
        processor.process_video_stream(
            source=video_path, 
            output_path=output_path,
            display=True
        )
    except Exception as e:
        print(f"Error during video processing: {e}")
        return False
    
    if output_path:
        print(f"Output video saved to: {output_path}")
    
    return True


def benchmark_performance(config: ProcessingConfig):
    """Benchmark system performance."""
    print("Running performance benchmark...")
    
    # Create test image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    processor = FacialLandmarkProcessor(config)
    
    # Warm up
    for _ in range(5):
        processor.process_image(test_image)
    
    # Benchmark
    import time
    times = []
    
    for i in range(100):
        start_time = time.time()
        result = processor.process_image(test_image)
        end_time = time.time()
        times.append(end_time - start_time)
        
        if i % 20 == 0:
            print(f"  Progress: {i+1}/100")
    
    # Calculate statistics
    avg_time = np.mean(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    fps = 1.0 / np.mean(times)
    
    print(f"\nBenchmark Results:")
    print(f"  Average processing time: {avg_time:.2f}ms")
    print(f"  Min processing time: {min_time:.2f}ms")
    print(f"  Max processing time: {max_time:.2f}ms")
    print(f"  Estimated FPS: {fps:.1f}")
    
    return True


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        validate_args(args)
    except ValueError as e:
        print(f"Argument validation error: {e}")
        return 1
    
    setup_logging(debug=False)
    
    config = create_config_from_args(args)
    
    # Print configuration
    print("Facial Landmarks Detection System")
    print("=" * 40)
    print(f"Face detection: {'Enabled' if config.enable_face_detection else 'Disabled'}")
    print(f"Landmark detection: {'Enabled' if config.enable_landmark_detection else 'Disabled'}")
    print(f"Max faces: {config.max_num_faces}")
    print(f"Parallel processing: {'Enabled' if config.enable_parallel_processing else 'Disabled'}")
    print(f"Landmark subset: {args.landmark_subset}")
    print("=" * 40)
    
    # Execute based on input type
    success = False
    
    try:
        if args.webcam:
            success = demo_webcam(config)
        elif args.image:
            success = demo_image(args.image, config, args.output_dir)
        elif args.video:
            success = demo_video(args.video, config, args.output)
        elif args.benchmark:
            success = benchmark_performance(config)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        success = True
    except Exception as e:
        print(f"Error: {e}")
        success = False
    
    if success:
        print("\nCompleted successfully!")
        return 0
    else:
        print("\nFailed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
