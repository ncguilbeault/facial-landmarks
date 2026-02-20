#!/usr/bin/env python3
"""
Simple example demonstrating facial landmark detection.
This script shows basic usage of the facial landmarks system.
"""

import cv2
import numpy as np
from facial_landmarks.facial_processor import FacialLandmarkProcessor, ProcessingConfig
from facial_landmarks.landmark_detector import LandmarkSubset


def create_test_image():
    """Create a simple test image with basic shapes (for testing without a real image)."""
    # Create a blank image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128  # Gray background
    
    # Add some geometric shapes to make it interesting
    cv2.rectangle(img, (50, 50), (200, 200), (100, 150, 200), -1)
    cv2.circle(img, (450, 150), 80, (200, 100, 150), -1)
    cv2.ellipse(img, (320, 350), (100, 60), 0, 0, 360, (150, 200, 100), -1)
    
    # Add some text
    cv2.putText(img, "Test Image", (250, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, "No faces detected here", (180, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
    
    return img


def example_basic_usage():
    """Demonstrate basic usage of the facial landmark system."""
    print("Example 1: Basic Usage")
    print("-" * 30)
    
    # Create processor with default settings
    processor = FacialLandmarkProcessor()
    
    # Create a test image (since we might not have a real image)
    test_image = create_test_image()
    
    # Process the image
    result = processor.process_image(test_image)
    
    # Print results
    print(f"Faces detected: {len(result.faces)}")
    print(f"Landmarks detected: {len(result.landmarks)}")
    print(f"Processing time: {result.processing_time*1000:.2f}ms")
    print(f"Estimated FPS: {result.fps:.1f}")
    
    # Draw results (even if no faces are detected, it will return the original image)
    annotated_image = processor.draw_results(test_image, result)
    
    return annotated_image


def example_custom_config():
    """Demonstrate custom configuration."""
    print("\\nExample 2: Custom Configuration")
    print("-" * 35)
    
    # Create custom configuration
    config = ProcessingConfig(
        # Face detection settings
        face_detection_confidence=0.8,  # Higher confidence threshold
        max_num_faces=3,  # Allow up to 3 faces
        
        # Landmark settings
        refine_landmarks=True,
        landmark_detection_confidence=0.6,
        
        # Visualization settings
        draw_face_boxes=True,
        draw_face_keypoints=True,
        draw_landmarks=True,
        landmark_subset=LandmarkSubset.FACE_OVAL,  # Only draw face outline
        draw_landmark_connections=False  # Don't draw connections
    )
    
    processor = FacialLandmarkProcessor(config)
    
    # Process test image
    test_image = create_test_image()
    result = processor.process_image(test_image)
    
    print(f"Configuration - Max faces: {config.max_num_faces}")
    print(f"Configuration - Face confidence: {config.face_detection_confidence}")
    print(f"Configuration - Landmark subset: {config.landmark_subset}")
    print(f"Processing result - Faces: {len(result.faces)}, Landmarks: {len(result.landmarks)}")
    
    return processor.draw_results(test_image, result)


def example_enhanced_landmarks():
    """Demonstrate enhanced landmark visualization options."""
    print("\\nExample 2: Enhanced Landmark Visualization")
    print("-" * 40)
    
    # Try to use webcam for real face detection
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Webcam not available, using test image for enhanced landmarks demo")
        # Fall back to test image
        processor = FacialLandmarkProcessor()
        test_image = create_test_image()
        result = processor.process_image(test_image)
        return processor.draw_results(test_image, result)
    
    print("Webcam opened successfully!")
    print("Show your face to the camera to see enhanced landmarks...")
    
    # Configuration for detailed landmarks
    config = ProcessingConfig(
        draw_landmarks=True,
        landmark_subset=LandmarkSubset.ALL,
        draw_landmark_connections=True,
        draw_detailed_landmarks=True,
        draw_face_boxes=True,
        show_landmark_indices=False
    )
    
    processor = FacialLandmarkProcessor(config)
    
    # Capture a few frames to get a good face detection
    frames_captured = 0
    best_result = None
    best_frame = None
    
    while frames_captured < 30:  # Try for 1 second at 30fps
        ret, frame = cap.read()
        if not ret:
            break
            
        result = processor.process_image(frame)
        
        if len(result.faces) > 0:
            # Found a face, keep the best one (highest confidence)
            if best_result is None or (len(result.faces) > 0 and 
                                     result.faces[0].confidence > best_result.faces[0].confidence):
                best_result = result
                best_frame = frame.copy()
        
        frames_captured += 1
    
    cap.release()
    
    if best_result and len(best_result.faces) > 0:
        print(f"✓ Detected face with confidence: {best_result.faces[0].confidence:.3f}")
        print(f"✓ Total landmarks: {len(best_result.landmarks[0].landmarks) if best_result.landmarks else 0}")
        
        # Create different visualizations
        configs = [
            ("All Landmarks (Detailed)", ProcessingConfig(
                draw_landmarks=True, landmark_subset=LandmarkSubset.ALL,
                draw_landmark_connections=True, draw_detailed_landmarks=True
            )),
            ("Face Oval Only", ProcessingConfig(
                draw_landmarks=True, landmark_subset=LandmarkSubset.FACE_OVAL,
                draw_landmark_connections=True
            )),
            ("Eyes Only", ProcessingConfig(
                draw_landmarks=True, landmark_subset=LandmarkSubset.LEFT_EYE,
                draw_landmark_connections=True
            )),
            ("Debug Mode (with indices)", ProcessingConfig(
                draw_landmarks=True, show_landmark_indices=True
            ))
        ]
        
        print("\\nGenerating different visualizations...")
        for name, cfg in configs:
            temp_processor = FacialLandmarkProcessor(cfg)
            annotated = temp_processor.draw_results(best_frame, best_result)
            
            # Save the result
            filename = f"enhanced_{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
            cv2.imwrite(filename, annotated)
            print(f"  - {name}: saved as {filename}")
        
        return best_frame
    else:
        print("✗ No faces detected in webcam feed")
        print("  Make sure your face is visible and well-lit")
        # Return test image as fallback
        processor = FacialLandmarkProcessor()
        test_image = create_test_image()
        result = processor.process_image(test_image)
        return processor.draw_results(test_image, result)


def example_performance_test():
    """Demonstrate performance testing."""
    print("\\nExample 3: Performance Test")
    print("-" * 30)
    
    processor = FacialLandmarkProcessor()
    test_image = create_test_image()
    
    # Warm up
    for _ in range(5):
        processor.process_image(test_image)
    
    # Performance test
    import time
    times = []
    num_iterations = 50
    
    print(f"Running {num_iterations} iterations...")
    
    for i in range(num_iterations):
        start_time = time.time()
        result = processor.process_image(test_image)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Calculate statistics
    avg_time = np.mean(times) * 1000
    min_time = np.min(times) * 1000
    max_time = np.max(times) * 1000
    std_time = np.std(times) * 1000
    fps = 1.0 / np.mean(times)
    
    print(f"Performance Results:")
    print(f"  Average time: {avg_time:.2f}ms (±{std_time:.2f}ms)")
    print(f"  Min time: {min_time:.2f}ms")
    print(f"  Max time: {max_time:.2f}ms")
    print(f"  Estimated FPS: {fps:.1f}")


def example_webcam_processing():
    """Demonstrate webcam processing (if camera is available)."""
    print("\\nExample 4: Webcam Processing")
    print("-" * 32)
    
    # Test if camera is available
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No camera available for webcam demo")
        cap.release()
        return
    
    cap.release()
    
    print("Camera detected! Starting webcam demo...")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Save screenshot")
    
    try:
        # Create processor
        config = ProcessingConfig(
            face_detection_confidence=0.7,
            max_num_faces=2,
            draw_landmarks=True,
            landmark_subset=LandmarkSubset.FACE_OVAL
        )
        
        processor = FacialLandmarkProcessor(config)
        
        # Process webcam (this will open a window)
        processor.process_video_stream(source=0, display=True)
        
    except Exception as e:
        print(f"Webcam processing error: {e}")


def example_image_analysis():
    """Demonstrate detailed image analysis."""
    print("\\nExample 5: Detailed Analysis")
    print("-" * 31)
    
    processor = FacialLandmarkProcessor()
    test_image = create_test_image()
    
    # Get detailed analysis
    analysis = processor.get_facial_analysis(test_image)
    
    print("Detailed Analysis Results:")
    print(f"  Total faces detected: {analysis['num_faces']}")
    print(f"  Processing time: {analysis['processing_time']*1000:.2f}ms")
    
    # Face details
    for i, face_data in enumerate(analysis['faces']):
        print(f"  Face {i+1}:")
        print(f"    Confidence: {face_data['confidence']:.3f}")
        print(f"    Bounding box: {face_data['bbox']}")
        if face_data['keypoints']:
            print(f"    Keypoints: {len(face_data['keypoints'])} detected")
    
    # Landmark details
    for i, landmark_data in enumerate(analysis['landmarks']):
        print(f"  Face {i+1} Landmarks:")
        print(f"    Total landmarks: {landmark_data['num_landmarks']}")
        
        features = landmark_data['features']
        print(f"    Left eye aspect ratio: {features['left_eye_aspect_ratio']:.3f}")
        print(f"    Right eye aspect ratio: {features['right_eye_aspect_ratio']:.3f}")
        print(f"    Mouth aspect ratio: {features['mouth_aspect_ratio']:.3f}")
        
        pose = features['pose']
        print(f"    Head pose:")
        print(f"      Pitch: {pose['pitch']:.1f}°")
        print(f"      Yaw: {pose['yaw']:.1f}°")
        print(f"      Roll: {pose['roll']:.1f}°")


def main():
    """Run all examples."""
    print("Facial Landmarks Detection - Examples")
    print("=" * 45)
    
    try:
        # Example 1: Basic usage
        result_image1 = example_basic_usage()
        
        # Example 2: Enhanced landmarks visualization
        result_image2 = example_enhanced_landmarks()
        
        # Example 3: Custom configuration
        result_image3 = example_custom_config()
        
        # Example 3: Performance test
        example_performance_test()
        
        # Example 4: Detailed analysis
        example_image_analysis()
        
        # Show results if display is available
        try:
            print("\\nDisplaying results...")
            print("Press any key to continue between images, 'q' to quit")
            
            # Show first result
            cv2.imshow('Example 1: Basic Usage', result_image1)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            
            # Show second result
            cv2.imshow('Example 2: Custom Config', result_image2)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                cv2.destroyAllWindows()
                return
            
            # Show third result
            cv2.imshow('Example 2: Enhanced Landmarks', result_image3)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            
            if key == ord('q'):
                return
                
        except Exception as e:
            print(f"Display not available: {e}")
            print("Results processed successfully (no display)")
        
        # Example 5: Webcam (optional, only if user wants it)
        response = input("\\nWould you like to try the webcam demo? (y/n): ").lower().strip()
        if response == 'y' or response == 'yes':
            example_webcam_processing()
        
        print("\\nAll examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
