#!/usr/bin/env python3
"""
Enhanced Facial Landmarks Demo
Demonstrates the comprehensive facial landmark detection capabilities.
"""

import cv2
import numpy as np
import os
import sys
from pathlib import Path

# Add the facial_landmarks module to path
sys.path.append(str(Path(__file__).parent / "facial_landmarks"))

from facial_processor import FacialLandmarkProcessor, ProcessingConfig
from landmark_detector import LandmarkSubset


def create_demo_grid(processor, image):
    """Create a grid showing different landmark visualizations."""
    # Resize image for consistent display
    height, width = image.shape[:2]
    if width > 400:
        scale = 400 / width
        new_width = int(width * scale)
        new_height = int(height * scale)
        image = cv2.resize(image, (new_width, new_height))
    
    # Create different visualization configs
    configs = [
        # All landmarks with connections (detailed)
        ProcessingConfig(
            draw_landmarks=True,
            landmark_subset=LandmarkSubset.ALL,
            draw_landmark_connections=True,
            draw_detailed_landmarks=True,
            draw_face_boxes=False
        ),
        # All landmarks as points only
        ProcessingConfig(
            draw_landmarks=True,
            landmark_subset=LandmarkSubset.ALL,
            draw_landmark_connections=False,
            draw_detailed_landmarks=False,
            draw_face_boxes=False
        ),
        # Face oval only
        ProcessingConfig(
            draw_landmarks=True,
            landmark_subset=LandmarkSubset.FACE_OVAL,
            draw_landmark_connections=True,
            draw_face_boxes=False
        ),
        # Eyes and eyebrows
        ProcessingConfig(
            draw_landmarks=True,
            landmark_subset=LandmarkSubset.LEFT_EYE,
            draw_landmark_connections=True,
            draw_face_boxes=False
        ),
        # Nose landmarks
        ProcessingConfig(
            draw_landmarks=True,
            landmark_subset=LandmarkSubset.NOSE,
            draw_landmark_connections=True,
            draw_face_boxes=False
        ),
        # Lips landmarks
        ProcessingConfig(
            draw_landmarks=True,
            landmark_subset=LandmarkSubset.LIPS,
            draw_landmark_connections=True,
            draw_face_boxes=False
        )
    ]
    
    titles = [
        "All Landmarks (Detailed)",
        "All Points Only",
        "Face Oval",
        "Left Eye",
        "Nose",
        "Lips"
    ]
    
    # Process image with each config
    results = []
    for config in configs:
        temp_processor = FacialLandmarkProcessor(config)
        result = temp_processor.process_image(image)
        annotated = temp_processor.draw_results(image, result)
        results.append(annotated)
    
    # Create grid layout (2x3)
    rows = 2
    cols = 3
    
    # Add titles to images
    titled_images = []
    for img, title in zip(results, titles):
        # Add title
        titled_img = img.copy()
        cv2.putText(titled_img, title, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(titled_img, title, (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        titled_images.append(titled_img)
    
    # Combine into grid
    top_row = np.hstack(titled_images[:3])
    bottom_row = np.hstack(titled_images[3:])
    grid = np.vstack([top_row, bottom_row])
    
    return grid


def demo_webcam_enhanced():
    """Enhanced webcam demo with multiple visualization modes."""
    print("Enhanced Facial Landmarks Webcam Demo")
    print("=====================================")
    print("Controls:")
    print("  'q' - Quit")
    print("  '1' - All landmarks (detailed)")
    print("  '2' - All landmarks (points only)")
    print("  '3' - Face oval")
    print("  '4' - Eyes")
    print("  '5' - Nose")
    print("  '6' - Lips")
    print("  '7' - Debug mode (with indices)")
    print("  's' - Save screenshot")
    print("  ' ' - Toggle connections")
    
    # Initial config
    config = ProcessingConfig(
        draw_landmarks=True,
        landmark_subset=LandmarkSubset.ALL,
        draw_landmark_connections=True,
        draw_detailed_landmarks=True,
        draw_face_boxes=True
    )
    
    processor = FacialLandmarkProcessor(config)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    # Set camera properties for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    current_mode = "detailed"
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            result = processor.process_image(frame)
            annotated_frame = processor.draw_results(frame, result)
            
            # Add mode indicator
            mode_text = f"Mode: {current_mode} | FPS: {result.fps:.1f}"
            cv2.putText(annotated_frame, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, mode_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
            
            cv2.imshow('Enhanced Facial Landmarks', annotated_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('1'):
                # All landmarks detailed
                config.landmark_subset = LandmarkSubset.ALL
                config.draw_landmark_connections = True
                config.draw_detailed_landmarks = True
                config.show_landmark_indices = False
                processor = FacialLandmarkProcessor(config)
                current_mode = "All Detailed"
            elif key == ord('2'):
                # All landmarks points only
                config.landmark_subset = LandmarkSubset.ALL
                config.draw_landmark_connections = False
                config.draw_detailed_landmarks = False
                config.show_landmark_indices = False
                processor = FacialLandmarkProcessor(config)
                current_mode = "All Points"
            elif key == ord('3'):
                # Face oval
                config.landmark_subset = LandmarkSubset.FACE_OVAL
                config.draw_landmark_connections = True
                config.show_landmark_indices = False
                processor = FacialLandmarkProcessor(config)
                current_mode = "Face Oval"
            elif key == ord('4'):
                # Eyes
                config.landmark_subset = LandmarkSubset.LEFT_EYE
                config.draw_landmark_connections = True
                config.show_landmark_indices = False
                processor = FacialLandmarkProcessor(config)
                current_mode = "Eyes"
            elif key == ord('5'):
                # Nose
                config.landmark_subset = LandmarkSubset.NOSE
                config.draw_landmark_connections = True
                config.show_landmark_indices = False
                processor = FacialLandmarkProcessor(config)
                current_mode = "Nose"
            elif key == ord('6'):
                # Lips
                config.landmark_subset = LandmarkSubset.LIPS
                config.draw_landmark_connections = True
                config.show_landmark_indices = False
                processor = FacialLandmarkProcessor(config)
                current_mode = "Lips"
            elif key == ord('7'):
                # Debug mode with indices
                config.landmark_subset = LandmarkSubset.ALL
                config.show_landmark_indices = True
                processor = FacialLandmarkProcessor(config)
                current_mode = "Debug (indices)"
            elif key == ord(' '):
                # Toggle connections
                config.draw_landmark_connections = not config.draw_landmark_connections
                processor = FacialLandmarkProcessor(config)
            elif key == ord('s'):
                # Save screenshot
                timestamp = cv2.getTickCount()
                filename = f"enhanced_landmarks_{timestamp}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Screenshot saved: {filename}")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()


def demo_image_grid(image_path):
    """Create and display a grid showing different landmark visualizations."""
    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        return
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image: {image_path}")
        return
    
    print(f"Creating enhanced landmarks grid for: {image_path}")
    
    processor = FacialLandmarkProcessor()
    grid = create_demo_grid(processor, image)
    
    # Save and display grid
    output_path = "enhanced_landmarks_grid.jpg"
    cv2.imwrite(output_path, grid)
    print(f"Grid saved to: {output_path}")
    
    # Display if possible
    try:
        cv2.imshow('Enhanced Facial Landmarks Grid', grid)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except:
        print("Display not available, grid saved to file.")


def main():
    """Main demo function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Facial Landmarks Demo")
    parser.add_argument('--webcam', action='store_true', help='Run webcam demo')
    parser.add_argument('--image', type=str, help='Create grid from image file')
    
    args = parser.parse_args()
    
    if args.webcam:
        demo_webcam_enhanced()
    elif args.image:
        demo_image_grid(args.image)
    else:
        print("Usage:")
        print("  python demo_enhanced_landmarks.py --webcam")
        print("  python demo_enhanced_landmarks.py --image path/to/image.jpg")
        print("\nRunning webcam demo by default...")
        demo_webcam_enhanced()


if __name__ == "__main__":
    main()
