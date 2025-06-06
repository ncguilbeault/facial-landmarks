"""
Test configuration and shared utilities for facial landmarks tests.
"""

import os
import cv2
import numpy as np
import pytest
from pathlib import Path
from typing import Tuple


# Test configuration
TEST_IMAGE_SIZE = (480, 640, 3)  # Height, Width, Channels
TEST_FACE_CONFIDENCE = 0.5
TEST_LANDMARK_CONFIDENCE = 0.5


def create_test_image(size: Tuple[int, int, 3] = TEST_IMAGE_SIZE, width: int = None, height: int = None) -> np.ndarray:
    """Create a simple test image for testing."""
    if width is not None and height is not None:
        size = (height, width, 3)
    h, w, c = size
    img = np.ones((h, w, c), dtype=np.uint8) * 128  # Gray background
    
    # Add some geometric shapes
    cv2.rectangle(img, (50, 50), (200, 200), (100, 150, 200), -1)
    cv2.circle(img, (w-150, 150), 80, (200, 100, 150), -1)
    cv2.ellipse(img, (w//2, h-100), (100, 60), 0, 0, 360, (150, 200, 100), -1)
    
    # Add text
    cv2.putText(img, "Test Image", (w//2-80, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    return img


def create_face_like_image(size: Tuple[int, int, 3] = TEST_IMAGE_SIZE) -> np.ndarray:
    """Create a simple face-like image for testing face detection."""
    h, w, c = size
    img = np.ones((h, w, c), dtype=np.uint8) * 200  # Light background
    
    # Face oval
    center = (w//2, h//2)
    face_width = w//3
    face_height = h//2
    cv2.ellipse(img, center, (face_width, face_height), 0, 0, 360, (180, 150, 120), -1)
    
    # Eyes
    eye_y = center[1] - face_height//4
    left_eye = (center[0] - face_width//3, eye_y)
    right_eye = (center[0] + face_width//3, eye_y)
    cv2.ellipse(img, left_eye, (20, 10), 0, 0, 360, (50, 50, 50), -1)
    cv2.ellipse(img, right_eye, (20, 10), 0, 0, 360, (50, 50, 50), -1)
    
    # Nose
    nose_points = np.array([
        [center[0], center[1] - 10],
        [center[0] - 10, center[1] + 20],
        [center[0] + 10, center[1] + 20]
    ], np.int32)
    cv2.fillPoly(img, [nose_points], (160, 130, 100))
    
    # Mouth
    mouth_center = (center[0], center[1] + face_height//3)
    cv2.ellipse(img, mouth_center, (30, 15), 0, 0, 180, (120, 80, 80), -1)
    
    return img


def create_noisy_image(size: Tuple[int, int, 3] = TEST_IMAGE_SIZE) -> np.ndarray:
    """Create a noisy image for testing robustness."""
    h, w, c = size
    return np.random.randint(0, 255, (h, w, c), dtype=np.uint8)


def create_black_image(size: Tuple[int, int, 3] = TEST_IMAGE_SIZE) -> np.ndarray:
    """Create a black image for testing edge cases."""
    h, w, c = size
    return np.zeros((h, w, c), dtype=np.uint8)


def create_white_image(size: Tuple[int, int, 3] = TEST_IMAGE_SIZE) -> np.ndarray:
    """Create a white image for testing edge cases."""
    h, w, c = size
    return np.ones((h, w, c), dtype=np.uint8) * 255


def create_test_video(output_path: str, duration: float = 2.0, fps: int = 30, 
                     width: int = 640, height: int = 480) -> bool:
    """
    Create a test video file for testing.
    
    Args:
        output_path: Path where to save the video
        duration: Duration in seconds
        fps: Frames per second
        width: Video width
        height: Video height
        
    Returns:
        True if video was created successfully, False otherwise
    """
    try:
        # Define codec and create VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            return False
        
        # Calculate total frames
        total_frames = int(duration * fps)
        
        for frame_num in range(total_frames):
            # Create a test frame with moving circle
            frame = create_test_image((height, width, 3))
            
            # Add moving circle to make it more realistic
            center_x = int(width // 2 + 100 * np.sin(frame_num * 0.1))
            center_y = int(height // 2 + 50 * np.cos(frame_num * 0.1))
            cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), -1)
            
            # Add frame number text
            cv2.putText(frame, f"Frame {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            writer.write(frame)
        
        writer.release()
        return True
        
    except Exception as e:
        print(f"Error creating test video: {e}")
        return False


@pytest.fixture
def test_image():
    """Provide a standard test image."""
    return create_test_image()


@pytest.fixture
def face_like_image():
    """Provide a face-like test image."""
    return create_face_like_image()


@pytest.fixture
def noisy_image():
    """Provide a noisy test image."""
    return create_noisy_image()


@pytest.fixture
def black_image():
    """Provide a black test image."""
    return create_black_image()


@pytest.fixture
def white_image():
    """Provide a white test image."""
    return create_white_image()


@pytest.fixture
def temp_output_dir(tmp_path):
    """Provide a temporary output directory."""
    output_dir = tmp_path / "test_output"
    output_dir.mkdir()
    return str(output_dir)


class MockVideoCapture:
    """Mock cv2.VideoCapture for testing."""
    
    def __init__(self, source=0, frames=None):
        self.source = source
        self.frames = frames or [create_face_like_image() for _ in range(10)]
        self.frame_index = 0
        self.is_opened = True
        
    def isOpened(self):
        return self.is_opened
    
    def read(self):
        if self.frame_index < len(self.frames):
            frame = self.frames[self.frame_index]
            self.frame_index += 1
            return True, frame
        return False, None
    
    def set(self, prop, value):
        pass
    
    def get(self, prop):
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return 640
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480
        elif prop == cv2.CAP_PROP_FPS:
            return 30
        return 0
    
    def release(self):
        self.is_opened = False


def skip_if_no_display():
    """Skip test if no display is available."""
    return pytest.mark.skipif(
        os.environ.get('DISPLAY') is None,
        reason="No display available"
    )


def skip_if_no_camera():
    """Skip test if no camera is available."""
    try:
        cap = cv2.VideoCapture(0)
        has_camera = cap.isOpened()
        cap.release()
        return pytest.mark.skipif(
            not has_camera,
            reason="No camera available"
        )
    except:
        return pytest.mark.skipif(True, reason="Camera test failed")
