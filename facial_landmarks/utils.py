"""
Utility functions for image processing, analysis, and visualization.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any, Union
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import os


def resize_image(image: np.ndarray, 
                max_width: int = 1280, 
                max_height: int = 720,
                maintain_aspect: bool = True) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        max_width: Maximum width
        max_height: Maximum height
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if maintain_aspect:
        # Calculate scale factor
        scale = min(max_width / w, max_height / h)
        if scale < 1.0:
            new_width = int(w * scale)
            new_height = int(h * scale)
            return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        if w > max_width or h > max_height:
            return cv2.resize(image, (max_width, max_height), interpolation=cv2.INTER_AREA)
    
    return image


def enhance_image(image: np.ndarray, 
                 brightness: float = 0.0,
                 contrast: float = 1.0,
                 gamma: float = 1.0) -> np.ndarray:
    """
    Enhance image with brightness, contrast, and gamma correction.
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast multiplier (0.5 to 3.0)
        gamma: Gamma correction (0.5 to 2.0)
        
    Returns:
        Enhanced image
    """
    # Apply brightness
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        image = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    
    # Apply contrast
    if contrast != 1.0:
        image = cv2.convertScaleAbs(image, alpha=contrast, beta=0)
    
    # Apply gamma correction
    if gamma != 1.0:
        inv_gamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
        image = cv2.LUT(image, table)
    
    return image


def create_comparison_image(images: List[np.ndarray], 
                          titles: Optional[List[str]] = None,
                          max_width: int = 1920) -> np.ndarray:
    """
    Create a side-by-side comparison of multiple images.
    
    Args:
        images: List of images to compare
        titles: Optional titles for each image
        max_width: Maximum width of the comparison image
        
    Returns:
        Comparison image
    """
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Resize all images to the same height
    min_height = min(img.shape[0] for img in images)
    resized_images = []
    
    for img in images:
        if img.shape[0] != min_height:
            scale = min_height / img.shape[0]
            new_width = int(img.shape[1] * scale)
            resized_img = cv2.resize(img, (new_width, min_height))
            resized_images.append(resized_img)
        else:
            resized_images.append(img)
    
    # Calculate total width and scale if necessary
    total_width = sum(img.shape[1] for img in resized_images)
    if total_width > max_width:
        scale = max_width / total_width
        resized_images = [cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale))) 
                         for img in resized_images]
        min_height = resized_images[0].shape[0]
    
    # Add titles if provided
    if titles:
        title_height = 40
        for i, (img, title) in enumerate(zip(resized_images, titles)):
            # Create title image
            title_img = np.ones((title_height, img.shape[1], 3), dtype=np.uint8) * 255
            cv2.putText(title_img, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            
            # Concatenate title with image
            resized_images[i] = np.vstack([title_img, img])
    
    # Concatenate images horizontally
    comparison = np.hstack(resized_images)
    return comparison


def extract_face_chips(image: np.ndarray, 
                      landmarks: List[Tuple[int, int]],
                      chip_size: Tuple[int, int] = (150, 150),
                      padding: float = 0.3) -> List[np.ndarray]:
    """
    Extract aligned face chips from landmarks.
    
    Args:
        image: Input image
        landmarks: List of facial landmarks as (x, y) coordinates
        chip_size: Size of output face chips
        padding: Padding around face
        
    Returns:
        List of aligned face chips
    """
    if len(landmarks) < 5:  # Need at least 5 points for alignment
        return []
    
    # Assume first 5 landmarks are: left eye, right eye, nose, left mouth, right mouth
    left_eye = np.array(landmarks[0])
    right_eye = np.array(landmarks[1])
    nose = np.array(landmarks[2]) if len(landmarks) > 2 else (left_eye + right_eye) / 2
    
    # Calculate face center and angle
    eye_center = (left_eye + right_eye) / 2
    face_center = (eye_center + nose) / 2
    
    # Calculate rotation angle
    eye_vector = right_eye - left_eye
    angle = np.arctan2(eye_vector[1], eye_vector[0]) * 180 / np.pi
    
    # Calculate face size
    eye_distance = np.linalg.norm(eye_vector)
    face_size = eye_distance * (2 + padding)
    
    # Create transformation matrix
    scale = min(chip_size) / face_size
    M = cv2.getRotationMatrix2D(tuple(face_center), angle, scale)
    
    # Adjust translation to center the face in the chip
    M[0, 2] += chip_size[0] / 2 - face_center[0] * scale
    M[1, 2] += chip_size[1] / 2 - face_center[1] * scale
    
    # Apply transformation
    face_chip = cv2.warpAffine(image, M, chip_size, flags=cv2.INTER_CUBIC)
    
    return [face_chip]


def calculate_face_quality(image: np.ndarray, 
                          landmarks: Optional[List[Tuple[int, int]]] = None) -> Dict[str, float]:
    """
    Calculate face quality metrics.
    
    Args:
        image: Face image
        landmarks: Optional facial landmarks
        
    Returns:
        Dictionary with quality metrics
    """
    quality_metrics = {}
    
    # Blur detection (Laplacian variance)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality_metrics['sharpness'] = float(blur_score)
    
    # Brightness
    brightness = np.mean(gray)
    quality_metrics['brightness'] = float(brightness)
    
    # Contrast (standard deviation)
    contrast = np.std(gray)
    quality_metrics['contrast'] = float(contrast)
    
    # Exposure quality (histogram analysis)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_norm = hist.flatten() / hist.sum()
    
    # Check for over/under exposure
    overexposed = np.sum(hist_norm[240:]) * 100  # Top 6% of values
    underexposed = np.sum(hist_norm[:16]) * 100  # Bottom 6% of values
    
    quality_metrics['overexposed_percent'] = float(overexposed)
    quality_metrics['underexposed_percent'] = float(underexposed)
    
    # Overall quality score (0-100)
    blur_norm = min(blur_score / 1000, 1.0)  # Normalize blur score
    brightness_norm = 1.0 - abs(brightness - 128) / 128  # Optimal brightness around 128
    contrast_norm = min(contrast / 64, 1.0)  # Normalize contrast
    exposure_penalty = (overexposed + underexposed) / 200  # Penalty for poor exposure
    
    overall_quality = (blur_norm + brightness_norm + contrast_norm) / 3 - exposure_penalty
    quality_metrics['overall_quality'] = max(0.0, min(1.0, overall_quality)) * 100
    
    return quality_metrics


def create_landmark_heatmap(image_shape: Tuple[int, int], 
                           landmarks: List[Tuple[int, int]],
                           sigma: float = 5.0) -> np.ndarray:
    """
    Create a heatmap visualization of facial landmarks.
    
    Args:
        image_shape: Shape of the output heatmap (height, width)
        landmarks: List of (x, y) landmark coordinates
        sigma: Gaussian sigma for heatmap generation
        
    Returns:
        Heatmap as numpy array
    """
    h, w = image_shape
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    for x, y in landmarks:
        if 0 <= x < w and 0 <= y < h:
            # Create Gaussian around landmark
            y_indices, x_indices = np.ogrid[:h, :w]
            gaussian = np.exp(-((x_indices - x)**2 + (y_indices - y)**2) / (2 * sigma**2))
            heatmap += gaussian
    
    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    # Convert to color heatmap
    heatmap_color = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    return heatmap_color


def image_to_base64(image: np.ndarray, format: str = 'JPEG') -> str:
    """
    Convert OpenCV image to base64 string.
    
    Args:
        image: Input image
        format: Image format ('JPEG', 'PNG')
        
    Returns:
        Base64 encoded string
    """
    # Convert BGR to RGB for PIL
    if len(image.shape) == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Save to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format)
    
    # Encode to base64
    img_str = base64.b64encode(buffer.getvalue()).decode()
    
    return f"data:image/{format.lower()};base64,{img_str}"


def create_detection_summary(results: List[dict]) -> Dict[str, Any]:
    """
    Create a summary of detection results.
    
    Args:
        results: List of detection result dictionaries
        
    Returns:
        Summary statistics
    """
    if not results:
        return {'total_frames': 0, 'faces_detected': 0, 'avg_faces_per_frame': 0.0}
    
    total_faces = sum(len(result.get('faces', [])) for result in results)
    total_frames = len(results)
    
    # Calculate processing time statistics
    processing_times = [result.get('processing_time', 0) for result in results]
    avg_processing_time = np.mean(processing_times) if processing_times else 0
    
    # Calculate confidence statistics
    all_confidences = []
    for result in results:
        for face in result.get('faces', []):
            if 'confidence' in face:
                all_confidences.append(face['confidence'])
    
    summary = {
        'total_frames': total_frames,
        'faces_detected': total_faces,
        'avg_faces_per_frame': total_faces / total_frames if total_frames > 0 else 0.0,
        'avg_processing_time': avg_processing_time,
        'min_processing_time': min(processing_times) if processing_times else 0,
        'max_processing_time': max(processing_times) if processing_times else 0,
        'avg_confidence': np.mean(all_confidences) if all_confidences else 0.0,
        'min_confidence': min(all_confidences) if all_confidences else 0.0,
        'max_confidence': max(all_confidences) if all_confidences else 0.0
    }
    
    return summary


def visualize_face_mesh_3d(landmarks: List[Tuple[float, float, float]], 
                          save_path: Optional[str] = None) -> None:
    """
    Create 3D visualization of face mesh landmarks.
    
    Args:
        landmarks: List of (x, y, z) 3D landmarks
        save_path: Optional path to save the plot
    """
    try:
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Extract coordinates
        xs = [point[0] for point in landmarks]
        ys = [point[1] for point in landmarks]
        zs = [point[2] for point in landmarks]
        
        # Create scatter plot
        ax.scatter(xs, ys, zs, c=zs, cmap='viridis', s=1)
        
        # Set labels and title
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Facial Landmarks')
        
        # Set equal aspect ratio
        max_range = max(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)) / 2
        mid_x = (max(xs) + min(xs)) * 0.5
        mid_y = (max(ys) + min(ys)) * 0.5
        mid_z = (max(zs) + min(zs)) * 0.5
        
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D visualization saved to: {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    except ImportError:
        print("matplotlib not available for 3D visualization")
    except Exception as e:
        print(f"Error creating 3D visualization: {e}")


def batch_process_images(image_paths: List[str], 
                        processor_func,
                        output_dir: str = "batch_output",
                        show_progress: bool = True) -> List[Dict[str, Any]]:
    """
    Process multiple images in batch.
    
    Args:
        image_paths: List of image file paths
        processor_func: Function to process each image
        output_dir: Output directory for results
        show_progress: Whether to show progress
        
    Returns:
        List of processing results
    """
    import os
    try:
        from tqdm import tqdm
    except ImportError:
        # Fallback if tqdm is not available
        def tqdm(iterable, desc=None):
            return iterable
    
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    iterator = tqdm(image_paths, desc="Processing images") if show_progress else image_paths
    
    for i, image_path in enumerate(iterator):
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            # Process image
            result = processor_func(image)
            result['image_path'] = image_path
            result['output_index'] = i
            
            results.append(result)
            
        except Exception as e:
            print(f"Error processing {image_path}: {e}")
            continue
    
    return results


def validate_image(image: Optional[np.ndarray]) -> bool:
    """
    Validate if an image is valid for processing.
    
    Args:
        image: Input image to validate
        
    Returns:
        True if image is valid, False otherwise
    """
    if image is None:
        return False
    
    if not isinstance(image, np.ndarray):
        return False
    
    if image.size == 0:
        return False
    
    # Check dimensions (should be 2D or 3D)
    if len(image.shape) not in [2, 3]:
        return False
    
    # Check data type (should be uint8)
    if image.dtype != np.uint8:
        return False
    
    return True


def validate_video_path(video_path: Optional[str]) -> bool:
    """
    Validate if a video path exists and is accessible.
    
    Args:
        video_path: Path to video file
        
    Returns:
        True if video path is valid, False otherwise
    """
    if not video_path:
        return False
    
    if not isinstance(video_path, str):
        return False
    
    if not os.path.exists(video_path):
        return False
    
    # Check if file has video extension
    valid_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm']
    ext = os.path.splitext(video_path)[1].lower()
    if ext not in valid_extensions:
        return False
    
    return True


def calculate_distance(point1: Tuple[float, ...], point2: Tuple[float, ...]) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        point1: First point (x, y) or (x, y, z)
        point2: Second point (x, y) or (x, y, z)
        
    Returns:
        Distance between points
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have the same number of dimensions")
    
    return np.sqrt(sum((p1 - p2)**2 for p1, p2 in zip(point1, point2)))


def calculate_angle(point1: Tuple[float, float], point2: Tuple[float, float], 
                   point3: Tuple[float, float]) -> float:
    """
    Calculate angle between three points.
    
    Args:
        point1: First point
        point2: Vertex point
        point3: Third point
        
    Returns:
        Angle in degrees
    """
    # Vectors from point2 to point1 and point3
    v1 = np.array([point1[0] - point2[0], point1[1] - point2[1]])
    v2 = np.array([point3[0] - point2[0], point3[1] - point2[1]])
    
    # Calculate angle using dot product
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    return np.degrees(angle)


def normalize_landmarks(landmarks: List[Tuple[float, float]], 
                       image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
    """
    Normalize landmarks to [0, 1] range.
    
    Args:
        landmarks: List of (x, y) landmark coordinates
        image_shape: Shape of image (height, width)
        
    Returns:
        Normalized landmarks
    """
    height, width = image_shape
    normalized = []
    
    for x, y in landmarks:
        norm_x = x / width
        norm_y = y / height
        normalized.append((norm_x, norm_y))
    
    return normalized


def denormalize_landmarks(landmarks: List[Tuple[float, float]], 
                         image_shape: Tuple[int, int]) -> List[Tuple[float, float]]:
    """
    Denormalize landmarks from [0, 1] range to image coordinates.
    
    Args:
        landmarks: List of normalized (x, y) landmark coordinates
        image_shape: Shape of image (height, width)
        
    Returns:
        Denormalized landmarks
    """
    height, width = image_shape
    denormalized = []
    
    for norm_x, norm_y in landmarks:
        x = norm_x * width
        y = norm_y * height
        denormalized.append((x, y))
    
    return denormalized


def get_bounding_box(landmarks: List[Tuple[float, float]]) -> Tuple[int, int, int, int]:
    """
    Get bounding box from landmarks.
    
    Args:
        landmarks: List of (x, y) coordinates
        
    Returns:
        Bounding box (x, y, width, height)
    """
    if not landmarks:
        return (0, 0, 0, 0)
    
    x_coords = [point[0] for point in landmarks]
    y_coords = [point[1] for point in landmarks]
    
    min_x = int(min(x_coords))
    max_x = int(max(x_coords))
    min_y = int(min(y_coords))
    max_y = int(max(y_coords))
    
    width = max_x - min_x
    height = max_y - min_y
    
    return (min_x, min_y, width, height)


def crop_image_region(image: np.ndarray, x: int, y: int, 
                     width: int, height: int) -> np.ndarray:
    """
    Crop a region from an image.
    
    Args:
        image: Input image
        x: Left coordinate
        y: Top coordinate  
        width: Width of region
        height: Height of region
        
    Returns:
        Cropped image region
    """
    img_height, img_width = image.shape[:2]
    
    # Ensure coordinates are within bounds
    x = max(0, min(x, img_width))
    y = max(0, min(y, img_height))
    x2 = max(0, min(x + width, img_width))
    y2 = max(0, min(y + height, img_height))
    
    return image[y:y2, x:x2]


def resize_with_aspect_ratio(image: np.ndarray, target_width: Optional[int] = None,
                           target_height: Optional[int] = None) -> np.ndarray:
    """
    Resize image while maintaining aspect ratio.
    
    Args:
        image: Input image
        target_width: Target width
        target_height: Target height
        
    Returns:
        Resized image
    """
    height, width = image.shape[:2]
    
    if target_width is None and target_height is None:
        return image
    
    if target_width is None:
        # Scale by height
        scale = target_height / height
        new_width = int(width * scale)
        new_height = target_height
    elif target_height is None:
        # Scale by width
        scale = target_width / width
        new_width = target_width
        new_height = int(height * scale)
    else:
        # Scale by the smaller factor to maintain aspect ratio
        scale_w = target_width / width
        scale_h = target_height / height
        scale = min(scale_w, scale_h)
        new_width = int(width * scale)
        new_height = int(height * scale)
    
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def save_image(image: np.ndarray, path: str) -> bool:
    """
    Save image to file.
    
    Args:
        image: Image to save
        path: Output path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        return cv2.imwrite(path, image)
    except Exception:
        return False


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load image from file.
    
    Args:
        path: Image file path
        
    Returns:
        Loaded image or None if failed
    """
    try:
        image = cv2.imread(path)
        return image
    except Exception:
        return None


def create_video_writer(path: str, width: int, height: int, 
                       fps: float = 30.0) -> Optional[cv2.VideoWriter]:
    """
    Create video writer.
    
    Args:
        path: Output video path
        width: Video width
        height: Video height
        fps: Frames per second
        
    Returns:
        VideoWriter object or None if failed
    """
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
        return writer
    except Exception:
        return None


def estimate_pose(landmarks: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Estimate head pose from facial landmarks.
    
    Args:
        landmarks: Facial landmarks
        
    Returns:
        Dictionary with pose angles (yaw, pitch, roll)
    """
    # Simplified pose estimation - return dummy values for now
    return {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}


def calculate_eye_aspect_ratio(eye_landmarks: List[Tuple[float, float]]) -> float:
    """
    Calculate eye aspect ratio.
    
    Args:
        eye_landmarks: Eye landmarks
        
    Returns:
        Eye aspect ratio
    """
    if len(eye_landmarks) < 6:
        return 0.0
    
    # Vertical distances
    v1 = calculate_distance(eye_landmarks[1], eye_landmarks[5])
    v2 = calculate_distance(eye_landmarks[2], eye_landmarks[4])
    
    # Horizontal distance
    h = calculate_distance(eye_landmarks[0], eye_landmarks[3])
    
    if h == 0:
        return 0.0
    
    # Eye aspect ratio
    ear = (v1 + v2) / (2.0 * h)
    return ear


def calculate_mouth_aspect_ratio(mouth_landmarks: List[Tuple[float, float]]) -> float:
    """
    Calculate mouth aspect ratio.
    
    Args:
        mouth_landmarks: Mouth landmarks
        
    Returns:
        Mouth aspect ratio
    """
    if len(mouth_landmarks) < 4:
        return 0.0
    
    # Vertical distance
    v = calculate_distance(mouth_landmarks[1], mouth_landmarks[3])
    
    # Horizontal distance
    h = calculate_distance(mouth_landmarks[0], mouth_landmarks[2])
    
    if h == 0:
        return 0.0
    
    # Mouth aspect ratio
    mar = v / h
    return mar


def smooth_landmarks(landmarks_sequence: List[List[Tuple[float, float]]], 
                    window_size: int = 5) -> List[List[Tuple[float, float]]]:
    """
    Smooth landmarks over time using moving average.
    
    Args:
        landmarks_sequence: Sequence of landmark frames
        window_size: Size of smoothing window
        
    Returns:
        Smoothed landmarks sequence
    """
    if len(landmarks_sequence) < window_size:
        return landmarks_sequence
    
    smoothed = []
    for i in range(len(landmarks_sequence)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(landmarks_sequence), i + window_size // 2 + 1)
        
        # Average landmarks in window
        window_landmarks = landmarks_sequence[start_idx:end_idx]
        if not window_landmarks:
            smoothed.append(landmarks_sequence[i])
            continue
        
        num_landmarks = len(window_landmarks[0])
        averaged_landmarks = []
        
        for j in range(num_landmarks):
            x_sum = sum(frame[j][0] for frame in window_landmarks)
            y_sum = sum(frame[j][1] for frame in window_landmarks)
            x_avg = x_sum / len(window_landmarks)
            y_avg = y_sum / len(window_landmarks)
            averaged_landmarks.append((x_avg, y_avg))
        
        smoothed.append(averaged_landmarks)
    
    return smoothed


def interpolate_landmarks(landmarks1: List[Tuple[float, float]], 
                         landmarks2: List[Tuple[float, float]], 
                         alpha: float) -> List[Tuple[float, float]]:
    """
    Interpolate between two sets of landmarks.
    
    Args:
        landmarks1: First set of landmarks
        landmarks2: Second set of landmarks
        alpha: Interpolation factor (0-1)
        
    Returns:
        Interpolated landmarks
    """
    if len(landmarks1) != len(landmarks2):
        return landmarks1
    
    interpolated = []
    for (x1, y1), (x2, y2) in zip(landmarks1, landmarks2):
        x = x1 + alpha * (x2 - x1)
        y = y1 + alpha * (y2 - y1)
        interpolated.append((x, y))
    
    return interpolated
