import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import requests
import json
import time
from dotenv import load_dotenv

# Datalabs imports (for document extraction pipeline)
from datalabs import process_document, ProcessingStatus, process_document_smart, classify_document
from datalabs.config import settings as datalabs_settings
from datalabs.utils import normalize_entity_extraction_result

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# OpenRouter API configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY', '')
OPENROUTER_BASE_URL = 'https://openrouter.ai/api/v1/chat/completions'

# Model configuration
VISION_MODEL = os.getenv('VISION_MODEL', 'google/gemini-2.0-flash-001')
EXTRACTION_MODEL = os.getenv('EXTRACTION_MODEL', 'google/gemini-2.0-flash-001')

# Flask configuration
FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'true').lower() == 'true'
FLASK_PORT = int(os.getenv('FLASK_PORT', '5050'))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


def clear_upload_folder():
    """Delete all files in the upload folder to keep only the current image."""
    upload_folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")


# =============================================================================
# DOCUMENT DETECTION CONFIGURATION (all configurable via environment)
# =============================================================================
class DocumentDetectionConfig:
    """
    Configurable parameters for document detection.
    All thresholds adapt to image statistics when set to 'auto'.
    """
    def __init__(self):
        # Minimum area ratio (region must be at least this % of image)
        self.min_area_ratio = float(os.getenv('DOC_MIN_AREA_RATIO', '0.03'))

        # Texture threshold - regions with texture score above this are considered patterned
        # 'auto' = calculate based on image statistics
        self.texture_threshold = os.getenv('DOC_TEXTURE_THRESHOLD', 'auto')

        # Aspect ratio range for documents (min, max)
        self.min_aspect_ratio = float(os.getenv('DOC_MIN_ASPECT_RATIO', '1.2'))
        self.max_aspect_ratio = float(os.getenv('DOC_MAX_ASPECT_RATIO', '5.0'))

        # Padding around detected document (pixels)
        self.padding = int(os.getenv('DOC_PADDING', '15'))

        # Brightness percentile for adaptive white detection
        self.brightness_percentile = float(os.getenv('DOC_BRIGHTNESS_PERCENTILE', '70'))

        # Saturation threshold for white detection (0-255)
        self.max_saturation = int(os.getenv('DOC_MAX_SATURATION', '70'))

        # Minimum dimension for valid region (pixels)
        self.min_dimension = int(os.getenv('DOC_MIN_DIMENSION', '50'))

    def get_texture_threshold(self, image):
        """Get texture threshold - adaptive if set to 'auto'."""
        if self.texture_threshold == 'auto':
            # Calculate based on image statistics
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            # Use median of absolute laplacian as baseline
            baseline = np.median(np.abs(laplacian))
            # Threshold is 3x the baseline
            return max(300, baseline * 3)
        return float(self.texture_threshold)

    def get_brightness_threshold(self, image):
        """Get adaptive brightness threshold based on image histogram."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        # Use percentile to find bright regions
        threshold = np.percentile(gray, self.brightness_percentile)
        # Ensure reasonable bounds
        return max(120, min(200, threshold))


# Global config instance
doc_config = DocumentDetectionConfig()


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def image_to_base64(img):
    """Convert OpenCV image to base64 string for displaying in browser."""
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode('utf-8')


def to_python_types(obj):
    """Convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_python_types(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(to_python_types(item) for item in obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def calculate_texture_score(image_region):
    """
    Calculate texture score for a region - high score means patterned/textured.
    Low score means uniform (like paper).
    Uses multiple texture measures for robustness.
    """
    if image_region.size == 0:
        return float('inf')

    gray = cv2.cvtColor(image_region, cv2.COLOR_BGR2GRAY) if len(image_region.shape) == 3 else image_region

    # Measure 1: Laplacian variance (edge density)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian_var = laplacian.var()

    # Measure 2: Local standard deviation (texture roughness)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    local_std = np.std(gray.astype(float) - blur.astype(float))

    # Measure 3: Gradient magnitude variance
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.sqrt(sobel_x**2 + sobel_y**2)
    gradient_var = gradient_mag.var()

    # Combined texture score (weighted)
    return laplacian_var * 0.4 + local_std * 15 + gradient_var * 0.3


def calculate_color_uniformity(image_region):
    """
    Calculate how uniform the color is in a region.
    High score = uniform color (like white paper).
    Low score = varied colors (like patterned fabric).
    """
    if image_region.size == 0:
        return 0

    hsv = cv2.cvtColor(image_region, cv2.COLOR_BGR2HSV) if len(image_region.shape) == 3 else image_region

    # Calculate standard deviation of each channel
    h_std = np.std(hsv[:, :, 0])
    s_std = np.std(hsv[:, :, 1])
    v_std = np.std(hsv[:, :, 2])

    # Low std = uniform, so invert
    # Saturation uniformity is most important for detecting white paper
    uniformity = 1.0 / (1.0 + s_std / 30.0 + h_std / 50.0 + v_std / 40.0)

    return uniformity


def extract_white_document(image):
    """
    Extract the white/light colored document region from the image.
    Uses adaptive thresholds based on image statistics.
    Filters out patterned backgrounds by analyzing texture and color uniformity.
    """
    original = image.copy()
    height, width = image.shape[:2]
    image_area = height * width

    # Get adaptive thresholds
    brightness_threshold = doc_config.get_brightness_threshold(image)
    texture_threshold = doc_config.get_texture_threshold(image)

    # Convert to different color spaces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Adaptive white detection based on image brightness
    # Lower bound: adaptive brightness, low saturation
    lower_white = np.array([0, 0, int(brightness_threshold)], dtype=np.uint8)
    upper_white = np.array([180, doc_config.max_saturation, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Also include slightly darker regions (for shadows on paper)
    lower_light = np.array([0, 0, int(brightness_threshold * 0.8)], dtype=np.uint8)
    upper_light = np.array([180, doc_config.max_saturation + 10, 255], dtype=np.uint8)
    light_mask = cv2.inRange(hsv, lower_light, upper_light)

    # Combine masks
    combined_mask = cv2.bitwise_or(white_mask, light_mask)

    # Adaptive morphology based on image size
    kernel_size = max(3, min(7, int(min(width, height) / 100)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return original, False, None

    # Score each contour using multiple criteria
    scored_contours = []
    for contour in contours:
        area = cv2.contourArea(contour)

        # Skip tiny regions (configurable threshold)
        if area < image_area * doc_config.min_area_ratio:
            continue

        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)

        # Skip if dimensions too small
        if w < doc_config.min_dimension or h < doc_config.min_dimension:
            continue

        # Calculate rectangularity (how rectangular is the contour)
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0

        # Calculate aspect ratio score (prefer document-like ratios)
        aspect_ratio = max(w, h) / min(w, h) if min(w, h) > 0 else 1
        if doc_config.min_aspect_ratio < aspect_ratio < doc_config.max_aspect_ratio:
            aspect_score = 1.0
        else:
            # Penalize but don't reject (might be square document)
            aspect_score = 0.7

        # Extract region for analysis
        pad = 5
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(width, x + w + pad), min(height, y + h + pad)
        region = original[y1:y2, x1:x2]

        # Calculate texture score (low = good for paper)
        texture_score = calculate_texture_score(region)

        # Calculate color uniformity (high = good for paper)
        uniformity = calculate_color_uniformity(region)

        # Texture quality: inverse of texture score, normalized by adaptive threshold
        texture_quality = 1.0 / (1.0 + texture_score / texture_threshold)

        # Combined score: area * rectangularity * aspect * texture * uniformity
        final_score = (
            (area / image_area) *
            rectangularity *
            aspect_score *
            texture_quality *
            (uniformity + 0.2)  # Boost uniformity slightly
        )

        scored_contours.append({
            'contour': contour,
            'score': final_score,
            'area': area,
            'bounds': (x, y, w, h),
            'texture': texture_score,
            'uniformity': uniformity,
            'rectangularity': rectangularity,
            'aspect_ratio': aspect_ratio
        })

    if not scored_contours:
        return original, False, None

    # Sort by score (highest first)
    scored_contours.sort(key=lambda x: x['score'], reverse=True)

    # Use the best scoring contour
    best = scored_contours[0]

    # Check if it's significant enough
    if best['area'] < image_area * 0.05:
        return original, False, None

    x, y, w, h = best['bounds']

    # Add configurable padding
    padding = doc_config.padding
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(width - x, w + 2 * padding)
    h = min(height - y, h + 2 * padding)

    # Crop the document region
    cropped = original[y:y+h, x:x+w]

    bounds = {
        'x': x, 'y': y, 'width': w, 'height': h,
        'texture_score': round(best['texture'], 2),
        'uniformity': round(best['uniformity'], 3),
        'rectangularity': round(best['rectangularity'], 3),
        'aspect_ratio': round(best['aspect_ratio'], 2),
        'adaptive_thresholds': {
            'brightness': round(brightness_threshold, 1),
            'texture': round(texture_threshold, 1)
        }
    }

    return cropped, True, bounds


def order_points(pts):
    """
    Order points in: top-left, top-right, bottom-right, bottom-left order.
    """
    rect = np.zeros((4, 2), dtype="float32")

    # Sum of coordinates: top-left has smallest, bottom-right has largest
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right

    # Difference of coordinates: top-right has smallest, bottom-left has largest
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left

    return rect


def create_white_region_mask(image):
    """
    Create a mask of white/paper regions in the image.
    Used to constrain edge detection to document areas.
    """
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get adaptive brightness threshold
    brightness_threshold = doc_config.get_brightness_threshold(image)

    # Create white mask
    lower = np.array([0, 0, int(brightness_threshold * 0.85)], dtype=np.uint8)
    upper = np.array([180, doc_config.max_saturation + 20, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)

    return mask


def calculate_white_overlap(contour, white_mask):
    """
    Calculate what percentage of the contour area overlaps with white regions.
    High overlap = likely a document, low overlap = likely background.
    """
    # Create a mask from the contour
    contour_mask = np.zeros(white_mask.shape, dtype=np.uint8)
    cv2.drawContours(contour_mask, [contour], -1, 255, -1)

    # Calculate overlap
    overlap = cv2.bitwise_and(contour_mask, white_mask)
    contour_area = cv2.countNonZero(contour_mask)
    overlap_area = cv2.countNonZero(overlap)

    if contour_area == 0:
        return 0

    return overlap_area / contour_area


def crop_document(image):
    """
    Automatically detect and crop the document from the image.
    Uses white region mask to constrain edge detection and filter out patterned backgrounds.
    Applies perspective transformation to get a top-down view.
    """
    original = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = image.shape[:2]
    image_area = height * width

    # Create white region mask to constrain detection
    white_mask = create_white_region_mask(image)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection with multiple thresholds (adaptive based on image stats)
    median_val = np.median(blurred)
    lower = int(max(0, median_val * 0.5))
    upper = int(min(255, median_val * 1.5))

    edges1 = cv2.Canny(blurred, lower, upper)
    edges2 = cv2.Canny(blurred, 50, 150)
    edges3 = cv2.Canny(blurred, 30, 100)

    # Combine edges
    edges = cv2.bitwise_or(edges1, edges2)
    edges = cv2.bitwise_or(edges, edges3)

    # IMPORTANT: Mask edges to only keep edges within or near white regions
    # Dilate white mask slightly to include document borders
    dilated_white = cv2.dilate(white_mask, np.ones((15, 15), np.uint8), iterations=2)
    edges = cv2.bitwise_and(edges, dilated_white)

    # Dilate to connect edge segments
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=2)
    edges = cv2.erode(edges, kernel, iterations=1)

    document_contour = None
    result_with_corners = original.copy()
    detection_method = 'none'

    def find_best_document_contour(edge_image, white_mask, min_area_ratio=0.03):
        """
        Find document contour, scoring by white region overlap.
        Returns contour with highest white overlap score.
        """
        contours, _ = cv2.findContours(edge_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:20]

        candidates = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < image_area * min_area_ratio:
                continue

            peri = cv2.arcLength(contour, True)

            # Try to approximate to 4 points
            approx = None
            for tolerance in [0.01, 0.02, 0.03, 0.04, 0.05]:
                temp_approx = cv2.approxPolyDP(contour, tolerance * peri, True)
                if len(temp_approx) == 4:
                    approx = temp_approx
                    break
                elif 4 < len(temp_approx) <= 6:
                    hull = cv2.convexHull(contour)
                    hull_approx = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True)
                    if len(hull_approx) == 4:
                        approx = hull_approx
                        break

            if approx is None:
                continue

            # Calculate white overlap score
            white_overlap = calculate_white_overlap(approx, white_mask)

            # Calculate texture score for the region
            x, y, w, h = cv2.boundingRect(approx)
            region = original[max(0,y):min(height,y+h), max(0,x):min(width,x+w)]
            texture = calculate_texture_score(region) if region.size > 0 else float('inf')

            # Combined score: prefer high white overlap and low texture
            score = white_overlap * (1.0 / (1.0 + texture / 500.0))

            candidates.append({
                'contour': approx,
                'area': area,
                'white_overlap': white_overlap,
                'texture': texture,
                'score': score
            })

        if not candidates:
            return None, 0

        # Return best candidate
        best = max(candidates, key=lambda x: x['score'])

        # Require minimum white overlap (configurable threshold)
        if best['white_overlap'] < 0.4:
            return None, 0

        return best['contour'], best['white_overlap']

    # Try to find document contour using masked edges
    document_contour, white_overlap = find_best_document_contour(edges, white_mask)
    if document_contour is not None:
        detection_method = f'edge detection (white overlap: {white_overlap:.0%})'

    # Fallback: use white region bounding box directly
    if document_contour is None:
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if white_contours:
            # Find largest white contour
            largest_white = max(white_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_white)

            if area > image_area * 0.05:
                # Use bounding rectangle of white region
                rect = cv2.minAreaRect(largest_white)
                box = cv2.boxPoints(rect)
                document_contour = np.intp(box).reshape(4, 1, 2)
                detection_method = 'white region bounding box'

    # Track if we found a real document boundary
    used_fallback = False

    # Last resort: use image boundaries with margin
    if document_contour is None:
        used_fallback = True
        margin = int(min(width, height) * 0.02)
        document_contour = np.array([
            [[margin, margin]],
            [[width - margin, margin]],
            [[width - margin, height - margin]],
            [[margin, height - margin]]
        ], dtype=np.int32)
        detection_method = 'fallback (full image)'

    # Calculate original image area for crop ratio
    orig_height, orig_width = original.shape[:2]
    original_area = orig_height * orig_width

    crop_info = {
        'document_detected': not used_fallback,
        'crop_successful': False,
        'original_corners': None,
        'output_size': None,
        'original_size': {'width': orig_width, 'height': orig_height},
        'detection_method': detection_method,
        'crop_ratio': 1.0,  # Default: no cropping
        'crop_quality': 'good',
        'crop_warning': None
    }

    if document_contour is not None:
        # Get the corners and order them
        pts = document_contour.reshape(4, 2)
        rect = order_points(pts)

        # Draw corners on result image
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        labels = ['TL', 'TR', 'BR', 'BL']
        for i, (point, color, label) in enumerate(zip(rect, colors, labels)):
            cv2.circle(result_with_corners, tuple(point.astype(int)), 15, color, -1)
            cv2.putText(result_with_corners, label,
                       (int(point[0]) - 10, int(point[1]) - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.drawContours(result_with_corners, [document_contour], -1, (0, 255, 0), 3)

        # Calculate dimensions of the new image
        (tl, tr, br, bl) = rect

        # Width: max of top edge and bottom edge
        width_top = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        width_bottom = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        max_width = max(int(width_top), int(width_bottom))

        # Height: max of left edge and right edge
        height_left = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        height_right = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        max_height = max(int(height_left), int(height_right))

        # Ensure minimum dimensions
        max_width = max(max_width, doc_config.min_dimension)
        max_height = max(max_height, doc_config.min_dimension)

        # Destination points for perspective transform
        dst = np.array([
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1]
        ], dtype="float32")

        # Compute perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(rect, dst)
        cropped = cv2.warpPerspective(original, M, (max_width, max_height))

        # Calculate crop ratio and quality
        cropped_area = max_width * max_height
        crop_ratio = cropped_area / original_area if original_area > 0 else 1.0

        # Determine crop quality based on ratio
        if crop_ratio < 0.25:
            crop_quality = 'poor'
            crop_warning = 'Severe over-crop: Less than 25% of original image retained. Document may be cut off.'
        elif crop_ratio < 0.40:
            crop_quality = 'fair'
            crop_warning = 'Possible over-crop: Less than 40% of original retained. Some content may be missing.'
        elif crop_ratio > 0.95:
            crop_quality = 'minimal'
            crop_warning = None  # Almost no cropping, which is fine
        else:
            crop_quality = 'good'
            crop_warning = None

        # Check if corners are too close to edges (might be cutting off content)
        edge_margin = 20  # pixels
        corners_near_edge = []
        for i, (x, y) in enumerate(rect):
            near_edges = []
            if x < edge_margin:
                near_edges.append('left')
            if x > orig_width - edge_margin:
                near_edges.append('right')
            if y < edge_margin:
                near_edges.append('top')
            if y > orig_height - edge_margin:
                near_edges.append('bottom')
            if near_edges:
                corners_near_edge.append({'corner': ['TL', 'TR', 'BR', 'BL'][i], 'edges': near_edges})

        if corners_near_edge and crop_warning is None:
            crop_warning = f'Document extends to image edges - content may be cut off at: {", ".join([c["corner"] for c in corners_near_edge])}'

        crop_info.update({
            'crop_successful': True,
            'original_corners': rect.tolist(),
            'output_size': {'width': max_width, 'height': max_height},
            'detection_method': detection_method,
            'crop_ratio': round(crop_ratio, 3),
            'crop_quality': crop_quality,
            'crop_warning': crop_warning,
            'corners_near_edge': corners_near_edge
        })

        return cropped, result_with_corners, crop_info

    # If no document found, return original image
    return original, result_with_corners, crop_info


def auto_enhance_document(image):
    """
    Enhance the cropped document for better readability.
    """
    # Convert to grayscale for processing
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_gray = clahe.apply(gray)

    # Denoise
    denoised = cv2.fastNlMeansDenoising(enhanced_gray, None, 10, 7, 21)

    # Sharpen
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(denoised, -1, kernel)

    # Adaptive thresholding for document binarization
    binary = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)

    # Convert back to BGR for consistency
    enhanced_color = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2BGR)
    binary_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return enhanced_color, binary_color


def call_openrouter_vision(image_base64, prompt, model=None):
    """
    Call OpenRouter API with a vision-capable model.
    """
    if model is None:
        model = VISION_MODEL
    if not OPENROUTER_API_KEY:
        return {"error": "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5050",
        "X-Title": "Document Analyzer"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
        "provider": {
            "require_parameters": True
        },
        "reasoning": {
            "effort": "none"
        }
    }

    try:
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return {
            "success": True,
            "content": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "model": model,
            "usage": result.get("usage", {})
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


# =============================================================================
# LEGACY EXTRACTION - Direct GPT-4o extraction
# =============================================================================

LEGACY_EXTRACTION_PROMPT = """You are a deterministic, evidence-driven document analysis engine.

Your tasks are:

1. **Classify** the document or multiple overlaid documents. Choose one or more from:
   - `thermal_print` (narrow roll receipt, usually from POS)
   - `a4_invoice` (formal A4 invoice with structured layout)
   - `handprint` (handwritten)
   - `pos_receipt` (point-of-sale receipt with card/payment info)

2. **Extract structured data** ONLY from what is visible. For each field:
   - If clearly visible → return exact value
   - If partially visible → return partial value, mark `is_partially_visible: true`
   - If hidden, blurry, cropped, or cut-off → return `value: null`, `status: "missing_value"`
   - Never infer, guess, or assume

3. **Summarize** the document in natural language:
   - Describe what type(s) of documents are visible
   - Mention supplier if known
   - Explain what's readable and what isn't (e.g. blurred, covered)
   - Keep it factual, non-speculative, and strictly tied to visible evidence

---

⚠️ STRICT RULES:
- ❌ Never hallucinate supplier, logo, brand, location, or VAT from format or card
- ❌ Do not complete partially visible text
- ✅ Only extract what is explicitly and clearly visible
- ✅ Use null and clear notes for obscured/missing info

---

📤 OUTPUT FORMAT:

Return a JSON with 3 top-level keys:

{
  "classification": [
    {
      "document_type": "...",
      "reasoning": "..."
    }
  ],
  "extracted": {
    "supplier": {
      "value": "...",
      "is_partially_visible": true | false,
      "is_inferred": false,
      "status": "...",
      "notes": "..."
    },
    "vat_number": { "value": "...", "status": "...", "notes": "..." },
    "transaction_date": { "value": "...", "status": "...", "notes": "..." },
    "payment_method": { "value": "...", "status": "...", "notes": "..." },
    "vat_breakdown": {
      "rate": "...", "net": "...", "vat": "...", "gross": "...", "status": "..."
    },
    "items": [
      {
        "description": "...",
        "unit_price": "...",
        "quantity": "...",
        "total": "..."
      }
    ],
    "total_amount": { "value": "...", "status": "..." }
  },
  "summary": "A natural language summary describing the document(s), what is visible, what is readable, and any issues like blur, obstruction, or cropping."
}

Only output the JSON object, no explanations."""


LEGACY_MODELS = {
    "gpt-4o": "openai/gpt-4o",
    "qwen-vl": "qwen/qwen3-vl-30b-a3b-instruct",
    "gemini": "google/gemini-2.0-flash-001"
}


def legacy_extract(image_base64, model_key="gpt-4o"):
    """
    Legacy extraction - Send image directly to vision model for extraction.
    Simple, single-call approach without complex pipeline.

    model_key: "gpt-4o", "qwen-vl", or "gemini"
    """
    if not OPENROUTER_API_KEY:
        return {"error": "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."}

    # Get model from key
    model = LEGACY_MODELS.get(model_key, LEGACY_MODELS["gpt-4o"])

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5050",
        "X-Title": "Document Analyzer - Legacy"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": LEGACY_EXTRACTION_PROMPT
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.0
    }

    try:
        start_time = time.time()
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        elapsed_ms = int((time.time() - start_time) * 1000)

        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Try to parse as JSON
        try:
            # Remove markdown code blocks if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            parsed = json.loads(content.strip())
        except json.JSONDecodeError:
            parsed = None

        return {
            "success": True,
            "raw_response": content,
            "parsed": parsed,
            "model": model,
            "processing_time_ms": elapsed_ms,
            "usage": result.get("usage", {})
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def call_openrouter_vision_multi(images, prompt, model=None):
    """
    Call OpenRouter API with multiple images for vision analysis.
    images: list of dicts with 'base64' and 'label' keys
    """
    if model is None:
        model = VISION_MODEL
    if not OPENROUTER_API_KEY:
        return {"error": "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5050",
        "X-Title": "Document Analyzer"
    }

    # Build content array with all images first, then the prompt
    content = []
    for img in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img['base64']}"
            }
        })
    content.append({
        "type": "text",
        "text": prompt
    })

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
        "provider": {
            "require_parameters": True
        },
        "reasoning": {
            "effort": "none"
        }
    }

    try:
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return {
            "success": True,
            "content": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "model": model,
            "usage": result.get("usage", {})
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def convert_image_to_html(original_base64, binary_base64=None, enhanced_base64=None):
    """
    Send image(s) to LLM to convert to structured HTML.
    - original_base64: Original uncropped image (always sent to ensure no data loss)
    - binary_base64: Binary/thresholded version for text clarity
    - enhanced_base64: Enhanced cropped version (only when crop quality is good)
    """
    # Determine number of images being sent
    num_images = 1 + (1 if binary_base64 else 0) + (1 if enhanced_base64 else 0)

    # Build prompt based on which images we have
    if num_images == 3:
        # All three images: Original + Binary + Enhanced
        image_intro = """You are a document OCR specialist. You are provided with THREE images of the SAME document that you MUST use together:

═══════════════════════════════════════════════════════════════════
IMAGE 1 (Original): The full original image as captured
- Use for: Full document view, colors, logos, visual context
- Shows: Complete document with all content
═══════════════════════════════════════════════════════════════════
IMAGE 2 (Binary/Thresholded): Black-and-white high-contrast version
- Use for: Clear text edges, better character recognition
- Shows: High-contrast text that may be faint in original
═══════════════════════════════════════════════════════════════════
IMAGE 3 (Enhanced): Contrast-enhanced version for better readability
- Use for: Improved visibility of faded or low-contrast areas
- Shows: Same content as original with enhanced contrast
═══════════════════════════════════════════════════════════════════

CRITICAL - COMBINE ALL THREE IMAGES:
1. Use IMAGE 1 (Original) to see colors, logos, and layout
2. Use IMAGE 2 (Binary) to read text more accurately where original is unclear
3. Use IMAGE 3 (Enhanced) for better contrast on faded text
4. Cross-reference ALL images to maximize OCR accuracy

Your goal: Extract ALL visible text by leveraging all three image versions for maximum accuracy."""

    elif num_images == 2 and binary_base64:
        # Two images: Original + Binary
        image_intro = """You are a document OCR specialist. You are provided with TWO images of the SAME document that you MUST use together:

═══════════════════════════════════════════════════════════════════
IMAGE 1 (Original): The original color/grayscale document
- Use for: Colors, logos, visual context, layout understanding
- Shows: Natural appearance of the document
═══════════════════════════════════════════════════════════════════
IMAGE 2 (Binary/Thresholded): Black-and-white processed version
- Use for: Clear text edges, better character recognition
- Shows: High-contrast text that may be faint in original
═══════════════════════════════════════════════════════════════════

CRITICAL - COMBINE BOTH IMAGES:
1. Look at IMAGE 1 (Original) to understand document layout and structure
2. Look at IMAGE 2 (Binary) to read text more accurately, especially:
   - Faded or low-contrast text
   - Small print or fine details
   - Text in shadowed or uneven lighting areas
3. Cross-reference BOTH images for every piece of text you extract
4. If text is unclear in Original, check Binary for better clarity
5. Create ONE unified HTML output that combines insights from BOTH images

Your goal: Extract ALL visible text by leveraging both image versions for maximum accuracy."""
    else:
        image_intro = """You are a document OCR specialist. Analyze this document image and convert it to clean, semantic HTML."""

    prompt = f"""{image_intro}

STEP 1 - IMAGE QUALITY ANALYSIS (Include this at the TOP of your HTML output):
Analyze the image and report findings with SPECIFIC POSITIONS:
- Obstructions: Is any part covered? Specify POSITION (top-left, top-right, bottom-left, bottom-right, center, left-edge, right-edge, top-edge, bottom-edge)
- Folds/Creases: Are there fold lines? Specify POSITION and direction (horizontal/vertical/diagonal)
- Corners: Check EACH corner separately:
  * Top-Left: [Clear / Folded / Cut-off / Obstructed / Distorted]
  * Top-Right: [Clear / Folded / Cut-off / Obstructed / Distorted]
  * Bottom-Left: [Clear / Folded / Cut-off / Obstructed / Distorted]
  * Bottom-Right: [Clear / Folded / Cut-off / Obstructed / Distorted]
- Edges: Check EACH edge:
  * Top Edge: [Complete / Clipped / Damaged]
  * Bottom Edge: [Complete / Clipped / Damaged]
  * Left Edge: [Complete / Clipped / Damaged]
  * Right Edge: [Complete / Clipped / Damaged]
- Distortion: Is any area warped/curved/perspective-skewed? Specify POSITION
- Blur: Where are blurry areas? Specify POSITION
- Lighting: Where are shadows/glare? Specify POSITION
- Skew/Rotation: Document tilt direction and severity
- Overall Quality: [Good / Fair / Poor]

STEP 2 - TEXT EXTRACTION:

CRITICAL GUARDRAILS - YOU MUST FOLLOW THESE STRICTLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ ZERO TOLERANCE FOR INVENTED DATA - Only extract what you can CLEARLY SEE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEVER INVENT OR GUESS:
   - DO NOT guess dates, amounts, names, or any values
   - DO NOT complete partial words or numbers
   - DO NOT infer what "should" be there based on context
   - DO NOT use your training knowledge to fill gaps
   - If you're not 100% certain, mark it as unclear

2. USE PLACEHOLDERS FOR UNCLEAR DATA:
   - [UNREADABLE] - Text exists but cannot be read at all
   - [ILLEGIBLE] - Text is blurry, faded, or unclear
   - [PARTIAL: visible_part...] - Only part of text is visible (show what you see)
   - [OBSTRUCTED] - Text is covered by finger, fold, or object
   - [BLANK] - Field label exists but value area is empty

3. NUMBERS AND AMOUNTS:
   - If a digit is unclear, use "?" for that digit (e.g., "12?.99" or "?5.00")
   - NEVER round or estimate amounts
   - If decimal point is unclear, mark the whole amount as [ILLEGIBLE]
   - DO NOT assume currency symbols - only include if clearly visible

4. DATES:
   - If any part of date is unclear, mark it (e.g., "15/?/2024" or "[ILLEGIBLE]")
   - NEVER assume date formats or complete partial dates
   - DO NOT convert or standardize dates - extract exactly as shown

5. NAMES AND ADDRESSES:
   - If a letter is unclear, use "?" (e.g., "J?hn Sm?th")
   - NEVER guess spelling of names or places
   - Partial addresses should show only visible parts

6. EMPTY VS UNREADABLE:
   - If a field is genuinely empty in the document, leave it empty
   - If a field has content you can't read, use appropriate placeholder
{"- ALWAYS check the BINARY image (Image 2) when text in the Original (Image 1) is unclear - the binary version often reveals text that appears faded or hard to read in the original" if binary_base64 else ""}

INSTRUCTIONS:
1. Extract ALL text from the document exactly as it appears
2. Preserve the document structure using appropriate HTML tags
3. Use semantic HTML: <header>, <section>, <table>, <address>, etc.
4. For tables, use proper <table>, <tr>, <th>, <td> tags
5. For addresses, use <address> tags with <br> for line breaks
6. Include any visible labels (like "Bill To:", "Invoice #:", etc.) as <strong> or <label> tags
7. Maintain the visual hierarchy using headings (h1-h6)
8. LOGO DETECTION: If you see any company/brand logo in the document:
   - Identify the company/brand name from the logo or surrounding text
   - Represent it as: <div class="logo" data-brand="[BRAND NAME]">[BRAND NAME] Logo</div>
   - Place it in the appropriate position in the HTML structure (usually in header)

OUTPUT FORMAT:
Return ONLY the HTML code, no explanations. Start with <!DOCTYPE html>.
Include the image analysis as the FIRST element in <body> using this format:

<div class="image-analysis" style="background:#f8f9fa;padding:20px;margin-bottom:20px;border-radius:8px;border-left:4px solid #dc3545;">
  <h3 style="margin:0 0 15px 0;color:#333;">📋 Image Quality Analysis</h3>

  <div style="margin-bottom:15px;">
    <strong style="color:#dc3545;">Obstructions:</strong> [None / Yes - specify position: top-left, center, etc.]
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#fd7e14;">Folds/Creases:</strong> [None / Yes - specify position and direction]
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#6f42c1;">Corner Status:</strong>
    <ul style="margin:5px 0 0 20px;padding:0;">
      <li>Top-Left: [Clear / Folded / Cut-off / Obstructed / Distorted]</li>
      <li>Top-Right: [Clear / Folded / Cut-off / Obstructed / Distorted]</li>
      <li>Bottom-Left: [Clear / Folded / Cut-off / Obstructed / Distorted]</li>
      <li>Bottom-Right: [Clear / Folded / Cut-off / Obstructed / Distorted]</li>
    </ul>
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#20c997;">Edge Status:</strong>
    <ul style="margin:5px 0 0 20px;padding:0;">
      <li>Top: [Complete / Clipped / Damaged]</li>
      <li>Bottom: [Complete / Clipped / Damaged]</li>
      <li>Left: [Complete / Clipped / Damaged]</li>
      <li>Right: [Complete / Clipped / Damaged]</li>
    </ul>
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#0dcaf0;">Distortion:</strong> [None / Yes - specify position and type]
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#6c757d;">Blur:</strong> [None / Yes - specify position]
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#ffc107;">Lighting Issues:</strong> [None / Shadows at position / Glare at position]
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#198754;">Skew/Rotation:</strong> [None / Slight / Significant - direction]
  </div>

  <div style="padding:10px;background:#fff;border-radius:5px;margin-top:10px;">
    <strong>Overall Quality:</strong>
    <span style="padding:3px 10px;border-radius:3px;background:[#28a745 for Good / #ffc107 for Fair / #dc3545 for Poor];color:#fff;">[Good / Fair / Poor]</span>
  </div>
</div>

Then include the extracted document content below."""

    # Build images list based on what's provided
    images = [{"base64": original_base64, "label": "original"}]

    if binary_base64:
        images.append({"base64": binary_base64, "label": "binary"})

    if enhanced_base64:
        images.append({"base64": enhanced_base64, "label": "enhanced"})

    # Use multi-image API if we have more than one image
    if len(images) > 1:
        return call_openrouter_vision_multi(images, prompt)
    else:
        return call_openrouter_vision(original_base64, prompt)


def convert_image_to_html_advanced(enhanced_base64, gradient_base64):
    """
    Send multiple images to LLM for advanced document analysis.
    Uses both enhanced image and gradient image for better text detection.
    """
    if not OPENROUTER_API_KEY:
        return {"error": "OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable."}

    prompt = """You are a document OCR specialist with advanced image analysis capabilities. You are provided with TWO images of the same document:

IMAGE 1 (Enhanced): A contrast-enhanced version of the document for clear text visibility
IMAGE 2 (Gradient): A gradient/edge analysis showing text boundaries and structural elements

STEP 1 - IMAGE QUALITY ANALYSIS (Include this at the TOP of your HTML output):
Analyze the image and report findings with SPECIFIC POSITIONS:
- Obstructions: Is any part covered? Specify POSITION (top-left, top-right, bottom-left, bottom-right, center, left-edge, right-edge, top-edge, bottom-edge)
- Folds/Creases: Are there fold lines? Specify POSITION and direction (horizontal/vertical/diagonal)
- Corners: Check EACH corner separately:
  * Top-Left: [Clear / Folded / Cut-off / Obstructed / Distorted]
  * Top-Right: [Clear / Folded / Cut-off / Obstructed / Distorted]
  * Bottom-Left: [Clear / Folded / Cut-off / Obstructed / Distorted]
  * Bottom-Right: [Clear / Folded / Cut-off / Obstructed / Distorted]
- Edges: Check EACH edge:
  * Top Edge: [Complete / Clipped / Damaged]
  * Bottom Edge: [Complete / Clipped / Damaged]
  * Left Edge: [Complete / Clipped / Damaged]
  * Right Edge: [Complete / Clipped / Damaged]
- Distortion: Is any area warped/curved/perspective-skewed? Specify POSITION
- Blur: Where are blurry areas? Specify POSITION
- Lighting: Where are shadows/glare? Specify POSITION
- Skew/Rotation: Document tilt direction and severity
- Overall Quality: [Good / Fair / Poor]

STEP 2 - TEXT EXTRACTION:

CRITICAL GUARDRAILS - YOU MUST FOLLOW THESE STRICTLY:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️ ZERO TOLERANCE FOR INVENTED DATA - Only extract what you can CLEARLY SEE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. NEVER INVENT OR GUESS:
   - DO NOT guess dates, amounts, names, or any values
   - DO NOT complete partial words or numbers
   - DO NOT infer what "should" be there based on context
   - DO NOT use your training knowledge to fill gaps
   - If you're not 100% certain, mark it as unclear

2. USE PLACEHOLDERS FOR UNCLEAR DATA:
   - [UNREADABLE] - Text exists but cannot be read at all
   - [ILLEGIBLE] - Text is blurry, faded, or unclear
   - [PARTIAL: visible_part...] - Only part of text is visible (show what you see)
   - [OBSTRUCTED] - Text is covered by finger, fold, or object
   - [BLANK] - Field label exists but value area is empty

3. NUMBERS AND AMOUNTS:
   - If a digit is unclear, use "?" for that digit (e.g., "12?.99" or "?5.00")
   - NEVER round or estimate amounts
   - If decimal point is unclear, mark the whole amount as [ILLEGIBLE]
   - DO NOT assume currency symbols - only include if clearly visible

4. DATES:
   - If any part of date is unclear, mark it (e.g., "15/?/2024" or "[ILLEGIBLE]")
   - NEVER assume date formats or complete partial dates
   - DO NOT convert or standardize dates - extract exactly as shown

5. NAMES AND ADDRESSES:
   - If a letter is unclear, use "?" (e.g., "J?hn Sm?th")
   - NEVER guess spelling of names or places
   - Partial addresses should show only visible parts

6. EMPTY VS UNREADABLE:
   - If a field is genuinely empty in the document, leave it empty
   - If a field has content you can't read, use appropriate placeholder

ADVANCED ANALYSIS INSTRUCTIONS:
1. Use the ENHANCED image as your primary source for reading text content
2. Use the GRADIENT image to identify text boundaries, table structures, and layout elements
3. Cross-reference both images to improve accuracy - if text is unclear in one, check the other
4. The gradient image helps identify: table borders, text line separations, section boundaries
5. Extract ALL text from the document exactly as it appears
6. Preserve the document structure using appropriate HTML tags
7. Use semantic HTML: <header>, <section>, <table>, <address>, etc.
8. For tables, use proper <table>, <tr>, <th>, <td> tags
9. For addresses, use <address> tags with <br> for line breaks
10. Include any visible labels (like "Bill To:", "Invoice #:", etc.) as <strong> or <label> tags
11. Maintain the visual hierarchy using headings (h1-h6)
12. LOGO DETECTION: If you see any company/brand logo in the document:
    - Identify the company/brand name from the logo or surrounding text
    - Represent it as: <div class="logo" data-brand="[BRAND NAME]">[BRAND NAME] Logo</div>
    - Place it in the appropriate position in the HTML structure (usually in header)

OUTPUT FORMAT:
Return ONLY the HTML code, no explanations. Start with <!DOCTYPE html>.
Include the image analysis as the FIRST element in <body> using this format:

<div class="image-analysis" style="background:#f8f9fa;padding:20px;margin-bottom:20px;border-radius:8px;border-left:4px solid #dc3545;">
  <h3 style="margin:0 0 15px 0;color:#333;">📋 Image Quality Analysis</h3>

  <div style="margin-bottom:15px;">
    <strong style="color:#dc3545;">Obstructions:</strong> [None / Yes - specify position: top-left, center, etc.]
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#fd7e14;">Folds/Creases:</strong> [None / Yes - specify position and direction]
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#6f42c1;">Corner Status:</strong>
    <ul style="margin:5px 0 0 20px;padding:0;">
      <li>Top-Left: [Clear / Folded / Cut-off / Obstructed / Distorted]</li>
      <li>Top-Right: [Clear / Folded / Cut-off / Obstructed / Distorted]</li>
      <li>Bottom-Left: [Clear / Folded / Cut-off / Obstructed / Distorted]</li>
      <li>Bottom-Right: [Clear / Folded / Cut-off / Obstructed / Distorted]</li>
    </ul>
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#20c997;">Edge Status:</strong>
    <ul style="margin:5px 0 0 20px;padding:0;">
      <li>Top: [Complete / Clipped / Damaged]</li>
      <li>Bottom: [Complete / Clipped / Damaged]</li>
      <li>Left: [Complete / Clipped / Damaged]</li>
      <li>Right: [Complete / Clipped / Damaged]</li>
    </ul>
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#0dcaf0;">Distortion:</strong> [None / Yes - specify position and type]
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#6c757d;">Blur:</strong> [None / Yes - specify position]
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#ffc107;">Lighting Issues:</strong> [None / Shadows at position / Glare at position]
  </div>

  <div style="margin-bottom:15px;">
    <strong style="color:#198754;">Skew/Rotation:</strong> [None / Slight / Significant - direction]
  </div>

  <div style="padding:10px;background:#fff;border-radius:5px;margin-top:10px;">
    <strong>Overall Quality:</strong>
    <span style="padding:3px 10px;border-radius:3px;background:[#28a745 for Good / #ffc107 for Fair / #dc3545 for Poor];color:#fff;">[Good / Fair / Poor]</span>
  </div>
</div>

Then include the extracted document content below."""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5050",
        "X-Title": "Document Analyzer"
    }

    payload = {
        "model": VISION_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{enhanced_base64}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{gradient_base64}"
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ],
        "max_tokens": 4096,
        "temperature": 0.1,
        "provider": {
            "require_parameters": True
        },
        "reasoning": {
            "effort": "none"
        }
    }

    try:
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        return {
            "success": True,
            "content": result.get("choices", [{}])[0].get("message", {}).get("content", ""),
            "model": VISION_MODEL,
            "usage": result.get("usage", {})
        }
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def extract_document_entities(html_content):
    """
    Extract business entities from document HTML using LLM with confidence scoring.
    """
    prompt = f"""You are a precision document extraction system for accounting purposes.
Extract ONLY what is visible in the HTML, with accurate confidence scoring.

### DOCUMENT HTML:
\"\"\"
{html_content}
\"\"\"

### EXTRACTION RULES:

1. CONFIDENCE SCORING (0.0 - 1.0):
   - 0.9-1.0: Crystal clear, labeled field
   - 0.7-0.8: Clear text, standard position
   - 0.5-0.6: Readable but some uncertainty
   - 0.3-0.4: Derived/inferred value (max 0.75 for derived)
   - Below 0.3: Use null instead

2. DIRECT vs DERIVED:
   - DIRECT (is_inferred=false): Text with explicit label like "Supplier:", "Invoice #:"
   - DERIVED (is_inferred=true): Deduced from context (footer text, email domain, calculations)

3. COMMON ERRORS TO AVOID:
   - Store ID/Address is NOT supplier name ("STORE# 5793" → reject, look in footer)
   - Cardholder is NOT customer (name near VISA/card → ignore for customer)
   - Never complete partial words ("SWINDO" stays "SWINDO")

4. NULL RULES:
   - No evidence exists → null
   - Text unreadable → null + warning
   - Evidence exists but readable → extract with low confidence

### OUTPUT FORMAT (JSON only, no markdown):

{{
  "document_analysis": {{
    "type": "invoice|receipt|credit_note|expense_receipt|unknown",
    "type_confidence": 0.0-1.0,
    "overall_quality": "good|fair|poor"
  }},
  "extracted_fields": {{
    "supplier": {{
      "value": "string or null",
      "confidence": 0.0-1.0,
      "is_inferred": false,
      "source_text": "exact text found",
      "inference_reason": "only if is_inferred=true"
    }},
    "customer": {{
      "value": "string or null",
      "confidence": 0.0-1.0,
      "is_inferred": false,
      "source_text": "exact text found",
      "inference_reason": "only if is_inferred=true"
    }},
    "supplier_address": {{
      "value": "string with newlines or null",
      "confidence": 0.0-1.0,
      "is_inferred": false
    }},
    "supplier_vat_id": {{
      "value": "string or null",
      "confidence": 0.0-1.0,
      "is_inferred": false
    }},
    "customer_address": {{
      "value": "string with newlines or null",
      "confidence": 0.0-1.0,
      "is_inferred": false
    }},
    "invoice_number": {{
      "value": "string or null",
      "confidence": 0.0-1.0,
      "is_inferred": false,
      "source_text": "exact text found"
    }},
    "date": {{
      "value": "YYYY-MM-DD or null",
      "raw_value": "original format",
      "confidence": 0.0-1.0,
      "is_inferred": false
    }},
    "due_date": {{
      "value": "YYYY-MM-DD or null",
      "raw_value": "original format",
      "confidence": 0.0-1.0,
      "is_inferred": false
    }},
    "currency": {{
      "value": "GBP|USD|EUR or null",
      "confidence": 0.0-1.0,
      "is_inferred": true,
      "inference_reason": "derived from symbol"
    }},
    "subtotal": {{
      "value": 0.00,
      "confidence": 0.0-1.0,
      "is_inferred": false,
      "source_text": "exact text"
    }},
    "tax_amount": {{
      "value": 0.00,
      "confidence": 0.0-1.0,
      "is_inferred": false,
      "source_text": "exact text"
    }},
    "tax_rate": {{
      "value": "20%",
      "confidence": 0.0-1.0,
      "is_inferred": true,
      "inference_reason": "calculated from tax/subtotal"
    }},
    "total": {{
      "value": 0.00,
      "confidence": 0.0-1.0,
      "is_inferred": false,
      "source_text": "exact text"
    }},
    "payment_terms": {{
      "value": "string or null",
      "confidence": 0.0-1.0,
      "is_inferred": false
    }}
  }},
  "line_items": {{
    "items": [
      {{
        "description": "string",
        "quantity": 0,
        "unit_price": 0.00,
        "amount": 0.00,
        "confidence": 0.0-1.0
      }}
    ],
    "count": 0,
    "confidence": 0.0-1.0
  }},
  "validation": {{
    "amounts_reconcile": true|false,
    "reconciliation_detail": "subtotal + tax = total check",
    "warnings": ["list of issues, low confidence fields, truncated text"]
  }}
}}

CRITICAL:
- Derived values (from footer, email, calculations) max confidence = 0.75
- If no "Bill To"/"Customer" section exists, customer = null (don't use cardholder)
- Include inference_reason for ALL is_inferred=true fields"""

    if not OPENROUTER_API_KEY:
        return {"error": "OpenRouter API key not configured"}

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5050",
        "X-Title": "Document Analyzer"
    }

    payload = {
        "model": EXTRACTION_MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a JSON extraction API. You MUST return ONLY valid JSON in the EXACT format specified. No markdown, no explanations, no deviations from the schema."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 4096,
        "temperature": 0,
        "response_format": {"type": "json_object"}
    }

    try:
        response = requests.post(OPENROUTER_BASE_URL, headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Try to parse as JSON
        try:
            # Clean up the response - remove markdown code blocks if present
            content_clean = content.strip()
            if content_clean.startswith("```json"):
                content_clean = content_clean[7:]
            if content_clean.startswith("```"):
                content_clean = content_clean[3:]
            if content_clean.endswith("```"):
                content_clean = content_clean[:-3]
            content_clean = content_clean.strip()

            extracted_data = json.loads(content_clean)
            # Normalize to ensure consistent field structure with value, confidence, isInferred
            normalized_data = normalize_entity_extraction_result(extracted_data)
            return {
                "success": True,
                "extracted_data": normalized_data,
                "raw_response": content
            }
        except json.JSONDecodeError:
            return {
                "success": True,
                "extracted_data": None,
                "raw_response": content,
                "parse_error": "Could not parse JSON from response"
            }

    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": f"Unexpected error: {str(e)}"}


def detect_contours(image):
    """
    Detect and visualize contours in the image.
    Returns the image with contours drawn and contour statistics.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on a copy of the original image
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # Calculate statistics
    stats = {
        'total_contours': len(contours),
        'large_contours': sum(1 for c in contours if cv2.contourArea(c) > 1000),
        'small_contours': sum(1 for c in contours if cv2.contourArea(c) <= 1000)
    }

    return result, edges, stats


def detect_document_edges(image):
    """
    Detect document boundaries and check if document is properly aligned.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 75, 200)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

    result = image.copy()
    document_contour = None

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        # If the contour has 4 points, it's likely a document
        if len(approx) == 4:
            document_contour = approx
            cv2.drawContours(result, [approx], -1, (0, 255, 0), 3)
            break

    return result, document_contour is not None


def detect_folds(image):
    """
    Detect folds and creases in a document.
    Folds typically appear as lines with shadow patterns.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to detect variations
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

    # Detect lines using Hough Transform
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Probabilistic Hough Line Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100,
                            minLineLength=100, maxLineGap=10)

    result = image.copy()
    fold_lines = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            # Filter for significant lines (potential folds)
            if length > image.shape[1] * 0.3:  # At least 30% of image width
                cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
                fold_lines.append({
                    'start': (x1, y1),
                    'end': (x2, y2),
                    'length': length
                })

    # Analyze intensity variations for fold detection
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = np.uint8(255 * magnitude / magnitude.max()) if magnitude.max() > 0 else np.zeros_like(gray)

    # Detect sharp intensity changes (folds create shadows)
    _, intensity_thresh = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

    fold_analysis = {
        'potential_folds_detected': len(fold_lines),
        'fold_lines': fold_lines,
        'has_significant_folds': len(fold_lines) > 0,
        'fold_severity': 'High' if len(fold_lines) > 3 else 'Medium' if len(fold_lines) > 0 else 'None'
    }

    return result, magnitude, fold_analysis


def detect_overlap(image):
    """
    Detect if multiple documents are overlapping in the image.
    Looks for multiple document boundaries and shadow patterns.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    result = image.copy()
    document_contours = []

    # Filter for rectangular contours (potential documents)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:  # Minimum area threshold
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

            if len(approx) >= 4 and len(approx) <= 6:
                document_contours.append({
                    'contour': approx,
                    'area': area,
                    'bounds': cv2.boundingRect(contour)
                })

    # Check for overlapping bounding boxes
    overlaps = []
    for i, doc1 in enumerate(document_contours):
        for j, doc2 in enumerate(document_contours[i+1:], i+1):
            x1, y1, w1, h1 = doc1['bounds']
            x2, y2, w2, h2 = doc2['bounds']

            # Check if rectangles overlap
            if (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2):
                overlaps.append((i, j))

    # Draw detected document regions
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    for idx, doc in enumerate(document_contours[:5]):
        color = colors[idx % len(colors)]
        cv2.drawContours(result, [doc['contour']], -1, color, 2)
        x, y, w, h = doc['bounds']
        cv2.rectangle(result, (x, y), (x+w, y+h), color, 1)

    # Analyze shadow patterns (overlapping documents create shadows)
    shadow_analysis = analyze_shadows(image)

    overlap_analysis = {
        'document_regions_detected': len(document_contours),
        'overlapping_pairs': len(overlaps),
        'has_overlap': len(overlaps) > 0 or shadow_analysis['has_shadow_edges'],
        'overlap_confidence': 'High' if len(overlaps) > 0 else 'Medium' if shadow_analysis['has_shadow_edges'] else 'Low'
    }

    return result, overlap_analysis


def analyze_shadows(image):
    """
    Analyze shadow patterns that might indicate overlapping documents.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Look for shadow edges (dark regions with sharp boundaries)
    _, dark_regions = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    # Find contours of dark regions
    contours, _ = cv2.findContours(dark_regions, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    significant_shadows = [c for c in contours if cv2.contourArea(c) > 1000]

    return {
        'shadow_regions': len(significant_shadows),
        'has_shadow_edges': len(significant_shadows) > 2
    }


def detect_hand(image):
    """
    Detect if a human hand is covering part of the document.
    Uses skin color detection in HSV and YCrCb color spaces.
    """
    result = image.copy()
    image_area = image.shape[0] * image.shape[1]

    # Minimum area for hand detection (at least 2% of image)
    min_hand_area = image_area * 0.02

    # Convert to different color spaces for skin detection
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

    # HSV skin color range - more strict saturation to avoid paper colors
    hsv_lower = np.array([0, 40, 80], dtype=np.uint8)
    hsv_upper = np.array([18, 255, 255], dtype=np.uint8)
    hsv_mask1 = cv2.inRange(hsv, hsv_lower, hsv_upper)

    # Extended HSV range for different skin tones
    hsv_lower2 = np.array([170, 40, 80], dtype=np.uint8)
    hsv_upper2 = np.array([180, 255, 255], dtype=np.uint8)
    hsv_mask2 = cv2.inRange(hsv, hsv_lower2, hsv_upper2)

    hsv_mask = cv2.bitwise_or(hsv_mask1, hsv_mask2)

    # YCrCb skin color range - tighter range
    ycrcb_lower = np.array([0, 138, 80], dtype=np.uint8)
    ycrcb_upper = np.array([255, 170, 125], dtype=np.uint8)
    ycrcb_mask = cv2.inRange(ycrcb, ycrcb_lower, ycrcb_upper)

    # Combine masks - both must match
    skin_mask = cv2.bitwise_and(hsv_mask, ycrcb_mask)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    skin_mask = cv2.erode(skin_mask, kernel, iterations=2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

    # Find contours in skin mask
    contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    hand_regions = []
    total_skin_area = 0

    for contour in contours:
        area = cv2.contourArea(contour)
        # Filter by area - must be significant portion of image
        if area > min_hand_area:
            # Check convexity defects (hands have fingers creating defects)
            hull = cv2.convexHull(contour, returnPoints=False)

            defect_count = 0
            is_likely_hand = False

            if len(hull) > 3 and len(contour) > 3:
                try:
                    defects = cv2.convexityDefects(contour, hull)

                    if defects is not None:
                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            # Filter significant defects (finger gaps)
                            if d > 8000:
                                defect_count += 1

                    # Hands need at least 2 finger gaps to be detected
                    is_likely_hand = defect_count >= 2
                except:
                    is_likely_hand = False
                    defect_count = 0

            # Calculate bounding box
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0

            # Hand-like aspect ratio check (0.4 to 1.8)
            hand_like_shape = 0.4 < aspect_ratio < 1.8

            # Only consider as hand if has finger defects AND good shape
            is_likely_hand = is_likely_hand and hand_like_shape

            if is_likely_hand:
                total_skin_area += area
                hand_regions.append({
                    'bounds': (x, y, w, h),
                    'area': area,
                    'is_likely_hand': is_likely_hand,
                    'defects': defect_count
                })

                # Draw detection
                color = (0, 0, 255)
                cv2.drawContours(result, [contour], -1, color, 2)
                cv2.rectangle(result, (x, y), (x+w, y+h), color, 2)
                cv2.putText(result, "Hand", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,
                           0.7, color, 2)

    # Calculate coverage percentage
    coverage_percent = (total_skin_area / image_area) * 100 if image_area > 0 else 0

    # Create skin mask visualization
    skin_visualization = cv2.bitwise_and(image, image, mask=skin_mask)

    hand_analysis = {
        'skin_regions_detected': len(hand_regions),
        'likely_hands': len(hand_regions),
        'hand_detected': len(hand_regions) > 0,
        'coverage_percent': round(coverage_percent, 2),
        'is_obstructed': len(hand_regions) > 0,
        'obstruction_level': 'High' if coverage_percent > 10 else 'Medium' if len(hand_regions) > 0 else 'Low'
    }

    return result, skin_visualization, hand_analysis


def detect_distortion(image):
    """
    Detect if the document/working region is distorted.
    Checks for perspective distortion, warping, and irregular edges.
    """
    result = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # Find document contour
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    document_contour = None
    document_corners = None

    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

        if len(approx) == 4:
            document_contour = contour
            document_corners = approx.reshape(4, 2)
            break

    distortion_analysis = {
        'document_found': document_contour is not None,
        'is_distorted': False,
        'distortion_type': [],
        'distortion_score': 0,
        'perspective_skew': 0,
        'edge_irregularity': 0
    }

    if document_contour is not None:
        # Draw detected corners
        for corner in document_corners:
            cv2.circle(result, tuple(corner.astype(int)), 10, (0, 255, 0), -1)
        cv2.drawContours(result, [document_corners.astype(int)], -1, (0, 255, 0), 3)

        # Calculate edge lengths
        edges_lengths = []
        for i in range(4):
            p1 = document_corners[i]
            p2 = document_corners[(i + 1) % 4]
            length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            edges_lengths.append(length)

        # Check for perspective distortion (opposite edges should be similar)
        edge_ratio_1 = min(edges_lengths[0], edges_lengths[2]) / max(edges_lengths[0], edges_lengths[2]) if max(edges_lengths[0], edges_lengths[2]) > 0 else 1
        edge_ratio_2 = min(edges_lengths[1], edges_lengths[3]) / max(edges_lengths[1], edges_lengths[3]) if max(edges_lengths[1], edges_lengths[3]) > 0 else 1

        perspective_score = (edge_ratio_1 + edge_ratio_2) / 2
        perspective_skew = (1 - perspective_score) * 100

        # Check angles at corners (should be close to 90 degrees)
        angles = []
        for i in range(4):
            p1 = document_corners[(i - 1) % 4]
            p2 = document_corners[i]
            p3 = document_corners[(i + 1) % 4]

            v1 = p1 - p2
            v2 = p3 - p2

            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))
            angles.append(angle)

        angle_deviation = sum(abs(a - 90) for a in angles) / 4

        # Check edge straightness using the original contour
        straightness_scores = []
        hull = cv2.convexHull(document_contour)
        hull_area = cv2.contourArea(hull)
        contour_area = cv2.contourArea(document_contour)

        solidity = contour_area / hull_area if hull_area > 0 else 1
        edge_irregularity = (1 - solidity) * 100

        # Calculate overall distortion score
        distortion_score = (perspective_skew * 0.4 + angle_deviation * 0.4 + edge_irregularity * 0.2)

        # Determine distortion types
        distortion_types = []
        if perspective_skew > 10:
            distortion_types.append('Perspective')
        if angle_deviation > 15:
            distortion_types.append('Rotation/Tilt')
        if edge_irregularity > 5:
            distortion_types.append('Warping/Curving')

        distortion_analysis.update({
            'is_distorted': distortion_score > 15,
            'distortion_type': distortion_types,
            'distortion_score': round(distortion_score, 2),
            'perspective_skew': round(perspective_skew, 2),
            'angle_deviation': round(angle_deviation, 2),
            'edge_irregularity': round(edge_irregularity, 2),
            'distortion_level': 'High' if distortion_score > 30 else 'Medium' if distortion_score > 15 else 'Low',
            'corner_angles': [round(a, 1) for a in angles]
        })

        # Add visual indicators
        if distortion_score > 15:
            cv2.putText(result, f"Distortion: {distortion_analysis['distortion_level']}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        # No clear document boundary found - might indicate severe distortion
        distortion_analysis['distortion_type'].append('No clear boundary')
        distortion_analysis['is_distorted'] = True
        distortion_analysis['distortion_level'] = 'Unknown'

    # Create edge analysis visualization
    edge_viz = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return result, edge_viz, distortion_analysis


def comprehensive_analysis(image):
    """
    Perform comprehensive document analysis.
    """
    start_time = time.time()

    # Original image info
    orig_height, orig_width = image.shape[:2]

    # Skip extraction/cropping - use original image directly to prevent data loss
    # Previously: extracted_img, extraction_success, extraction_bounds = extract_white_document(image)
    extracted_img = image
    extraction_success = False
    extraction_bounds = None

    # Use original image for all processing (no cropping)
    processing_image = image
    height, width = processing_image.shape[:2]

    # Contour analysis
    contour_img, edges, contour_stats = detect_contours(processing_image)

    # Document edge detection
    doc_edge_img, has_document = detect_document_edges(processing_image)

    # Fold detection
    fold_img, gradient_img, fold_analysis = detect_folds(processing_image)

    # Overlap detection
    overlap_img, overlap_analysis = detect_overlap(processing_image)

    # Hand/obstruction detection
    hand_img, skin_mask_img, hand_analysis = detect_hand(processing_image)

    # Distortion detection
    distortion_img, distortion_edges, distortion_analysis = detect_distortion(processing_image)

    # Skip cropping - apply enhancement directly to processing image
    # This ensures no information is lost due to over-cropping
    enhanced_img, binary_img = auto_enhance_document(processing_image)

    # Calculate processing time
    processing_time = round(time.time() - start_time, 2)

    return {
        'original': image_to_base64(image),
        'extracted': {
            'image': image_to_base64(extracted_img),
            'success': extraction_success,
            'bounds': extraction_bounds
        },
        'contours': {
            'image': image_to_base64(contour_img),
            'edges': image_to_base64(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)),
            'stats': contour_stats
        },
        'document': {
            'image': image_to_base64(doc_edge_img),
            'detected': has_document
        },
        'folds': {
            'image': image_to_base64(fold_img),
            'gradient': image_to_base64(cv2.cvtColor(gradient_img, cv2.COLOR_GRAY2BGR)),
            'analysis': fold_analysis
        },
        'overlap': {
            'image': image_to_base64(overlap_img),
            'analysis': overlap_analysis
        },
        'hand': {
            'image': image_to_base64(hand_img),
            'skin_mask': image_to_base64(skin_mask_img),
            'analysis': hand_analysis
        },
        'distortion': {
            'image': image_to_base64(distortion_img),
            'edges': image_to_base64(distortion_edges),
            'analysis': distortion_analysis
        },
        'crop': {
            'enhanced': image_to_base64(enhanced_img),
            'binary': image_to_base64(binary_img),
            'info': {
                'crop_successful': False,
                'crop_skipped': True,
                'reason': 'Cropping disabled to prevent data loss'
            }
        },
        'image_info': {
            'original_width': orig_width,
            'original_height': orig_height,
            'processed_width': width,
            'processed_height': height,
            'aspect_ratio': round(width/height, 2),
            'white_region_extracted': extraction_success
        },
        'timing': {
            'processing_seconds': processing_time
        }
    }


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if file and allowed_file(file.filename):
        # Clear previous images before saving new one
        clear_upload_folder()

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Read and process image
        image = cv2.imread(filepath)
        if image is None:
            return jsonify({'error': 'Could not read image'}), 400

        # Perform analysis
        results = comprehensive_analysis(image)

        # Convert numpy types to Python native types for JSON serialization
        results = to_python_types(results)

        return jsonify(results)

    return jsonify({'error': 'Invalid file type'}), 400


@app.route('/extract-legacy', methods=['POST'])
def extract_legacy():
    """
    Legacy extraction endpoint - Direct vision model extraction.
    Simple, single-call approach: Image → Vision Model → Structured JSON

    Accepts:
      - JSON with 'image' (base64) and optional 'model' (gpt-4o, qwen-vl, gemini)
      - OR file upload (multipart/form-data) with optional 'model' field

    Returns structured extraction with classification, entities, and summary.
    """
    image_base64 = None
    model_key = "gpt-4o"  # Default model

    # Handle JSON input
    if request.is_json:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({'error': 'No image data provided. Send JSON with "image" (base64) field.'}), 400
        image_base64 = data['image']
        model_key = data.get('model', 'gpt-4o')

    # Handle file upload
    elif 'file' in request.files:
        file = request.files['file']
        model_key = request.form.get('model', 'gpt-4o')

        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        if file and allowed_file(file.filename):
            # Clear previous uploads
            clear_upload_folder()

            # Save and read file
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Convert to base64
            with open(filepath, 'rb') as f:
                image_base64 = base64.b64encode(f.read()).decode('utf-8')
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    else:
        return jsonify({'error': 'No image provided. Send JSON with "image" field or file upload.'}), 400

    # Validate model key
    if model_key not in LEGACY_MODELS:
        return jsonify({'error': f'Invalid model. Choose from: {", ".join(LEGACY_MODELS.keys())}'}), 400

    # Call legacy extraction with selected model
    result = legacy_extract(image_base64, model_key)

    if 'error' in result:
        return jsonify(result), 500

    return jsonify({
        'success': True,
        'method': f'legacy_{model_key}',
        'model': result.get('model'),
        'processing_time_ms': result.get('processing_time_ms'),
        'extraction': result.get('parsed'),
        'raw_response': result.get('raw_response'),
        'usage': result.get('usage')
    })


@app.route('/analyze-llm', methods=['POST'])
def analyze_with_llm():
    """
    Endpoint to analyze document with LLM.
    Expects JSON with:
      - 'image' (base64): Original uncropped image (required)
      - 'binary_image' (base64): Binary/thresholded version for text clarity (optional)
      - 'enhanced_image' (base64): Enhanced cropped version when crop quality is good (optional)
      - 'crop_quality': Quality of the crop ('good', 'fair', 'poor', 'minimal')
      - 'crop_warning': Whether there's a crop warning
    Returns the images sent, HTML conversion, and extracted entities.
    """
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    original_base64 = data['image']  # Original uncropped image
    binary_base64 = data.get('binary_image')  # Binary for text clarity
    enhanced_base64 = data.get('enhanced_image')  # Enhanced cropped (when crop is good)
    crop_quality = data.get('crop_quality', 'unknown')
    has_crop_warning = data.get('crop_warning', False)

    # Check if API key is configured
    if not OPENROUTER_API_KEY:
        return jsonify({
            'error': 'OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.',
            'image_sent': original_base64[:100] + '...'  # Show truncated for reference
        }), 400

    # Start timing
    start_time = time.time()

    # Step 1: Convert image(s) to HTML
    # Send original + binary, and optionally enhanced if crop quality is good
    html_start = time.time()
    html_result = convert_image_to_html(original_base64, binary_base64, enhanced_base64)
    html_time = round(time.time() - html_start, 2)

    if 'error' in html_result:
        return jsonify({
            'error': html_result['error'],
            'image_sent': original_base64,
            'step_failed': 'html_conversion'
        }), 500

    html_content = html_result.get('content', '')

    # Step 2: Extract entities from HTML
    extraction_start = time.time()
    extraction_result = extract_document_entities(html_content)
    extraction_time = round(time.time() - extraction_start, 2)

    total_time = round(time.time() - start_time, 2)

    # Determine mode based on images sent
    num_images = 1 + (1 if binary_base64 else 0) + (1 if enhanced_base64 else 0)
    if num_images == 3:
        mode = 'triple_image'
    elif num_images == 2:
        mode = 'dual_image'
    else:
        mode = 'single_image'

    response = {
        'success': True,
        'images_sent': {
            'original': original_base64,
            'binary': binary_base64,
            'enhanced': enhanced_base64
        },
        'num_images': num_images,
        'crop_quality': crop_quality,
        'has_crop_warning': has_crop_warning,
        'html_conversion': {
            'html': html_content,
            'model': html_result.get('model', 'unknown'),
            'usage': html_result.get('usage', {}),
            'mode': mode
        },
        'entity_extraction': extraction_result,
        'timing': {
            'html_conversion_seconds': html_time,
            'entity_extraction_seconds': extraction_time,
            'total_seconds': total_time
        }
    }

    return jsonify(response)


@app.route('/analyze-llm-advanced', methods=['POST'])
def analyze_with_llm_advanced():
    """
    Advanced endpoint to analyze document with LLM using multiple images.
    Expects JSON with 'enhanced_image' and 'gradient_image' (base64) fields.
    Returns the images sent, HTML conversion, and extracted entities.
    """
    data = request.get_json()

    if not data or 'enhanced_image' not in data or 'gradient_image' not in data:
        return jsonify({'error': 'Both enhanced_image and gradient_image are required'}), 400

    enhanced_base64 = data['enhanced_image']
    gradient_base64 = data['gradient_image']

    # Check if API key is configured
    if not OPENROUTER_API_KEY:
        return jsonify({
            'error': 'OpenRouter API key not configured. Set OPENROUTER_API_KEY environment variable.',
        }), 400

    # Start timing
    start_time = time.time()

    # Step 1: Convert images to HTML using advanced method
    html_start = time.time()
    html_result = convert_image_to_html_advanced(enhanced_base64, gradient_base64)
    html_time = round(time.time() - html_start, 2)

    if 'error' in html_result:
        return jsonify({
            'error': html_result['error'],
            'image_sent': enhanced_base64,
            'step_failed': 'html_conversion'
        }), 500

    html_content = html_result.get('content', '')

    # Step 2: Extract entities from HTML
    extraction_start = time.time()
    extraction_result = extract_document_entities(html_content)
    extraction_time = round(time.time() - extraction_start, 2)

    total_time = round(time.time() - start_time, 2)

    response = {
        'success': True,
        'image_sent': enhanced_base64,  # Show enhanced image in results
        'html_conversion': {
            'html': html_content,
            'model': html_result.get('model', 'unknown'),
            'usage': html_result.get('usage', {}),
            'mode': 'advanced'
        },
        'entity_extraction': extraction_result,
        'timing': {
            'html_conversion_seconds': html_time,
            'entity_extraction_seconds': extraction_time,
            'total_seconds': total_time
        }
    }

    return jsonify(response)


@app.route('/set-api-key', methods=['POST'])
def set_api_key():
    """
    Endpoint to set OpenRouter API key at runtime.
    """
    global OPENROUTER_API_KEY
    data = request.get_json()

    if not data or 'api_key' not in data:
        return jsonify({'error': 'No API key provided'}), 400

    OPENROUTER_API_KEY = data['api_key']
    return jsonify({'success': True, 'message': 'API key configured'})


@app.route('/check-api-key', methods=['GET'])
def check_api_key():
    """
    Check if API key is configured.
    """
    return jsonify({
        'configured': bool(OPENROUTER_API_KEY),
        'key_preview': OPENROUTER_API_KEY[:10] + '...' if OPENROUTER_API_KEY else None
    })


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# =============================================================================
# UNIFIED DOCUMENT EXTRACTION PIPELINE
# Flow: Upload → Preprocessing → Classify (dual images) → Type-specific Extract
# =============================================================================

@app.route('/extract', methods=['POST'])
def unified_extract():
    """
    Unified Document Extraction Pipeline.

    Flow:
    1. Image uploaded
    2. Image processing (preprocessing)
    3. Original + Binary sent to LLM for classification
    4. LLM classifies category (thermal_print, a4_invoice, handprint, pos_receipt)
    5. Route to type-specific extractor

    Accepts: file upload (multipart/form-data)
    Returns: Classification + Extracted data
    """
    from datalabs.unified_pipeline import process_document
    import time

    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'success': False, 'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read and preprocess image
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'success': False, 'error': 'Could not read image'}), 400

    # Run preprocessing to get original and binary images
    preprocessing_results = comprehensive_analysis(image)

    # Get original and binary images
    original_base64 = preprocessing_results['original']
    binary_base64 = preprocessing_results['crop']['binary']

    try:
        # Run unified pipeline
        result = process_document(
            original_base64=original_base64,
            binary_base64=binary_base64
        )

        preprocessing_time = int((time.time() - start_time) * 1000) - result.get('total_time_ms', 0)

        return jsonify({
            'success': result['status'] in ['success', 'partial'],
            **result,
            'filename': filename,
            'preprocessing_time_ms': preprocessing_time,
            'preprocessing': {
                'crop_success': preprocessing_results['crop']['info']['crop_successful'],
                'extraction_success': preprocessing_results['extracted']['success']
            }
        })

    except Exception as e:
        logger.error(f"Unified extraction error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# DATALABS ROUTES - Document Extraction Pipeline (Legacy)
# =============================================================================

@app.route('/datalabs/health', methods=['GET'])
def datalabs_health():
    """
    Check datalabs configuration status.
    Returns whether API keys are configured.
    """
    return jsonify({
        'status': 'healthy',
        'datalab_configured': bool(datalabs_settings.DATALAB_API_KEY and datalabs_settings.DATALAB_API_KEY != 'your-datalab-api-key-here'),
        'llm_configured': bool(datalabs_settings.OPENROUTER_API_KEY),
        'extraction_model': datalabs_settings.EXTRACTION_MODEL,
        'datalab_base_url': datalabs_settings.DATALAB_BASE_URL
    })


@app.route('/datalabs/extract', methods=['POST'])
def datalabs_extract():
    """
    Full datalabs pipeline: OCR -> Extraction -> Mapping

    Accepts:
        - file: File upload (multipart/form-data)
        OR
        - JSON with 'image' (base64) and 'filename'

    Returns:
        Full extraction result with confidence scores and timing
    """
    # Handle file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        file_content = file.read()

    # Handle JSON with base64 image
    elif request.is_json:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400

        try:
            file_content = base64.b64decode(data['image'])
            filename = data.get('filename', 'image.png')
        except Exception as e:
            return jsonify({'error': f'Invalid base64 image: {str(e)}'}), 400
    else:
        return jsonify({'error': 'No file or image data provided'}), 400

    # Determine MIME type
    ext = os.path.splitext(filename)[1].lower()
    mime_map = {
        '.pdf': 'application/pdf',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.tiff': 'image/tiff',
        '.bmp': 'image/bmp',
        '.webp': 'image/webp'
    }
    mime_type = mime_map.get(ext, 'application/octet-stream')

    # Run datalabs pipeline
    try:
        result = process_document(
            file_content=file_content,
            filename=filename,
            mime_type=mime_type,
            debug_mode=True
        )

        # Format response
        response = {
            'success': result.get('status') == ProcessingStatus.COMPLETE.value,
            'status': result.get('status'),
            'extraction': result.get('mapped_output'),
            'at_a_glance': result.get('at_a_glance'),
            'warnings': result.get('warnings', []),
            'timing': {
                'ocr_ms': result.get('ocr_time_ms', 0),
                'extraction_ms': result.get('extraction_time_ms', 0),
                'mapping_ms': result.get('mapping_time_ms', 0),
                'total_ms': result.get('total_time_ms', 0)
            }
        }

        if result.get('error'):
            response['error'] = result.get('error')
            response['error_stage'] = result.get('error_stage')

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/datalabs/extract-enhanced', methods=['POST'])
def datalabs_extract_enhanced():
    """
    Chain: imagePreprocessor enhancement -> datalabs pipeline

    1. Runs comprehensive_analysis() to enhance image
    2. Uses enhanced/cropped image for datalabs OCR
    3. Returns both preprocessing info and extraction result

    Query params:
        - use_image: 'enhanced' | 'cropped' | 'extracted' | 'original' (default: enhanced)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read and preprocess image
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Could not read image'}), 400

    # Run preprocessing
    preprocessing_results = comprehensive_analysis(image)
    preprocessing_results = to_python_types(preprocessing_results)

    # Get enhanced image for datalabs
    use_image_key = request.args.get('use_image', 'enhanced')

    if use_image_key == 'enhanced' and preprocessing_results['crop']['info']['crop_successful']:
        image_base64 = preprocessing_results['crop']['enhanced']
    elif use_image_key == 'cropped' and preprocessing_results['crop']['info']['crop_successful']:
        image_base64 = preprocessing_results['crop']['cropped']
    elif use_image_key == 'extracted' and preprocessing_results['extracted']['success']:
        image_base64 = preprocessing_results['extracted']['image']
    else:
        image_base64 = preprocessing_results['original']

    # Convert base64 back to bytes for datalabs
    file_content = base64.b64decode(image_base64)

    # Run datalabs pipeline on enhanced image
    try:
        result = process_document(
            file_content=file_content,
            filename=f"enhanced_{filename}",
            mime_type='image/png',
            debug_mode=True
        )

        response = {
            'success': result.get('status') == ProcessingStatus.COMPLETE.value,
            'status': result.get('status'),
            'preprocessing': {
                'document_detected': preprocessing_results['crop']['info']['document_detected'],
                'crop_successful': preprocessing_results['crop']['info']['crop_successful'],
                'image_used': use_image_key,
                'hand_detected': preprocessing_results['hand']['analysis']['hand_detected'],
                'distortion_level': preprocessing_results['distortion']['analysis'].get('distortion_level', 'Unknown'),
                'processing_seconds': preprocessing_results['timing']['processing_seconds']
            },
            'extraction': result.get('mapped_output'),
            'at_a_glance': result.get('at_a_glance'),
            'warnings': result.get('warnings', []),
            'timing': {
                'preprocessing_ms': int(preprocessing_results['timing']['processing_seconds'] * 1000),
                'ocr_ms': result.get('ocr_time_ms', 0),
                'extraction_ms': result.get('extraction_time_ms', 0),
                'mapping_ms': result.get('mapping_time_ms', 0),
                'total_ms': result.get('total_time_ms', 0)
            }
        }

        if result.get('error'):
            response['error'] = result.get('error')
            response['error_stage'] = result.get('error_stage')

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'preprocessing': {
                'document_detected': preprocessing_results['crop']['info']['document_detected'],
                'crop_successful': preprocessing_results['crop']['info']['crop_successful']
            }
        }), 500


@app.route('/datalabs/ocr', methods=['POST'])
def datalabs_ocr_only():
    """
    Datalabs OCR only - returns HTML content.
    Useful for debugging or when you only need OCR.
    """
    from datalabs.tools import ocr_tool

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    file_content = file.read()
    file_content_base64 = base64.b64encode(file_content).decode('utf-8')

    try:
        html_content = ocr_tool._run(
            file_content_base64=file_content_base64,
            filename=filename,
            output_format='html',
            force_ocr=request.args.get('force_ocr', 'false').lower() == 'true'
        )

        return jsonify({
            'success': True,
            'html': html_content,
            'filename': filename
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/vision/to-html', methods=['POST'])
def vision_to_html_route():
    """
    Convert document image to HTML using free Qwen vision model.
    Shows exactly what the model "sees" in the document.

    Accepts: file upload OR JSON with base64 image
    Returns: HTML representation of the document
    """
    from datalabs.tools import convert_image_to_html
    import time

    start_time = time.time()
    image_base64 = None
    filename = "document"

    # Handle file upload
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '':
            filename = secure_filename(file.filename)
            file_content = file.read()
            image_base64 = base64.b64encode(file_content).decode('utf-8')

    # Handle JSON with base64 image
    elif request.is_json:
        data = request.get_json()
        image_base64 = data.get('image') or data.get('image_base64')
        filename = data.get('filename', 'document')

    if not image_base64:
        return jsonify({'error': 'No image provided. Send file or JSON with image_base64'}), 400

    try:
        # Convert image to HTML
        result = convert_image_to_html(image_base64)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Add CSS styles for rendering
        default_styles = """
        <style>
            .document {
                font-family: 'Courier New', monospace;
                background: white;
                padding: 20px;
                border: 1px solid #ccc;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin: 0 auto;
            }
            .document.receipt {
                max-width: 320px;
                font-size: 12px;
            }
            .document.invoice {
                max-width: 800px;
                font-size: 14px;
            }
            .document header {
                text-align: center;
                margin-bottom: 15px;
            }
            .document table {
                width: 100%;
                border-collapse: collapse;
            }
            .document table td, .document table th {
                padding: 4px 8px;
                border-bottom: 1px dotted #ddd;
            }
            .document hr {
                border: none;
                border-top: 1px dashed #999;
                margin: 10px 0;
            }
            .document footer {
                text-align: center;
                margin-top: 15px;
                font-size: 0.9em;
                color: #666;
            }
            .document .unclear {
                background: #ffeb3b;
                padding: 0 4px;
                border-radius: 2px;
            }
            .document .confidence-low {
                color: #999;
                font-style: italic;
            }
            .document .total {
                font-weight: bold;
                font-size: 1.2em;
            }
            .document .row {
                display: flex;
                justify-content: space-between;
            }
        </style>
        """

        html_with_styles = default_styles + result.get('html', '')

        return jsonify({
            'success': True,
            'html': html_with_styles,
            'html_raw': result.get('html', ''),
            'model': result.get('model', 'qwen/qwen3-4b:free'),
            'filename': filename,
            'processing_time_ms': elapsed_ms,
            'char_count': result.get('char_count', 0)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'html': f'<div class="document error"><p style="color:red;">Error: {str(e)}</p></div>'
        }), 500


@app.route('/vision/to-html-enhanced', methods=['POST'])
def vision_to_html_enhanced():
    """
    Preprocess image, then convert to HTML using vision model.
    Uses enhanced/cropped image for better results.
    """
    from datalabs.tools import convert_image_to_html
    import time

    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read and preprocess image
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Could not read image'}), 400

    # Run preprocessing
    preprocessing_results = comprehensive_analysis(image)

    # Choose which image to use (enhanced is usually best)
    use_image = request.args.get('use_image', 'enhanced')

    if use_image == 'enhanced':
        image_base64 = preprocessing_results['crop']['enhanced']
    elif use_image == 'cropped':
        image_base64 = preprocessing_results['crop']['cropped']
    elif use_image == 'extracted':
        image_base64 = preprocessing_results['extracted']['image']
    else:
        image_base64 = preprocessing_results['original']

    try:
        # Convert to HTML
        result = convert_image_to_html(image_base64)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Add CSS styles
        default_styles = """
        <style>
            .document {
                font-family: 'Courier New', monospace;
                background: white;
                padding: 20px;
                border: 1px solid #ccc;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                margin: 0 auto;
            }
            .document.receipt { max-width: 320px; font-size: 12px; }
            .document.invoice { max-width: 800px; font-size: 14px; }
            .document header { text-align: center; margin-bottom: 15px; }
            .document table { width: 100%; border-collapse: collapse; }
            .document table td { padding: 4px 8px; border-bottom: 1px dotted #ddd; }
            .document hr { border: none; border-top: 1px dashed #999; margin: 10px 0; }
            .document footer { text-align: center; margin-top: 15px; color: #666; }
            .document .unclear { background: #ffeb3b; padding: 0 4px; }
            .document .confidence-low { color: #999; font-style: italic; }
        </style>
        """

        return jsonify({
            'success': True,
            'html': default_styles + result.get('html', ''),
            'html_raw': result.get('html', ''),
            'model': result.get('model', 'qwen/qwen3-4b:free'),
            'filename': filename,
            'image_used': use_image,
            'processing_time_ms': elapsed_ms,
            'preprocessing': {
                'extraction_success': preprocessing_results['extracted']['success'],
                'crop_success': preprocessing_results['crop']['info']['crop_successful']
            }
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# VISION CHAIN ROUTES - Image → Qwen VL (HTML) → LLM (Extraction)
# =============================================================================

@app.route('/vision/chain', methods=['POST'])
def vision_chain():
    """
    Vision Chain: Image → Qwen VL (HTML) → Gemini (Extraction)

    Two-step process:
    1. Qwen VL vision model converts image to HTML (what it sees)
    2. Gemini extracts structured data from HTML

    Accepts: JSON with base64 image or file upload
    """
    from datalabs.vision_chain import process_vision_chain

    # Get image from request
    if request.is_json:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        image_base64 = data['image']
        document_type = data.get('document_type', 'auto')
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        file_content = file.read()
        image_base64 = base64.b64encode(file_content).decode('utf-8')
        document_type = request.args.get('document_type', 'auto')
    else:
        return jsonify({'error': 'No image or file provided'}), 400

    try:
        result = process_vision_chain(
            image_base64=image_base64,
            document_type=document_type
        )
        return jsonify({
            'success': result['status'] in ['success', 'partial'],
            **result
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/vision/chain-enhanced', methods=['POST'])
def vision_chain_enhanced():
    """
    Vision Chain with preprocessing: Preprocess → Qwen VL (HTML) → Gemini (Extraction)

    Uses enhanced/binary image from preprocessing for better OCR results.
    """
    from datalabs.vision_chain import process_vision_chain
    import time

    start_time = time.time()

    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read and preprocess image
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Could not read image'}), 400

    # Run preprocessing
    preprocessing_results = comprehensive_analysis(image)

    # Choose which image to use
    use_image = request.args.get('use_image', 'enhanced')
    document_type = request.args.get('document_type', 'auto')

    if use_image == 'enhanced':
        image_base64 = preprocessing_results['crop']['enhanced']
    elif use_image == 'binary':
        image_base64 = preprocessing_results['crop']['binary']
    elif use_image == 'cropped':
        image_base64 = preprocessing_results['crop']['cropped']
    elif use_image == 'extracted':
        image_base64 = preprocessing_results['extracted']['image']
    else:
        image_base64 = preprocessing_results['original']

    try:
        result = process_vision_chain(
            image_base64=image_base64,
            document_type=document_type
        )

        preprocessing_time = int((time.time() - start_time) * 1000) - result.get('total_time_ms', 0)

        return jsonify({
            'success': result['status'] in ['success', 'partial'],
            **result,
            'filename': filename,
            'image_used': use_image,
            'preprocessing_time_ms': preprocessing_time,
            'preprocessing': {
                'extraction_success': preprocessing_results['extracted']['success'],
                'crop_success': preprocessing_results['crop']['info']['crop_successful']
            }
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


# =============================================================================
# SMART PIPELINE ROUTES - Classification + Type-Specific Extraction
# =============================================================================

@app.route('/smart/classify', methods=['POST'])
def smart_classify():
    """
    Classify document type using vision LLM.

    Returns document type (thermal_receipt, a4_invoice, handwritten, etc.)
    with confidence score and characteristics.
    """
    # Get image from request
    if request.is_json:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        image_base64 = data['image']
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        file_content = file.read()
        image_base64 = base64.b64encode(file_content).decode('utf-8')
    else:
        return jsonify({'error': 'No image or file provided'}), 400

    try:
        result = classify_document(image_base64=image_base64)
        return jsonify(result)
    except Exception as e:
        return jsonify({
            'status': 'failed',
            'error': str(e)
        }), 500


@app.route('/smart/extract', methods=['POST'])
def smart_extract():
    """
    Full smart pipeline: Classify -> Extract with type-specific prompt.

    Uses vision LLM directly on image (no OCR API needed).
    Different prompts for thermal receipts, A4 invoices, handwritten, etc.

    Accepts:
        - JSON with 'image' (base64)
        - file upload
    """
    # Get image from request
    if request.is_json:
        data = request.get_json()
        if 'image' not in data:
            return jsonify({'error': 'No image data provided'}), 400
        image_base64 = data['image']
        filename = data.get('filename', 'document')
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        filename = secure_filename(file.filename)
        file_content = file.read()
        image_base64 = base64.b64encode(file_content).decode('utf-8')
    else:
        return jsonify({'error': 'No image or file provided'}), 400

    try:
        result = process_document_smart(
            image_base64=image_base64,
            filename=filename
        )

        # Format response
        response = {
            'success': result.get('status') == 'complete',
            'status': result.get('status'),
            'classification': {
                'document_type': result.get('document_type'),
                'confidence': result.get('classification_confidence'),
                'characteristics': result.get('characteristics'),
                'reasoning': result.get('classification_reasoning')
            },
            'extraction': result.get('extracted_data'),
            'timing': {
                'classification_ms': result.get('classification_time_ms', 0),
                'extraction_ms': result.get('extraction_time_ms', 0),
                'total_ms': result.get('total_time_ms', 0)
            }
        }

        if result.get('error'):
            response['error'] = result.get('error')
            response['error_stage'] = result.get('error_stage')

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/smart/extract-enhanced', methods=['POST'])
def smart_extract_enhanced():
    """
    Preprocessing + Smart pipeline.

    1. Runs image preprocessing (crop, enhance)
    2. Classifies document type
    3. Extracts with type-specific prompt

    Query params:
        - use_image: 'enhanced' | 'cropped' | 'original' (default: enhanced)
    """
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Read and preprocess image
    image = cv2.imread(filepath)
    if image is None:
        return jsonify({'error': 'Could not read image'}), 400

    # Run preprocessing
    preprocessing_results = comprehensive_analysis(image)
    preprocessing_results = to_python_types(preprocessing_results)

    # Get enhanced image
    use_image_key = request.args.get('use_image', 'enhanced')

    if use_image_key == 'enhanced' and preprocessing_results['crop']['info']['crop_successful']:
        image_base64 = preprocessing_results['crop']['enhanced']
    elif use_image_key == 'cropped' and preprocessing_results['crop']['info']['crop_successful']:
        image_base64 = preprocessing_results['crop']['cropped']
    else:
        image_base64 = preprocessing_results['original']

    try:
        result = process_document_smart(
            image_base64=image_base64,
            filename=f"enhanced_{filename}"
        )

        response = {
            'success': result.get('status') == 'complete',
            'status': result.get('status'),
            'preprocessing': {
                'document_detected': preprocessing_results['crop']['info']['document_detected'],
                'crop_successful': preprocessing_results['crop']['info']['crop_successful'],
                'image_used': use_image_key,
                'hand_detected': preprocessing_results['hand']['analysis']['hand_detected'],
                'processing_seconds': preprocessing_results['timing']['processing_seconds']
            },
            'classification': {
                'document_type': result.get('document_type'),
                'confidence': result.get('classification_confidence'),
                'characteristics': result.get('characteristics'),
                'reasoning': result.get('classification_reasoning')
            },
            'extraction': result.get('extracted_data'),
            'timing': {
                'preprocessing_ms': int(preprocessing_results['timing']['processing_seconds'] * 1000),
                'classification_ms': result.get('classification_time_ms', 0),
                'extraction_ms': result.get('extraction_time_ms', 0),
                'total_ms': result.get('total_time_ms', 0)
            }
        }

        if result.get('error'):
            response['error'] = result.get('error')
            response['error_stage'] = result.get('error_stage')

        return jsonify(response)

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'preprocessing': {
                'document_detected': preprocessing_results['crop']['info']['document_detected'],
                'crop_successful': preprocessing_results['crop']['info']['crop_successful']
            }
        }), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=FLASK_DEBUG, port=FLASK_PORT)
