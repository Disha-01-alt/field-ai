"""
Image Preprocessing Module for Scanned Receipts.

Pipeline:
    Raw Image -> Resize (bounded) -> Grayscale -> Denoise (Bilateral / Gaussian)
              -> Illumination / Contrast Enhancement (CLAHE) -> Deskew -> (Optional Binarization)
"""

import cv2
import numpy as np
from typing import Tuple, Dict, Any, Optional
from pathlib import Path


def load_image(image_input: Any) -> np.ndarray:
    """Load image from filepath or return copy of numpy array."""
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input))
        if img is None:
            raise FileNotFoundError(f"Failed to load image from path: {image_input}")
        return img
    elif isinstance(image_input, np.ndarray):
        return image_input.copy()
    else:
        raise ValueError(f"Unsupported image input type: {type(image_input)}")


def resize_bounded(image: np.ndarray, max_dim: int = 2000, min_dim: int = 800) -> Tuple[np.ndarray, float]:
    """
    Scale image so longest side <= max_dim and shortest side >= min_dim.
    Returns (resized_image, scale_factor).
    """
    h, w = image.shape[:2]
    scale = 1.0

    # If image is too small, upscale to improve OCR detection
    if min(h, w) < min_dim:
        scale = min_dim / float(min(h, w))
    # If image is excessively large, downscale to save memory and processing time
    elif max(h, w) > max_dim:
        scale = max_dim / float(max(h, w))

    if scale != 1.0:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA)
        return resized, scale
    return image, 1.0


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert BGR/RGB image to grayscale."""
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def denoise(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply edge-preserving bilateral filtering to reduce grain without blurring characters.
    """
    return cv2.bilateralFilter(gray_image, d=9, sigmaColor=75, sigmaSpace=75)


def enhance_contrast(gray_image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to handle uneven lighting, faded thermal ink, and shadows.
    """
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray_image)


def estimate_skew_angle(gray_image: np.ndarray) -> float:
    """
    Estimate skew angle in degrees using Hough line transform on edge map.
    Returns angle in degrees (negative = counterclockwise, positive = clockwise).
    """
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=80, maxLineGap=10)

    if lines is None or len(lines) == 0:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx = x2 - x1
        dy = y2 - y1
        if dx == 0:
            continue
        angle = np.degrees(np.arctan2(dy, dx))
        # Receipts are roughly horizontal; only consider near-horizontal lines (-35 to +35 degrees)
        if -35.0 <= angle <= 35.0:
            angles.append(angle)

    if not angles:
        return 0.0

    # Use median angle to reject outliers
    median_angle = float(np.median(angles))
    return median_angle


def rotate_image(image: np.ndarray, angle: float, background_color: Tuple[int, int, int] = (255, 255, 255)) -> np.ndarray:
    """
    Rotate image by specified angle in degrees around its center, expanding borders so nothing is clipped.
    """
    if abs(angle) < 0.3:
        return image

    h, w = image.shape[:2]
    cx, cy = w // 2, h // 2

    m = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    cos = np.abs(m[0, 0])
    sin = np.abs(m[0, 1])

    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    m[0, 2] += (new_w / 2) - cx
    m[1, 2] += (new_h / 2) - cy

    border_val = 255 if len(image.shape) == 2 else background_color
    rotated = cv2.warpAffine(image, m, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=border_val)
    return rotated


def deskew(image: np.ndarray, gray_image: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """
    Detect document skew and rotate image upright.
    Returns (deskewed_image, detected_angle).
    """
    if gray_image is None:
        gray = to_grayscale(image)
    else:
        gray = gray_image

    angle = estimate_skew_angle(gray)
    if abs(angle) >= 0.5:
        deskewed = rotate_image(image, angle)
        return deskewed, angle
    return image, 0.0


def adaptive_binarize(gray_image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu thresholding combined with adaptive Gaussian thresholding
    for high contrast text on receipt paper.
    """
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
    binarized = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=15, C=8
    )
    return binarized


def preprocess_receipt(
    image_input: Any,
    deskew_enabled: bool = True,
    binarize: bool = False,
    return_color: bool = True,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Full receipt image preprocessing pipeline:
        1. Load & validate image
        2. Resize to optimal resolution band
        3. Convert to grayscale & denoise
        4. CLAHE contrast enhancement
        5. Deskew correction
        6. Optional binarization or conversion to 3-channel for OCR engine

    Returns:
        (preprocessed_image, metadata)
    """
    img = load_image(image_input)
    orig_h, orig_w = img.shape[:2]

    # Resize if needed
    resized, scale = resize_bounded(img)

    # Grayscale & enhance
    gray = to_grayscale(resized)
    denoised = denoise(gray)
    enhanced = enhance_contrast(denoised)

    # Deskew
    angle = 0.0
    if deskew_enabled:
        enhanced, angle = deskew(enhanced)
        if abs(angle) >= 0.5:
            resized = rotate_image(resized, angle)

    # Binarize if requested, otherwise return enhanced grayscale or 3-channel
    if binarize:
        output_image = adaptive_binarize(enhanced)
    elif return_color:
        output_image = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    else:
        output_image = enhanced

    metadata = {
        "original_size": (orig_w, orig_h),
        "processed_size": (output_image.shape[1], output_image.shape[0]),
        "scale_factor": scale,
        "skew_angle": angle,
    }

    return output_image, metadata

