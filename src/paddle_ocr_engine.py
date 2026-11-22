"""
PaddleOCR Engine Wrapper for Receipt Information Extraction.

Provides:
1. High-accuracy text detection (DBNet) + recognition (SVTR/CRNN) + angle classification.
2. Polygon-to-axis-aligned bounding box conversion.
3. Coordinate normalization to [0, 1000] scale.
4. Conversion to structured OCRToken list for downstream spatial & regex extraction.
"""

import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path

from src.advanced_regex import OCRToken


class PaddleOCREngine:
    """Wrapper around PaddleOCR for receipt documents."""

    def __init__(self, lang: str = "en", use_angle_cls: bool = True, use_gpu: bool = False):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.use_gpu = use_gpu
        self._ocr = None

    def _init_ocr(self):
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR
                # PaddleOCR 2.x/3.x argument compatibility
                try:
                    self._ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang, use_gpu=self.use_gpu)
                except TypeError:
                    # In newer PaddleOCR/PaddleX versions
                    self._ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang)
            except ImportError as e:
                raise ImportError(
                    "PaddleOCR is not installed. Please install via: pip install paddlepaddle paddleocr"
                ) from e

    def extract(self, image: np.ndarray) -> List[OCRToken]:
        """
        Run PaddleOCR on an image array (RGB/BGR or Grayscale).
        Returns list of OCRToken objects with normalized [0, 1000] coordinates.
        """
        self._init_ocr()

        # Handle grayscale by converting to 3-channel
        if len(image.shape) == 2:
            import cv2
            img_to_run = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            img_to_run = image

        h, w = img_to_run.shape[:2]
        if h == 0 or w == 0:
            return []

        # Run OCR inference
        results = self._ocr.ocr(img_to_run, cls=self.use_angle_cls)

        tokens: List[OCRToken] = []

        if not results or results[0] is None:
            return tokens

        # PaddleOCR returns a list of results per page; page 0 is results[0]
        page_results = results[0] if len(results) > 0 else []

        for line in page_results:
            if not line or len(line) < 2:
                continue

            poly, (text, conf) = line
            if not text or not str(text).strip():
                continue

            # poly is [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]
            xs = [pt[0] for pt in poly]
            ys = [pt[1] for pt in poly]

            x_min = max(0.0, min(xs))
            y_min = max(0.0, min(ys))
            x_max = min(float(w), max(xs))
            y_max = min(float(h), max(ys))

            # Normalize to [0, 1000]
            norm_x0 = min(1000, max(0, int(round((x_min / w) * 1000))))
            norm_y0 = min(1000, max(0, int(round((y_min / h) * 1000))))
            norm_x1 = min(1000, max(0, int(round((x_max / w) * 1000))))
            norm_y1 = min(1000, max(0, int(round((y_max / h) * 1000))))

            token = OCRToken(
                text=str(text).strip(),
                box=[norm_x0, norm_y0, norm_x1, norm_y1],
                confidence=float(conf),
                raw_box=poly,
            )
            tokens.append(token)

        # Sort tokens naturally in top-to-bottom, left-to-right order
        tokens.sort(key=lambda t: (t.y0, t.x0))
        return tokens

