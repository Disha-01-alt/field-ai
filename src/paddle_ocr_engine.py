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
                import paddleocr
                from paddleocr import PaddleOCR
            except ImportError as e:
                raise ImportError(
                    "PaddleOCR is not installed. Please install via: pip install paddlepaddle paddleocr"
                ) from e

            self._is_v2 = str(getattr(paddleocr, "__version__", "2")).startswith("2")
            if self._is_v2:
                try:
                    self._ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang,
                                          use_gpu=self.use_gpu, show_log=False)
                except TypeError:
                    self._ocr = PaddleOCR(use_angle_cls=self.use_angle_cls, lang=self.lang)
            else:
                self._ocr = self._build_v3(PaddleOCR)

    def _build_v3(self, PaddleOCR, **extra):
        # PaddleOCR 3.x: skip the whole-page orientation / unwarping models,
        # our own preprocessing already deskews the receipt.
        return PaddleOCR(
            lang=self.lang,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=self.use_angle_cls,
            **extra,
        )

    def _run_ocr(self, image: np.ndarray) -> List[Tuple[Any, str, float]]:
        """Run OCR and return [(polygon, text, confidence), ...] for either PaddleOCR 2.x or 3.x."""
        if self._is_v2:
            results = self._ocr.ocr(image, cls=self.use_angle_cls)
        else:
            try:
                results = self._ocr.predict(image)
            except NotImplementedError:
                # Some paddlepaddle 3.x CPU builds crash inside the oneDNN (MKL-DNN)
                # backend ("ConvertPirAttribute2RuntimeAttribute not support").
                # Rebuild without oneDNN and try again.
                from paddleocr import PaddleOCR
                self._ocr = self._build_v3(PaddleOCR, enable_mkldnn=False)
                results = self._ocr.predict(image)
        return self.parse_results(results)

    @staticmethod
    def parse_results(results: Any) -> List[Tuple[Any, str, float]]:
        """
        Convert raw PaddleOCR output into [(polygon, text, confidence), ...].
        - 2.x: [[ [poly, (text, conf)], ... ]]  (one list per page)
        - 3.x: [ {"rec_texts": [...], "rec_scores": [...], "rec_polys": [...]}, ... ]
        """
        items: List[Tuple[Any, str, float]] = []
        if not results:
            return items
        page = results[0]
        if page is None:
            return items

        # 3.x result object (dict-like)
        try:
            texts = page["rec_texts"]
            scores = page["rec_scores"]
            polys = page["rec_polys"] if "rec_polys" in page else page["dt_polys"]
            for poly, text, conf in zip(polys, texts, scores):
                items.append((np.asarray(poly).tolist(), str(text), float(conf)))
            return items
        except (KeyError, TypeError, IndexError):
            pass

        # 2.x nested lists
        for line in page:
            if not line or len(line) < 2:
                continue
            poly, (text, conf) = line
            items.append((poly, str(text), float(conf)))
        return items

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

        return self.to_tokens(self._run_ocr(img_to_run), w, h)

    @staticmethod
    def to_tokens(items: List[Tuple[Any, str, float]], w: int, h: int) -> List[OCRToken]:
        """Polygons in pixels -> OCRTokens with boxes normalized to [0, 1000]."""
        tokens: List[OCRToken] = []
        for poly, text, conf in items:
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

            tokens.append(OCRToken(
                text=str(text).strip(),
                box=[norm_x0, norm_y0, norm_x1, norm_y1],
                confidence=float(conf),
                raw_box=poly,
            ))

        # Sort tokens naturally in top-to-bottom, left-to-right order
        tokens.sort(key=lambda t: (t.y0, t.x0))
        return tokens

