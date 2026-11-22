"""
Master Pipeline:
Receipt Image -> Image Preprocessing -> PaddleOCR -> Text + Bounding Boxes -> Advanced Regex -> Company, Date, Total
"""

import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union

from src.preprocessing import preprocess_receipt, load_image
from src.paddle_ocr_engine import PaddleOCREngine
from src.advanced_regex import AdvancedRegexExtractor, OCRToken, cluster_tokens_into_lines


class ReceiptExtractionPipeline:
    """
    End-to-End Key Information Extraction (KIE) Pipeline for Receipts:
    1. Preprocessing: Resizing, denoising, CLAHE contrast enhancement, deskewing
    2. OCR: Deep learning text detection & recognition via PaddleOCR
    3. Structural Layout: Polygon conversion, normalization to [0, 1000], reading order line clustering
    4. Advanced Extraction: Spatial-regex reasoning for Company, Date, and Total
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
        deskew: bool = True,
        binarize: bool = False,
    ):
        self.deskew = deskew
        self.binarize = binarize
        self.ocr_engine = PaddleOCREngine(lang=lang, use_angle_cls=use_angle_cls, use_gpu=use_gpu)

    def process(self, image_input: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """
        Process receipt image through full pipeline:
        Receipt Image -> Preprocessing -> PaddleOCR -> Text + Bounding Boxes -> Advanced Regex -> Fields
        """
        # Step 1: Image Preprocessing
        preprocessed_img, prep_meta = preprocess_receipt(
            image_input,
            deskew_enabled=self.deskew,
            binarize=self.binarize,
            return_color=True,
        )

        # Step 2: PaddleOCR Detection & Recognition
        tokens: List[OCRToken] = self.ocr_engine.extract(preprocessed_img)

        # Step 3: Text & Bounding Box Line Clustering
        lines = cluster_tokens_into_lines(tokens)

        # Step 4: Advanced Regex & Spatial Field Extraction
        fields = {
            "company": AdvancedRegexExtractor.extract_company(lines),
            "date": AdvancedRegexExtractor.extract_date(lines),
            "total": AdvancedRegexExtractor.extract_total(lines),
        }

        return {
            "fields": fields,
            "tokens_count": len(tokens),
            "lines_count": len(lines),
            "preprocessing": prep_meta,
            "tokens": [
                {
                    "text": t.text,
                    "box": t.box,
                    "confidence": round(t.confidence, 4),
                }
                for t in tokens
            ],
            "preprocessed_image": preprocessed_img,
        }

    @staticmethod
    def visualize_result(
        image: np.ndarray,
        tokens: List[Dict[str, Any]],
        fields: Dict[str, str],
        output_path: Optional[str] = None,
    ) -> np.ndarray:
        """
        Draw color-coded bounding boxes on the receipt image:
        - Green for Company
        - Blue for Date
        - Red for Total
        - Light gray for other tokens
        """
        vis_img = image.copy()
        if len(vis_img.shape) == 2:
            vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)

        h, w = vis_img.shape[:2]

        company_str = fields.get("company", "").lower()
        date_str = fields.get("date", "").lower()
        total_str = fields.get("total", "").lower()

        for t in tokens:
            text = t["text"]
            box = t["box"]
            lower_text = text.lower()

            # Denormalize box [0, 1000] -> pixel coordinates
            x0 = int(box[0] * w / 1000.0)
            y0 = int(box[1] * h / 1000.0)
            x1 = int(box[2] * w / 1000.0)
            y1 = int(box[3] * h / 1000.0)

            # Determine color
            color = (200, 200, 200)  # default gray
            thickness = 1
            label = None

            if company_str and (lower_text in company_str or company_str in lower_text) and len(lower_text) > 2:
                color = (0, 200, 0)  # Green
                thickness = 2
                label = "COMPANY"
            elif date_str and (date_str in lower_text or lower_text in date_str):
                color = (255, 120, 0)  # Blue
                thickness = 2
                label = "DATE"
            elif total_str and (total_str in lower_text or lower_text in total_str):
                color = (0, 0, 255)  # Red
                thickness = 2
                label = "TOTAL"

            cv2.rectangle(vis_img, (x0, y0), (x1, y1), color, thickness)
            if label:
                cv2.putText(
                    vis_img,
                    label,
                    (x0, max(y0 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    2,
                )

        if output_path:
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            cv2.imwrite(output_path, vis_img)

        return vis_img


def run_pipeline_cli():
    """Command line runner for the receipt extraction pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description="Receipt Information Extraction Pipeline")
    parser.add_argument("image_path", help="Path to scanned receipt image")
    parser.add_argument("--save-viz", default=None, help="Path to save visual output")
    parser.add_argument("--no-deskew", action="store_true", help="Disable deskewing")
    parser.add_argument("--binarize", action="store_true", help="Enable binarization")
    args = parser.parse_args()

    pipeline = ReceiptExtractionPipeline(
        deskew=not args.no_deskew,
        binarize=args.binarize,
    )

    print(f"\nProcessing receipt image: {args.image_path}")
    result = pipeline.process(args.image_path)

    print("\n" + "=" * 45)
    print("EXTRACTED RECEIPT FIELDS")
    print("=" * 45)
    for field, val in result["fields"].items():
        print(f"  {field.upper():<10}: {val}")
    print("=" * 45)
    print(f"Detected {result['tokens_count']} tokens across {result['lines_count']} lines.")
    print(f"Preprocessing: scale={result['preprocessing']['scale_factor']:.2f}, skew_angle={result['preprocessing']['skew_angle']:.2f}°")

    if args.save_viz:
        pipeline.visualize_result(
            result["preprocessed_image"],
            result["tokens"],
            result["fields"],
            output_path=args.save_viz,
        )
        print(f"Visualization saved to: {args.save_viz}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_pipeline_cli()
    else:
        print("Usage: python -m src.pipeline <path_to_receipt_image> [--save-viz <output_path>]")

