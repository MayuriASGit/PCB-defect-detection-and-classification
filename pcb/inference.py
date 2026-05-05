import cv2
import numpy as np
import pandas as pd
from PIL import Image
from dataclasses import dataclass, field
from datetime import datetime
import os

from .preprocessing import preprocess_images, subtract_images, apply_threshold
from .contour import extract_contours, extract_rois, draw_contours, draw_difference_overlay, DefectROI
from .model import (
    get_model, predict_roi, simulate_trained_predictions,
    DEFECT_CLASSES, DEFECT_DESCRIPTIONS, PCBDefectClassifier
)


@dataclass
class InferenceResult:
    defect_count: int
    labels: list[str]
    confidences: list[float]
    all_probs: list[list[float]]
    rois: list[DefectROI]
    annotated_image: np.ndarray
    diff_image: np.ndarray
    binary_mask: np.ndarray
    comparison_image: np.ndarray
    processing_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, (roi, label, conf) in enumerate(zip(self.rois, self.labels, self.confidences)):
            rows.append({
                "defect_id": i + 1,
                "class": label,
                "description": DEFECT_DESCRIPTIONS.get(label, "Unknown defect"),
                "confidence": f"{conf:.4f}",
                "confidence_pct": f"{conf:.1%}",
                "bbox_x": roi.x,
                "bbox_y": roi.y,
                "bbox_w": roi.w,
                "bbox_h": roi.h,
                "area_px": int(roi.area),
                "timestamp": self.timestamp,
            })
        return pd.DataFrame(rows) if rows else pd.DataFrame(columns=[
            "defect_id", "class", "description", "confidence",
            "confidence_pct", "bbox_x", "bbox_y", "bbox_w", "bbox_h",
            "area_px", "timestamp"
        ])


def run_inference(
    template_pil: Image.Image,
    test_pil: Image.Image,
    threshold_method: str = "otsu",
    manual_threshold: int = 30,
    min_defect_area: int = 50,
    max_defect_area: int = 50000,
    model: PCBDefectClassifier = None,
    use_simulation: bool = True,
    device: str = "cpu",
) -> InferenceResult:
    """Full PCB defect detection and classification pipeline."""
    import time
    start = time.time()

    template_cv, test_cv = preprocess_images(template_pil, test_pil)

    diff = subtract_images(template_cv, test_cv)

    binary_mask = apply_threshold(diff, method=threshold_method, manual_threshold=manual_threshold)

    contours = extract_contours(binary_mask, min_area=min_defect_area, max_area=max_defect_area)

    rois = extract_rois(test_cv, contours, padding=10, roi_size=(128, 128))

    labels, confidences, all_probs = [], [], []
    for i, roi in enumerate(rois):
        if model is not None and not use_simulation:
            label, conf, probs = predict_roi(model, roi.roi_image, device=device)
        else:
            label, conf, probs = simulate_trained_predictions(roi.roi_image, roi_index=i)
        labels.append(label)
        confidences.append(conf)
        all_probs.append(probs)

    annotated = draw_contours(test_cv, rois, labels=labels, confidences=confidences)

    diff_8bit = diff.copy()
    comparison = draw_difference_overlay(template_cv, test_cv, binary_mask)

    elapsed_ms = (time.time() - start) * 1000

    return InferenceResult(
        defect_count=len(rois),
        labels=labels,
        confidences=confidences,
        all_probs=all_probs,
        rois=rois,
        annotated_image=annotated,
        diff_image=diff_8bit,
        binary_mask=binary_mask,
        comparison_image=comparison,
        processing_time_ms=elapsed_ms,
    )


def run_full_pipeline(
    template_pil: Image.Image,
    test_pil: Image.Image,
    settings: dict = None
) -> InferenceResult:
    """Convenience wrapper with default settings."""
    settings = settings or {}
    return run_inference(
        template_pil=template_pil,
        test_pil=test_pil,
        threshold_method=settings.get("threshold_method", "otsu"),
        manual_threshold=settings.get("manual_threshold", 30),
        min_defect_area=settings.get("min_defect_area", 50),
        max_defect_area=settings.get("max_defect_area", 50000),
        model=settings.get("model", None),
        use_simulation=settings.get("use_simulation", True),
        device=settings.get("device", "cpu"),
    )


def save_outputs(result: InferenceResult, output_dir: str = "outputs") -> dict[str, str]:
    """Save annotated image, diff map, binary mask, and CSV log."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = {}

    annotated_path = os.path.join(output_dir, f"annotated_{ts}.png")
    cv2.imwrite(annotated_path, result.annotated_image)
    paths["annotated"] = annotated_path

    diff_path = os.path.join(output_dir, f"diff_map_{ts}.png")
    cv2.imwrite(diff_path, result.diff_image)
    paths["diff_map"] = diff_path

    mask_path = os.path.join(output_dir, f"binary_mask_{ts}.png")
    cv2.imwrite(mask_path, result.binary_mask)
    paths["binary_mask"] = mask_path

    comparison_path = os.path.join(output_dir, f"comparison_{ts}.png")
    cv2.imwrite(comparison_path, result.comparison_image)
    paths["comparison"] = comparison_path

    df = result.to_dataframe()
    csv_path = os.path.join(output_dir, f"defect_log_{ts}.csv")
    df.to_csv(csv_path, index=False)
    paths["csv_log"] = csv_path

    return paths
