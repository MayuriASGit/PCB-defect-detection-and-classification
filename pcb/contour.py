import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass


@dataclass
class DefectROI:
    x: int
    y: int
    w: int
    h: int
    contour: np.ndarray
    area: float
    roi_image: np.ndarray


def extract_contours(
    binary_mask: np.ndarray,
    min_area: int = 50,
    max_area: int = 50000
) -> list:
    """Extract contours from binary defect mask."""
    contours, hierarchy = cv2.findContours(
        binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    filtered = [c for c in contours if min_area <= cv2.contourArea(c) <= max_area]
    filtered.sort(key=cv2.contourArea, reverse=True)
    return filtered


def extract_rois(
    test_image: np.ndarray,
    contours: list,
    padding: int = 10,
    roi_size: tuple = (128, 128)
) -> list[DefectROI]:
    """Extract Region of Interest crops for each detected defect."""
    h, w = test_image.shape[:2]
    rois = []

    for contour in contours:
        area = cv2.contourArea(contour)
        x, y, bw, bh = cv2.boundingRect(contour)

        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(w, x + bw + padding)
        y2 = min(h, y + bh + padding)

        roi_crop = test_image[y1:y2, x1:x2]
        if roi_crop.size == 0:
            continue

        roi_resized = cv2.resize(roi_crop, roi_size)

        rois.append(DefectROI(
            x=x1, y=y1, w=x2 - x1, h=y2 - y1,
            contour=contour,
            area=area,
            roi_image=roi_resized
        ))

    return rois


def draw_contours(
    image: np.ndarray,
    rois: list[DefectROI],
    labels: list[str] = None,
    confidences: list[float] = None,
    color_map: dict = None
) -> np.ndarray:
    """Draw bounding boxes and labels on image."""
    output = image.copy()

    default_colors = {
        "missing_hole": (0, 0, 255),
        "mouse_bite": (255, 165, 0),
        "open_circuit": (255, 0, 0),
        "short": (0, 255, 255),
        "spur": (255, 0, 255),
        "spurious_copper": (0, 128, 255),
        "defect": (0, 255, 0),
    }
    if color_map:
        default_colors.update(color_map)

    for i, roi in enumerate(rois):
        label = labels[i] if labels else "defect"
        conf = confidences[i] if confidences else None
        color = default_colors.get(label.lower().replace(" ", "_"), (0, 255, 0))

        cv2.rectangle(output, (roi.x, roi.y), (roi.x + roi.w, roi.y + roi.h), color, 2)

        display_label = label.replace("_", " ").title()
        if conf is not None:
            display_label += f" {conf:.0%}"

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (tw, th), baseline = cv2.getTextSize(display_label, font, font_scale, thickness)

        label_y = max(roi.y - 5, th + 5)
        cv2.rectangle(
            output,
            (roi.x, label_y - th - baseline),
            (roi.x + tw, label_y + baseline),
            color, -1
        )
        cv2.putText(
            output, display_label,
            (roi.x, label_y),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

    return output


def draw_difference_overlay(
    template: np.ndarray,
    test: np.ndarray,
    binary_mask: np.ndarray
) -> np.ndarray:
    """Create a side-by-side comparison with highlighted defects."""
    h, w = template.shape[:2]
    overlay = test.copy()

    colored_mask = np.zeros_like(test)
    colored_mask[binary_mask > 0] = (0, 0, 255)
    overlay = cv2.addWeighted(overlay, 0.7, colored_mask, 0.3, 0)

    separator = np.full((h, 4, 3), 128, dtype=np.uint8)
    combined = np.hstack([template, separator, test, separator, overlay])

    for i, title in enumerate(["Template", "Test Image", "Defect Overlay"]):
        x_pos = i * (w + 4) + 5
        cv2.putText(combined, title, (x_pos, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return combined
