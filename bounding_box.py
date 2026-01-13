

import cv2
import numpy as np
from typing import List, Tuple


def extract_bounding_boxes(contours: List[np.ndarray],
                           padding: int = 10,
                           image_shape: Tuple[int, int] = None) -> List[Tuple[int, int, int, int]]:

    boxes = []
    
    for contour in contours:
        # Get base bounding box
        x, y, w, h = cv2.boundingRect(contour)
        
        # Add padding
        x_padded = x - padding
        y_padded = y - padding
        w_padded = w + 2 * padding
        h_padded = h + 2 * padding
        
        # Clip to image boundaries if shape provided
        if image_shape is not None:
            img_h, img_w = image_shape
            x_padded = max(0, x_padded)
            y_padded = max(0, y_padded)
            w_padded = min(w_padded, img_w - x_padded)
            h_padded = min(h_padded, img_h - y_padded)
        
        boxes.append((x_padded, y_padded, w_padded, h_padded))
    
    return boxes


def compute_box_statistics(boxes: List[Tuple[int, int, int, int]]) -> dict:

    if len(boxes) == 0:
        return {
            'count': 0,
            'avg_width': 0,
            'avg_height': 0,
            'avg_area': 0,
            'min_area': 0,
            'max_area': 0
        }
    
    widths = [w for _, _, w, _ in boxes]
    heights = [h for _, _, _, h in boxes]
    areas = [w * h for _, _, w, h in boxes]
    
    return {
        'count': len(boxes),
        'avg_width': float(np.mean(widths)),
        'avg_height': float(np.mean(heights)),
        'avg_area': float(np.mean(areas)),
        'min_area': float(np.min(areas)),
        'max_area': float(np.max(areas)),
        'std_width': float(np.std(widths)),
        'std_height': float(np.std(heights))
    }


def merge_overlapping_boxes(boxes: List[Tuple[int, int, int, int]],
                            iou_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:

    if len(boxes) <= 1:
        return boxes
    
    # Convert to (x1, y1, x2, y2) format
    boxes_xyxy = [(x, y, x + w, y + h) for x, y, w, h in boxes]
    
    merged = []
    used = [False] * len(boxes_xyxy)
    
    for i in range(len(boxes_xyxy)):
        if used[i]:
            continue
        
        x1, y1, x2, y2 = boxes_xyxy[i]
        
        # Find all overlapping boxes
        for j in range(i + 1, len(boxes_xyxy)):
            if used[j]:
                continue
            
            xj1, yj1, xj2, yj2 = boxes_xyxy[j]
            
            # Compute IoU
            xi1 = max(x1, xj1)
            yi1 = max(y1, yj1)
            xi2 = min(x2, xj2)
            yi2 = min(y2, yj2)
            
            inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
            box1_area = (x2 - x1) * (y2 - y1)
            box2_area = (xj2 - xj1) * (yj2 - yj1)
            union_area = box1_area + box2_area - inter_area
            
            iou = inter_area / union_area if union_area > 0 else 0
            
            # Merge if overlapping
            if iou > iou_threshold:
                x1 = min(x1, xj1)
                y1 = min(y1, yj1)
                x2 = max(x2, xj2)
                y2 = max(y2, yj2)
                used[j] = True
        
        merged.append((x1, y1, x2 - x1, y2 - y1))
        used[i] = True
    
    return merged


def visualize_boxes(image: np.ndarray,
                    boxes: List[Tuple[int, int, int, int]],
                    color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2,
                    show_labels: bool = True) -> np.ndarray:

    # Convert grayscale to BGR if needed
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Draw each box
    for i, (x, y, w, h) in enumerate(boxes):
        # Draw rectangle
        cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw label
        if show_labels:
            label = f"#{i+1}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(vis_image, (x, y - label_size[1] - 5), 
                         (x + label_size[0], y), color, -1)
            cv2.putText(vis_image, label, (x, y - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_image


if __name__ == "__main__":
    # Test bounding box extraction
    import sys
    from contour_detection import find_contours
    from filter_contour import ContourFilter
    
    # Load mask
    mask = cv2.imread("binary_mask.png", cv2.IMREAD_GRAYSCALE)
    
    if mask is None:
        print(f"Error: Could not load {sys.argv[1]}")
        sys.exit(1)
    
    padding = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    
    print(f"✓ Loaded mask: {mask.shape}")
    print(f"✓ Padding: {padding} pixels")
    
    # Find and filter contours
    contours = find_contours(mask)
    print(f"✓ Found {len(contours)} contours")
    
    filter_obj = ContourFilter(min_area=50.0)
    filtered, _ = filter_obj.filter_contours(contours, verbose=False)
    print(f"✓ Filtered to {len(filtered)} contours")
    
    # Extract bounding boxes
    boxes = extract_bounding_boxes(filtered, padding=padding, image_shape=mask.shape)
    print(f"✓ Extracted {len(boxes)} bounding boxes")
    
    # Compute statistics
    stats = compute_box_statistics(boxes)
    print(f"\nBounding Box Statistics:")
    print(f"  Count: {stats['count']}")
    print(f"  Avg size: {stats['avg_width']:.1f} × {stats['avg_height']:.1f}")
    print(f"  Avg area: {stats['avg_area']:.1f} px²")
    print(f"  Area range: {stats['min_area']:.1f} - {stats['max_area']:.1f} px²")
    
    # Visualize
    vis = visualize_boxes(mask, boxes, color=(0, 255, 0), thickness=2)
    
    output_path = "bounding_boxes_visualization.png"
    cv2.imwrite(output_path, vis)
    print(f"\n✓ Saved visualization: {output_path}")

    """
    ✓ Loaded mask: (1586, 3034)
✓ Padding: 10 pixels
✓ Found 3 contours
✓ Filtered to 3 contours
✓ Extracted 3 bounding boxes

Bounding Box Statistics:
  Count: 3
  Avg size: 51.3 × 44.0
  Avg area: 2259.0 px²
  Area range: 2193.0 - 2340.0 px²

✓ Saved visualization: bounding_boxes_visualization.png"""