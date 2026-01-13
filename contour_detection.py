
import cv2
import numpy as np
from typing import List, Tuple


def find_contours(binary_mask: np.ndarray, mode: str = 'external') -> List[np.ndarray]:

    if binary_mask is None or binary_mask.size == 0:
        raise ValueError("Binary mask is empty")
    
    if len(binary_mask.shape) != 2:
        raise ValueError(f"Expected 2D grayscale mask, got shape {binary_mask.shape}")
    
    # Select retrieval mode
    retrieval_mode = cv2.RETR_EXTERNAL if mode == 'external' else cv2.RETR_TREE
    
    # Find contours
    # cv2.findContours returns (contours, hierarchy) in OpenCV 4.x
    contours, hierarchy = cv2.findContours(
        binary_mask,
        retrieval_mode,
        cv2.CHAIN_APPROX_SIMPLE  # Compress horizontal/vertical segments
    )
    
    return list(contours)


def compute_contour_properties(contour: np.ndarray) -> dict:

    # Area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, closed=True)
    
    # Centroid using moments
    M = cv2.moments(contour)
    if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
    else:
        cx, cy = 0, 0
    
    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    
    # Circularity: measures how close shape is to a perfect circle
    # 1.0 = perfect circle, lower values = elongated/irregular shapes
    circularity = 0.0
    if perimeter > 0:
        circularity = 4 * np.pi * area / (perimeter * perimeter)
    
    # Aspect ratio
    aspect_ratio = w / h if h > 0 else 0.0
    
    return {
        'area': float(area),
        'perimeter': float(perimeter),
        'centroid': (cx, cy),
        'bounding_rect': (x, y, w, h),
        'circularity': float(circularity),
        'aspect_ratio': float(aspect_ratio)
    }


def visualize_contours(image: np.ndarray, contours: List[np.ndarray], 
                       color: Tuple[int, int, int] = (0, 255, 0),
                       thickness: int = 2) -> np.ndarray:

    # Convert grayscale to BGR if needed
    if len(image.shape) == 2:
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis_image = image.copy()
    
    # Draw all contours
    cv2.drawContours(vis_image, contours, -1, color, thickness)
    
    return vis_image


if __name__ == "__main__":
    # Test with a sample binary mask
    import sys
    

    # Load mask
    mask = cv2.imread("binary_mask.png", cv2.IMREAD_GRAYSCALE)
    

    
    print(f"✓ Loaded mask: {mask.shape}")
    
    # Find contours
    contours = find_contours(mask, mode='external')
    print(f"✓ Found {len(contours)} contours")
    
    # Analyze each contour
    print("\nContour Properties:")
    print("-" * 80)
    for i, contour in enumerate(contours[:10]):  # Show first 10
        props = compute_contour_properties(contour)
        print(f"\nContour {i+1}:")
        print(f"  Area: {props['area']:.1f} px²")
        print(f"  Perimeter: {props['perimeter']:.1f} px")
        print(f"  Centroid: {props['centroid']}")
        print(f"  Bounding box: {props['bounding_rect']}")
        print(f"  Circularity: {props['circularity']:.3f}")
        print(f"  Aspect ratio: {props['aspect_ratio']:.2f}")
    
    if len(contours) > 10:
        print(f"\n... and {len(contours) - 10} more contours")
    
    # Visualize
    vis = visualize_contours(mask, contours, color=(0, 255, 0), thickness=2)
    
    output_path = "contours_visualization.png"
    cv2.imwrite(output_path, vis)
    print(f"\n✓ Saved visualization: {output_path}")
    """
    ✓ Loaded mask: (1586, 3034)
✓ Found 3 contours

Contour Properties:
--------------------------------------------------------------------------------

Contour 1:
  Area: 554.5 px²
  Perimeter: 89.0 px
  Centroid: (2378, 831)
  Bounding box: (2364, 820, 31, 24)
  Circularity: 0.879
  Aspect ratio: 1.29

Contour 2:
  Area: 593.5 px²
  Perimeter: 93.0 px
  Centroid: (1530, 827)
  Bounding box: (1515, 815, 32, 25)
  Circularity: 0.862
  Aspect ratio: 1.28

Contour 3:
  Area: 522.5 px²
  Perimeter: 89.0 px
  Centroid: (2611, 265)
  Bounding box: (2597, 255, 31, 23)
  Circularity: 0.829
  Aspect ratio: 1.35

✓ Saved visualization: contours_visualization.png"""