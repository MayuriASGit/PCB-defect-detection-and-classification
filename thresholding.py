
import cv2
import numpy as np


def otsu_threshold(diff):

    threshold_value, binary_mask = cv2.threshold(
        diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    
    return binary_mask, int(threshold_value)


def compute_mask_stats(binary_mask):

    white_pixels = np.sum(binary_mask == 255)
    total_pixels = binary_mask.size
    
    return {
        "white_pixels": int(white_pixels),
        "black_pixels": int(total_pixels - white_pixels),
        "white_percentage": float(white_pixels / total_pixels * 100)
    }


if __name__ == "__main__":
    # Quick test
    import sys

    diff = cv2.imread("difference_map.png", cv2.IMREAD_GRAYSCALE)
    
    if diff is None:
        print("Failed to load difference map")
        sys.exit(1)
    
    # Threshold
    binary_mask, threshold_value = otsu_threshold(diff)
    stats = compute_mask_stats(binary_mask)
    
    print(f"✓ Otsu threshold value: {threshold_value}")
    print("\nMask Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save
    cv2.imwrite("binary_mask.png", binary_mask)
    print("\n✓ Saved: binary_mask.png")

    """ output ✓ Otsu threshold value: 42

Mask Statistics:
  white_pixels: 1791
  black_pixels: 4810133
  white_percentage: 0.04

✓ Saved: binary_mask.png
"""