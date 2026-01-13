import cv2
import numpy as np


def subtract_images(template, test):

    if template.shape != test.shape:
        raise ValueError(f"Shape mismatch: {template.shape} vs {test.shape}")
    
    # Compute absolute difference
    diff = cv2.absdiff(test, template)
    
    return diff


def compute_metrics(diff):

    return {
        "mean": float(np.mean(diff)),
        "std": float(np.std(diff)),
        "min": int(np.min(diff)),
        "max": int(np.max(diff)),
        "non_zero_percentage": float(np.count_nonzero(diff) / diff.size * 100)
    }


if __name__ == "__main__":
    # Quick test
    import sys
    
    template = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)
    test = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)
    
    from preprocess import preprocess_pair
    

    
    if template is None or test is None:
        print("Failed to load images")
        sys.exit(1)
    
    # Preprocess
    template_proc, test_proc = preprocess_pair(template, test)
    
    # Subtract
    diff = subtract_images(template_proc, test_proc)
    metrics = compute_metrics(diff)
    
    print("✓ Difference computed")
    print("\nMetrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Save
    cv2.imwrite("difference_map.png", diff)
    print("\n✓ Saved: difference_map.png")
    """
    ✓ Difference computed

Metrics:
  mean: 0.27
  std: 1.80
  min: 0
  max: 173
  non_zero_percentage: 23.70

✓ Saved: difference_map.png"""