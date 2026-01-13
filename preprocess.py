import cv2
import numpy as np


def preprocess(image, blur_kernel=(5, 5)):

    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, blur_kernel, 0)
    
    return blurred


def preprocess_pair(template, test, auto_resize=True):

    # Preprocess both images
    template_proc = preprocess(template)
    test_proc = preprocess(test)
    
    # Check dimensions
    if template_proc.shape != test_proc.shape:
        if auto_resize:
            print(f"⚠️  Resizing test image: {test_proc.shape} -> {template_proc.shape}")
            test_proc = cv2.resize(test_proc, 
                                  (template_proc.shape[1], template_proc.shape[0]),
                                  interpolation=cv2.INTER_LINEAR)
        else:
            raise ValueError(f"Shape mismatch: template{template_proc.shape} vs test{test_proc.shape}")
    
    return template_proc, test_proc


if __name__ == "__main__":
    # Quick test
    import sys
    
    template = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)
    test = cv2.imread("test.jpg", cv2.IMREAD_GRAYSCALE)

    
    if template is None or test is None:
        print("Failed to load images")
        sys.exit(1)
    
    template_proc, test_proc = preprocess_pair(template, test)
    
    print(f"✓ Preprocessed: template{template_proc.shape}, test{test_proc.shape}")
    
    # Save for inspection
    cv2.imwrite("project\template.JPG", template_proc)
    cv2.imwrite("project\test.jpg", test_proc)
    print("✓ Saved: preprocessed_template.png, preprocessed_test.png")