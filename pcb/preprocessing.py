import cv2
import numpy as np
from PIL import Image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    img = np.array(pil_image.convert("RGB"))
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    if len(cv2_image.shape) == 2:
        return Image.fromarray(cv2_image)
    rgb = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def align_images(template: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Align test image to template using ORB feature matching."""
    h, w = template.shape[:2]
    test_resized = cv2.resize(test, (w, h))

    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test_resized, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(nfeatures=1000)
    kp1, des1 = orb.detectAndCompute(gray_template, None)
    kp2, des2 = orb.detectAndCompute(gray_test, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return template, test_resized

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:min(50, len(matches))]

    if len(good_matches) < 4:
        return template, test_resized

    src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    if H is None:
        return template, test_resized

    aligned = cv2.warpPerspective(test_resized, H, (w, h))
    return template, aligned


def preprocess_images(
    template_img: Image.Image,
    test_img: Image.Image,
    target_size: tuple[int, int] = (512, 512)
) -> tuple[np.ndarray, np.ndarray]:
    """Preprocess and align template and test images."""
    template_cv = pil_to_cv2(template_img)
    test_cv = pil_to_cv2(test_img)

    template_cv = cv2.resize(template_cv, target_size)
    test_cv = cv2.resize(test_cv, target_size)

    template_cv = cv2.GaussianBlur(template_cv, (3, 3), 0)
    test_cv = cv2.GaussianBlur(test_cv, (3, 3), 0)

    template_aligned, test_aligned = align_images(template_cv, test_cv)

    return template_aligned, test_aligned


def subtract_images(
    template: np.ndarray,
    test: np.ndarray
) -> np.ndarray:
    """Compute absolute difference between template and test images."""
    gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray_test = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_template, gray_test)
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    return diff


def apply_threshold(
    diff_image: np.ndarray,
    method: str = "otsu",
    manual_threshold: int = 30
) -> np.ndarray:
    """Apply thresholding to difference image to isolate defects."""
    if method == "otsu":
        _, binary = cv2.threshold(
            diff_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    elif method == "adaptive":
        binary = cv2.adaptiveThreshold(
            diff_image, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
    else:
        _, binary = cv2.threshold(diff_image, manual_threshold, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.dilate(binary, kernel, iterations=1)

    return binary
