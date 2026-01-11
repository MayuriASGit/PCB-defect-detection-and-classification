import cv2
import matplotlib.pyplot as plt

template = cv2.imread("template.jpg", cv2.IMREAD_GRAYSCALE)
test = cv2.imread("input", cv2.IMREAD_GRAYSCALE)

plt.subplot(1,2,1); plt.imshow(template, cmap='gray'); plt.title("Template")
plt.subplot(1,2,2); plt.imshow(test, cmap='gray'); plt.title("Test")
plt.show()

template = cv2.GaussianBlur(template, (5,5), 0)
test = cv2.GaussianBlur(test, (5,5), 0)

diff = cv2.absdiff(test, template)
plt.imshow(diff, cmap='gray'); plt.title("Difference Map")
plt.show()

_, thresh = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
highlighted = cv2.dilate(clean, kernel, iterations=2)

plt.imshow(highlighted, cmap='gray'); plt.title("Defect Regions")
plt.show()
