import cv2
import numpy as np

# Assume 'highlighted' is your binary defect mask
contours, hierarchy = cv2.findContours(highlighted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw contours for visualization
contour_img = cv2.cvtColor(test, cv2.COLOR_GRAY2BGR)
cv2.drawContours(contour_img, contours, -1, (0,0,255), 2)

rois = []
for i, cnt in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(contour_img, (x,y), (x+w,y+h), (0,255,0), 2)

    # Crop defect ROI
    roi = test[y:y+h, x:x+w]
    rois.append((i, roi, (x,y,w,h)))

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interArea = max(0, xB-xA) * max(0, yB-yA)
    boxAArea = boxA[2]*boxA[3]
    boxBArea = boxB[2]*boxB[3]
    return interArea / float(boxAArea + boxBArea - interArea)

import os

for i, (idx, roi, bbox) in enumerate(rois):
    label = "spur"  # Example: replace with matched ground-truth label
    save_path = f"dataset/{label}/defect_{idx}.png"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, roi)
