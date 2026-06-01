import cv2
import numpy as np

# Load ref1.png
img = cv2.imread('ref1.png')
if img is None:
    print("Error: ref1.png not found")
    exit()

h, w, _ = img.shape

# Take a central crop (e.g., 30% of the image size)
crop_h, crop_w = int(h * 0.3), int(w * 0.3)
start_y, start_x = int(h * 0.4), int(w * 0.4)
crop = img[start_y:start_y+crop_h, start_x:start_x+crop_w]

# Rotate the crop by 30 degrees to test rotation invariance
center = (crop_w // 2, crop_h // 2)
M = cv2.getRotationMatrix2D(center, 30, 1.0)
rotated_crop = cv2.warpAffine(crop, M, (crop_w, crop_h))

# Save as test_crop.png
cv2.imwrite('test_crop.png', rotated_crop)
print("Generated test_crop.png (30% crop of ref1, rotated 30 degrees)")
