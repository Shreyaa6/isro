import cv2
import numpy as np

# Load ref1.png
img = cv2.imread('ref1.png')
if img is None:
    print("Error: ref1.png not found")
    exit()

# Apply some transformations (Rotation, slightly different scale)
rows, cols, ch = img.shape
M = cv2.getRotationMatrix2D((cols/2, rows/2), 15, 1.1)
transformed = cv2.warpAffine(img, M, (cols, rows))

# Save as test_match.png
cv2.imwrite('test_match.png', transformed)
print("Generated test_match.png from ref1.png")
