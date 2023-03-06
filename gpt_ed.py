import cv2
import numpy as np

# Read the image
img = cv2.imread('centre2.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Apply Canny edge detection
canny = cv2.Canny(blurred, 50, 200)

# Find lines using Hough transform
lines = cv2.HoughLinesP(canny, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)

# Filter lines
filtered_lines = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = abs(np.arctan2(y2 - y1, x2 - x1) * 180.0 / np.pi)
    length = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    if length > 100 and angle < 80:
        filtered_lines.append(line)

# Draw lines on the image
for line in filtered_lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Display the result
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
