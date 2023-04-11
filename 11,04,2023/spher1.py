import cv2
import numpy as np

# Load image
img = cv2.imread('sphere6.jpg')

# Convert to grayscale
#gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img = cv2.medianBlur(img,5)

# Apply Canny edge detection
#edges = cv2.Canny(gray, 50, 150)

# Apply Hough transform to detect circles
circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 346,
                           param1=418, param2=31, minRadius=37, maxRadius=87)

# Draw detected circles
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)

# Display image with detected circle
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Get center of detected circle
if circles is not None:
    center = (circles[0][0], circles[0][1])
    print("Detected center:", center)
