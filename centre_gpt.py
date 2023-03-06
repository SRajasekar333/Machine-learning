import cv2
import numpy as np

org_img = cv2.imread('centre2.jpg')
# Get the original image size
height, width = org_img.shape[:2]
print(height, width)
# Define the new image size
new_width = 413
new_height = 360
# Resize the image
img = cv2.resize(org_img, (new_width, new_height), interpolation = cv2.INTER_LINEAR)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#edges = cv2.Canny(gray, 100, 200)

#cv2.imshow('edges', edges)

#######
#Apply Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

#Detect the edges using LSD:
lsd = cv2.createLineSegmentDetector(0)
lines, _, _, _ = lsd.detect(blur)

#Draw the detected lines on the input image
drawn_img = lsd.drawSegments(gray, lines)
edges = drawn_img
#edges = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2GRAY)

cv2.imshow('edges', edges)

#####################

contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contours:
    # Get the perimeter of the contour
    perimeter = cv2.arcLength(cnt, True)
    
    # Approximate the contour to a polygon
    approx = cv2.approxPolyDP(cnt, 0.01*perimeter, True)
    
    # Check if the polygon has 6 sides (a cube)
    if len(approx) == 6:
        # Draw the contour on the image
        cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
        
        # Find the center of the contour
        M = cv2.moments(approx)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        
        # Draw a circle at the center of the contour
        cv2.circle(img, (cx, cy), 5, (255, 0, 0), -1)
        
        # Display the image
        cv2.imshow('Image', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
