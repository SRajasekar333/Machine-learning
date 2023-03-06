import cv2

# Load the image and convert it to grayscale
img = cv2.imread('cube3.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Threshold the image to create a binary image
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Find contours of the cube in the binary image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Compute the center of the contours using moments() function
M = cv2.moments(contours[0])
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

# Draw a circle at the center of the cube
cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

# Display the result
cv2.imshow('Cube', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
