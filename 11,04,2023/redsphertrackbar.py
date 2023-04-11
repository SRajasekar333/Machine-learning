import cv2
import numpy as np

def nothing(x):
    pass

# Create a black image and a window
img = np.zeros((300,512,3), np.uint8)
cv2.namedWindow('image')

# Create trackbars for color change
cv2.createTrackbar('Hue', 'image', 0, 179, nothing)
cv2.createTrackbar('Saturation', 'image', 50, 255, nothing)
cv2.createTrackbar('Value', 'image', 50, 255, nothing)

while(1):
    # Read the image
    frame = cv2.imread('imageR3.png')

    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Get trackbar positions
    h = cv2.getTrackbarPos('Hue', 'image')
    s = cv2.getTrackbarPos('Saturation', 'image')
    v = cv2.getTrackbarPos('Value', 'image')

    # Define the lower and upper bounds of the red color in HSV
    lower_red = np.array([h, s, v])
    upper_red = np.array([10, 255, 255])

    # Create a mask
    mask = cv2.inRange(hsv, lower_red, upper_red)
    mask1 = cv2.medianBlur(mask, 5)


    # Show the mask
    cv2.imshow('image', mask)
    cv2.imshow('image1', mask1)

    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
