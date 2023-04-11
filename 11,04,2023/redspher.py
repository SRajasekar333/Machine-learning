import cv2
import numpy as np

img = cv2.imread('imageR3.png')

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#lower_red = np.array([0, 50, 50])
upper_red = np.array([10, 255, 255])

lower_red = np.array([0, 17, 103])

mask = cv2.inRange(hsv, lower_red, upper_red)
cv2.imshow('maskimage', mask)


kernel = np.ones((5,5),np.uint8)
#mask = cv2.erode(mask,kernel,iterations = 1)
#mask = cv2.dilate(mask,kernel,iterations = 1)

#cv2.imshow('maskimage', mask)


contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

max_contour = max(contours, key=cv2.contourArea)
M = cv2.moments(max_contour)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])

cv2.circle(img, (cx,cy), 5, (0, 0, 255), -1)
cv2.circle(img, (cx,cy), int(M['m00']//max_contour.size), (0, 255, 0), 2)

cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
