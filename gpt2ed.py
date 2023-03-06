import cv2
import numpy as np

img = cv2.imread("centre2.jpg")
#img = cv2.imread("20230104_132423.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Apply Gaussian blur to the image
blur = cv2.GaussianBlur(gray, (5, 5), 0)

#Detect the edges using LSD:
lsd = cv2.createLineSegmentDetector(0)
lines, _, _, _ = lsd.detect(blur)

#Draw the detected lines on the input image
drawn_img = lsd.drawSegments(gray, lines)
edges = cv2.cvtColor(drawn_img, cv2.COLOR_BGR2GRAY)

cv2.imshow("LSD", drawn_img)
cv2.waitKey(0)

