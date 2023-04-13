import cv2 as cv
import numpy as np

def nothing (x):
    pass

cap = cv.VideoCapture(1)
cap1 = cv.VideoCapture(2)

cv.namedWindow("Trackbars")
cv.createTrackbar("L-H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv.createTrackbar("U-H", "Trackbars", 179, 179, nothing)
cv.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

while True:
    _, frame = cap.read()
    _, frame1 = cap1.read()
    hsv1 = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    hsv2 = cv.cvtColor(frame1,cv.COLOR_BGR2HSV)
    l_h = cv.getTrackbarPos("L-H", "Trackbars")
    l_s = cv.getTrackbarPos("L-S", "Trackbars")
    l_v = cv.getTrackbarPos("L-V", "Trackbars")
    u_h = cv.getTrackbarPos("U-H", "Trackbars")
    u_s = cv.getTrackbarPos("U-S", "Trackbars")
    u_v = cv.getTrackbarPos("U-V", "Trackbars")

    lower= np.array([[l_h, l_s, l_v]])
    upper= np.array([[u_h, u_s, u_v]])
    mask1 = cv.inRange(hsv1, lower, upper)
    mask2 = cv.inRange(hsv2, lower, upper)

    result1 = cv.bitwise_and(frame, frame, mask=mask1)
    result2 = cv.bitwise_and(frame1, frame1, mask=mask2)

    cv.imshow("frame", frame)
    cv.imshow("frame1", frame1)
    cv.imshow("mask1", mask1)
    cv.imshow("mask2", mask2)
    cv.imshow("result1", result1)
    cv.imshow("result2", result2)


    key = cv.waitKey(5)
    if key ==27:
        break
    

cap.release()
cap1.release()

cv.destroyAllWindows()

