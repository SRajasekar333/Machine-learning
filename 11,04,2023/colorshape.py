import cv2 as cv
import numpy as np

def nothing (x):
    pass

cap = cv.VideoCapture(0)
cv.namedWindow("Trackbars")
cv.createTrackbar("L-H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv.createTrackbar("U-H", "Trackbars", 179, 179, nothing)
cv.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

while True:
    _, frame = cap.read()
    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
    l_h = cv.getTrackbarPos("L-H", "Trackbars")
    l_s = cv.getTrackbarPos("L-S", "Trackbars")
    l_v = cv.getTrackbarPos("L-V", "Trackbars")
    u_h = cv.getTrackbarPos("U-H", "Trackbars")
    u_s = cv.getTrackbarPos("U-S", "Trackbars")
    u_v = cv.getTrackbarPos("U-V", "Trackbars")

    lower= np.array([[l_h, l_s, l_v]])
    upper= np.array([[u_h, u_s, u_v]])
    mask = cv.inRange(hsv, lower, upper)

    result = cv.bitwise_and(frame, frame, mask=mask)

    cv.imshow("frame", frame)
    cv.imshow("mask", mask)
    cv.imshow("result", result)

    key = cv.waitKey(1)
    if key ==27:
        break
    

cap.release()
cv.destroyAllWindows()

