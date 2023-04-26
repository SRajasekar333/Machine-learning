import cv2 as cv
import numpy as np

def nothing (x):
    pass

img = cv.imread("images/L20/imageL(5)1.png")
#hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#cap = cv.VideoCapture(0)
cv.namedWindow("Trackbars")
cv.resizeWindow("Trackbars",640,240)
cv.createTrackbar("L-H", "Trackbars", 0, 179, nothing)
cv.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
cv.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv.createTrackbar("U-H", "Trackbars", 179, 179, nothing)
cv.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv.createTrackbar("U-V", "Trackbars", 255, 255, nothing)

while True:
    #_, frame = cap.read()
    hsv = cv.cvtColor(img,cv.COLOR_BGR2HSV)
    l_h = cv.getTrackbarPos("L-H", "Trackbars")
    l_s = cv.getTrackbarPos("L-S", "Trackbars")
    l_v = cv.getTrackbarPos("L-V", "Trackbars")
    u_h = cv.getTrackbarPos("U-H", "Trackbars")
    u_s = cv.getTrackbarPos("U-S", "Trackbars")
    u_v = cv.getTrackbarPos("U-V", "Trackbars")

    lower= np.array([[l_h, l_s, l_v]])
    upper= np.array([[u_h, u_s, u_v]])
    mask = cv.inRange(hsv, lower, upper)

    result = cv.bitwise_and(img, img, mask=mask)

    kernel = np.ones((5,5),np.uint8)
    dilate = cv.dilate(mask,kernel,iterations = 1)
    closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    closing1 = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)

    cv.imshow("frame", img)
    cv.imshow("mask", mask)
    cv.imshow("result", result)
    #cv.imshow('dilate', dilate)
    cv.imshow('closing',closing)
    #cv.imshow('closing1',closing1)

    '''
    cv.namedWindow("TrackBars")
    cv.resizeWindow("TrackBars",640,240)
    cv.createTrackbar("centerDist","TrackBars",1,1000,nothing)
    cv.createTrackbar("param1","TrackBars",200,500,nothing)
    cv.createTrackbar("param2","TrackBars",30,50,nothing)
    cv.createTrackbar("minRadius","TrackBars",10,100,nothing)
    cv.createTrackbar("maxRadius","TrackBars",0,600,nothing)
    while True:
        cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
        centerDist = cv.getTrackbarPos("centerDist","TrackBars")
        param1 = cv.getTrackbarPos("param1", "TrackBars")
        param2 = cv.getTrackbarPos("param2", "TrackBars")
        minRadius = cv.getTrackbarPos("minRadius", "TrackBars")
        maxRadius = cv.getTrackbarPos("maxRadius", "TrackBars")
        #cv.setTrackbarMin("maxRadius", "TrackBars", minRadius)
        print(centerDist,param1,param2,minRadius,maxRadius)

        circles = cv.HoughCircles(closing,cv.HOUGH_GRADIENT,1,centerDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    '''


    key = cv.waitKey(1)
    if key ==27:
        break
    
cv.waitKey(1)
#cap.release()
cv.destroyAllWindows()

'''
kernel = np.ones((5,5),np.uint8)
dilate = cv.dilate(mask,kernel,iterations = 1)
cv.imshow('dilate', dilate)
closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
closing1 = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)
cv.imshow('closing',closing)
cv.imshow('closing1',closing1)

#cv.waitKey()
cv.destroyAllWindows()
'''

