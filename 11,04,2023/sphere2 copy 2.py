import numpy as np
import cv2 as cv

def empty(a):
    pass

img = cv.imread("blue4.jpg")

assert img is not None, "file could not be read, check with os.path.exists()"
#img = cv.medianBlur(img,5)
#cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
cv.imshow('image', img)

# redsphe.py ref
#img1 = cv.imread('imageR3.png')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#lower_red = np.array([0, 50, 50])
upper_blue = np.array([140, 255, 255])
lower_blue = np.array([66, 67, 40])
mask = cv.inRange(hsv, lower_blue, upper_blue)
cv.imshow('maskimage', mask)

#mask1 = cv.medianBlur(mask, 5)
kernel = np.ones((5,5),np.uint8)
#erode = cv.erode(mask,kernel,iterations = 1)
#cv.imshow('erode', erode)
dilate = cv.dilate(mask,kernel,iterations = 1)
cv.imshow('dilate', dilate)
# redsphe.py ref

#thresh = cv.threshold(img, 220, 255, cv.THRESH_BINARY)[1]
#cv.imshow(' circles',thresh)

# apply morphological closing operation
kernel = np.ones((5,5),np.uint8)
#closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
closing1 = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)
#mask1 = cv.medianBlur(closing1, 5)
#closing2 = cv.morphologyEx(closing1, cv.MORPH_CLOSE, kernel)

cv.imshow('closing',closing)
cv.imshow('closing1',closing1)
#cv.imshow('closing2',closing2)


cv.namedWindow("TrackBars")
cv.resizeWindow("TrackBars",640,240)
cv.createTrackbar("centerDist","TrackBars",10,1000,empty)
cv.createTrackbar("param1","TrackBars",200,500,empty)
cv.createTrackbar("param2","TrackBars",30,50,empty)
cv.createTrackbar("minRadius","TrackBars",10,100,empty)
cv.createTrackbar("maxRadius","TrackBars",0,600,empty)

contours, hierarchy = cv.findContours(closing, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#print(contours)

max_contour = max(contours, key=cv.contourArea)
#print(max_contour)

M = cv.moments(max_contour)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print(cx,cy)


cv.circle(img, (cx,cy), 2, (0, 0, 255), -1)
cv.circle(img, (cx,cy), int(M['m00']//max_contour.size), (0, 255, 0), 2)

cv.imshow('image1', img)
cv.waitKey(0)

'''
while True:
    #cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
    centerDist = cv.getTrackbarPos("centerDist","TrackBars")
    param1 = cv.getTrackbarPos("param1", "TrackBars")
    param2 = cv.getTrackbarPos("param2", "TrackBars")
    minRadius = cv.getTrackbarPos("minRadius", "TrackBars")
    maxRadius = cv.getTrackbarPos("maxRadius", "TrackBars")
    #cv.setTrackbarMin("maxRadius", "TrackBars", minRadius)
    print(centerDist,param1,param2,minRadius,maxRadius)

    
    #circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,centerDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    circles = cv.HoughCircles(img,cv.HOUGH_GRADIENT,1,centerDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    
    circles = np.uint16(np.around(circles))
    if circles.any():
        for i in circles[0,:]:
            # draw the outer circle
            #cv.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            cv.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            #cv.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
            cv.circle(img,(i[0],i[1]),2,(0,0,255),3)
    
    
    #cv.imshow('detected circles',cimg)
    #cv.imshow('detected circles',img)
    #cv.waitKey()
'''