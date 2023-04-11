import cv2
import numpy as np

def empty(a):
    pass

# read the input image
img = cv2.imread('sphere7.jpg')

# define color range for white sphere in HSV color space
lower_white = np.array([0, 0, 200])
upper_white = np.array([180, 50, 255])

# convert image to HSV color space
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# create a binary mask for white color range
mask = cv2.inRange(hsv, lower_white, upper_white)

# apply morphological closing operation to remove noise
kernel = np.ones((5,5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("centerDist","TrackBars",10,1000,empty)
cv2.createTrackbar("param1","TrackBars",200,500,empty)
cv2.createTrackbar("param2","TrackBars",30,50,empty)
cv2.createTrackbar("minRadius","TrackBars",10,100,empty)
cv2.createTrackbar("maxRadius","TrackBars",0,600,empty)

while True:
    cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    centerDist = cv2.getTrackbarPos("centerDist","TrackBars")
    param1 = cv2.getTrackbarPos("param1", "TrackBars")
    param2 = cv2.getTrackbarPos("param2", "TrackBars")
    minRadius = cv2.getTrackbarPos("minRadius", "TrackBars")
    maxRadius = cv2.getTrackbarPos("maxRadius", "TrackBars")
    #cv.setTrackbarMin("maxRadius", "TrackBars", minRadius)
    print(centerDist,param1,param2,minRadius,maxRadius)

    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,centerDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    circles = np.uint16(np.around(circles))
    if circles.any():
        for i in circles[0,:]:
            # draw the outer circle
            cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
            # draw the center of the circle
            cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
    cv2.imshow('detected circles',cimg)
    cv2.waitKey(1)

'''
# find circles using Hough transform
circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=30, minRadius=0, maxRadius=0)

# if circles are detected, draw them on the image and find center of the connected spheres
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), r, (0, 255, 0), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 3)
    
    # find the center of the two connected spheres
    center_x = (circles[0][0] + circles[1][0]) // 2
    center_y = (circles[0][1] + circles[1][1]) // 2
    center = (center_x, center_y)
    cv2.circle(img, center, 2, (255, 0, 0), 3)

# display the output image
cv2.imshow('Output Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
