import numpy as np
import cv2 as cv

def empty(a):
    pass

img = cv.imread("blue24.jpg")

assert img is not None, "file could not be read, check with os.path.exists()"
#img = cv.medianBlur(img,5)
#cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)
cv.imshow('image', img)

# redsphe.py ref
#img1 = cv.imread('imageR3.png')
hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
#lower_red = np.array([0, 50, 50])
upper_blue = np.array([122, 255, 255])
lower_blue = np.array([102, 81, 25])
mask = cv.inRange(hsv, lower_blue, upper_blue)
cv.imshow('maskimage', mask)

#mask = cv.imread("morphoblue2.jpg")
mask1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

#print(type(mask[0][0][0]))
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
cv.createTrackbar("centerDist","TrackBars",5,1000,empty)
cv.createTrackbar("param1","TrackBars",50,500,empty)
cv.createTrackbar("param2","TrackBars",30,50,empty)
cv.createTrackbar("minRadius","TrackBars",5,100,empty)
cv.createTrackbar("maxRadius","TrackBars",0,600,empty)

contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
#print((contours))

centers = []

for contour in contours:
    area = cv.contourArea(contour)
    
    if area>1500:
        cv.drawContours(mask, [contour], -1, (0, 255, 0), 2)
        M = cv.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        #centers1.append((cx))
        #centers.append((cx, cy))
        #cv.circle(img, (cx,cy), 2, (0, 0, 255), -1)
        #cv.imshow('center', img)

        if (cx-cy)<400:
              centers.append((cx,cy))
              cv.circle(img, (cx,cy), 2, (0, 0, 255), -1)
              cv.imshow('center', img)
print("SPher centers:",centers)

'''
for i in range(len(centers1)):
        for j in range(i+1, len(centers1)):
                # Calculate the difference between the i-th value and the j-th value
                diff = centers1[j] - centers1[i]
                #print('Difference between value at index', i, 'and value at index', j, ':', diff)
                if diff<400:
                      print(centers1[j], centers1[i])
                else: 
                      print('None')
            
        #centers.append((cx, cy))
        #print(centers)
        #cv.circle(img, (cx,cy), 2, (0, 0, 255), -1)
        #cv.imshow('image1', img)
'''
cv.waitKey(0)

#cv.imshow('Contours', mask)   

'''
max_contour = max(contours, key=cv.contourArea)
#print(len(max_contour))

'''
'''
M = cv.moments(max_contour)
cx = int(M['m10']/M['m00'])
cy = int(M['m01']/M['m00'])
print(cx,cy)

cv.circle(img, (cx,cy), 2, (0, 0, 255), -1)
cv.circle(img, (cx,cy), int(M['m00']//max_contour.size), (0, 255, 0), 2)
'''
'''

centers = []
m00 = []
m01 = []
m10 = []
for contour in contours:
    # Draw the contour on the original image
    #cv.drawContours(mask1, [contour], -1, (0, 255, 0), 2)
    # Calculate the center of the contour
    M = cv.moments(contour)
    M1 = cv.moments(max_contour)
    #print((M))
    #print(type(M))
    if M['m00'] != 0:
        m00.append(M['m00'])
        m01.append(M['m01'])
        m10.append(M['m10'])
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        #print(cx)
        centers.append((cx, cy))
#print(centers[0],centers[3])
    # Draw a circle at the center of the contour
#cv.circle(mask1, (centers[0]), 3, (0, 0, 255), -1)
#cv.circle(mask1, (centers[3]), 3, (0, 0, 255), -1)

#cv.imshow('image1', mask1)
cv.waitKey(0)


'''
'''
circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, 1, 50, param1=100, param2=30, minRadius=0, maxRadius=0)
print(circles)

centers = []
if circles is not None:
    circles = np.uint16(np.around(circles))
    for circle in circles[0, :]:
        # Draw the circle on the original image
        cv.circle(mask1, (circle[0], circle[1]), circle[2], (0, 255, 0), 2)
        # Calculate the center of the circle
        centers.append((circle[0], circle[1]))
        # Draw a circle at the center of the circle
        cv.circle(mask1, (circle[0], circle[1]), 3, (0, 0, 255), -1)
cv.imshow('image1', mask1)
cv.waitKey(0)
'''



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
    circles = cv.HoughCircles(closing,cv.HOUGH_GRADIENT,1,centerDist,param1=param1,param2=param2,minRadius=minRadius,maxRadius=maxRadius)
    print(circles)
    
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
    cv.imshow('detected circles',img)
    cv.waitKey()
'''