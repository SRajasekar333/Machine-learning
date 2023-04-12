import numpy as np
import cv2 as cv

def empty(a):
    pass

img = cv.imread("blue26.jpg")
cv.imshow('image', img)

assert img is not None, "file could not be read, check with os.path.exists()"
#img = cv.medianBlur(img,5)
#cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)


hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
upper_blue = np.array([122, 255, 255])
lower_blue = np.array([102, 81, 25])
mask = cv.inRange(hsv, lower_blue, upper_blue)
cv.imshow('maskimage', mask)

mask1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

#mask1 = cv.medianBlur(mask, 5)
kernel = np.ones((5,5),np.uint8)
#erode = cv.erode(mask,kernel,iterations = 1)
#cv.imshow('erode', erode)
dilate = cv.dilate(mask,kernel,iterations = 1)
#cv.imshow('dilate', dilate)

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
#cv.imshow('closing1',closing1)
#cv.imshow('closing2',closing2)


contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

centers = []

for contour in contours:
    area = cv.contourArea(contour)
    #print(area)
    if area>1000:
        cv.drawContours(mask, [contour], -1, (0, 255, 0), 2)
        M = cv.moments(contour)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        #centers.append((cx, cy))
        #cv.circle(img, (cx,cy), 2, (0, 0, 255), -1)
        #cv.imshow('center', img)
        #print(cx-cy)

        if (cx-cy)<500:
              centers.append((cx,cy))
              cv.circle(img, (cx,cy), 2, (0, 0, 255), -1)
              cv.imshow('center', img)
print("Sphere centers:",centers)
cv.waitKey(0)
