import numpy as np
import cv2 as cv

def empty(a):
    pass

#img = cv.imread("images/stereoLeft(J1) - 17042023_2samcam_sphere/imageL(60)18.png")
img = cv.imread("images/stereoLeft(H) - 17042023_2diffcam_sphere/sph/imageR(85)33.png")
#img = cv.imread("images/L20/imageL(100)39.png")
#img = cv.imread("images/R20/imageR(100)39.png")

#img = cv.imread("images/stereoRight(J) - 17042023_2samcam_sphere/imageR(60)18.png")
#img = cv.imread("images/stereoRight(J) - 17042023_2diffcam_sphere/imageR(50)19.png")


cv.imshow('image', img)

assert img is not None, "file could not be read, check with os.path.exists()"
#img = cv.medianBlur(img,5)
#cimg = cv.cvtColor(img,cv.COLOR_GRAY2BGR)


hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
upper_blue = np.array([130, 255, 255])
lower_blue = np.array([90, 70, 0])
mask = cv.inRange(hsv, lower_blue, upper_blue)
cv.imshow('maskimage', mask)
#cv.imwrite('maskimage', mask)

mask1 = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
mask2 = cv.cvtColor(mask, cv.COLOR_GRAY2BGR)

#mask1 = cv.medianBlur(mask, 5)
kernel = np.ones((5,5),np.uint8)
erode = cv.erode(mask,kernel,iterations = 1)
cv.imshow('erodeimage', erode)
dilate = cv.dilate(mask,kernel,iterations = 1)
cv.imshow('dilateimage', dilate)
#cv.imwrite('dilateimage', dilate)

#thresh = cv.threshold(img, 220, 255, cv.THRESH_BINARY)[1]
#cv.imshow(' circles',thresh)

# apply morphological closing operation
kernel = np.ones((5,5),np.uint8)
#closing = cv.morphologyEx(thresh, cv.MORPH_CLOSE, kernel)
closing = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
closing1 = cv.morphologyEx(dilate, cv.MORPH_CLOSE, kernel)
#mask1 = cv.medianBlur(closing1, 5)
closing2 = cv.morphologyEx(erode, cv.MORPH_CLOSE, kernel)
result1 = cv.bitwise_and(img, img, mask=mask)

cv.imshow('maskclosingimage',closing)
cv.imshow('dilateclosingimage',closing1)
cv.imshow('biwise',result1)

cv.imshow('erodeclosingimage',closing2)


contours, hierarchy = cv.findContours(closing1, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

centers = []

for contour in contours:
    area = cv.contourArea(contour)
    #print(area)
    if area>1000:
        cv.drawContours(mask, [contour], -1, (0, 255, 0), 2)
        M = cv.moments(contour)
        print(M)
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
