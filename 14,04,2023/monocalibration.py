import numpy as np
import cv2 as cv
import glob

objp = np.array([[0, 0, 0], [0, 54, 0]])
#objp2 = np.array([[0, 0, 0], [1, 55, 0]])

frameSize = (1280,720)
'''
coordinates = np.array([[(470,551), (736,465)],
               [(518,536), (658,249)],
               [(637,443), (377,298)],
               [(657,355), (363,228)],
               [(587,383), (415,170)],
               [(722,303), (347,296)]])
'''


coordinates1 = np.array([[470,551], [736,465]])
#coordinates = np.array(coordinates)
#print(type(coordinates))

#print(objp)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('images/monocalib/blue21.jpg')
'''
objpoints.append(objp)
objpoints.append(objp)
objpoints.append(objp)
objpoints.append(objp)
objpoints.append(objp)
objpoints.append(objp)
objpoints.append(objp2)
imgpoints.append(coordinates1)
imgpoints.append(coordinates1)
imgpoints.append(coordinates1)
imgpoints.append(coordinates1)
imgpoints.append(coordinates1)
imgpoints.append(coordinates1)
imgpoints.append(coordinates1)

'''
'''
for i, img_filename in enumerate(images):

    img = cv.imread(img_filename)
    for (x,y) in coordinates1[i]:
        radius = 2
        cv.circle(img, (x,y), radius, (0,0,255), thickness=2)
        #imgpoints.append(coordinates)
    #objpoints.append(objp)
    
    #print(objpoints)
    cv.imshow('img', img)
    cv.waitKey(1000)
#imgpoints.append(coordinates)    
imgpoints = np.array(coordinates1)
objpoints = np.array(objp)

print(objpoints)
print((imgpoints))

cv.destroyAllWindows()
'''
imgpoints = (coordinates1)
objpoints = (objp)
ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
