import numpy as np
import cv2 as cv
import glob

chessboardSize = (7,7)
frameSize = (640,480)

'''
# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
#size_of_chessboard_squares_mm = 22
objp = objp * size_of_chessboard_squares_mm


# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = sorted(glob.glob('images/stereoRight - Copy/imageR0.png'))
#images = sorted(glob.glob('images/stereoRight/*.png'))
#images = glob.glob('images/stereoLeft - 12042023_2samecam_chess/*.png')

for image in images:

    img = cv.imread(image)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, chessboardSize, None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, chessboardSize, corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(1000)

print((imgpoints))
print(objpoints)

cv.destroyAllWindows()



############## CALIBRATION #######################################################

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, frameSize, None, None)
print("INTRINSIC PARAMETERS:")

print("CameraMatrix =")
print(cameraMatrix)

'''
images = sorted(glob.glob('images/stereoRight - Copy/imageR0.png'))
'''
objp = np.array([[  0.,   0.,   0.],
       [ 20.,   0.,   0.],
       [ 40.,   0.,   0.],
       [ 60.,   0.,   0.],
       [ 80.,   0.,   0.],
       [100.,   0.,   0.],
       [120.,   0.,   0.],
       [  0.,  20.,   0.],
       [ 20.,  20.,   0.],
       [ 40.,  20.,   0.],
       [ 60.,  20.,   0.],
       [ 80.,  20.,   0.],
       [100.,  20.,   0.],
       [120.,  20.,   0.],
       [  0.,  40.,   0.],
       [ 20.,  40.,   0.],
       [ 40.,  40.,   0.],
       [ 60.,  40.,   0.],
       [ 80.,  40.,   0.],
       [100.,  40.,   0.],
       [120.,  40.,   0.],
       [  0.,  60.,   0.],
       [ 20.,  60.,   0.],
       [ 40.,  60.,   0.],
       [ 60.,  60.,   0.],
       [ 80.,  60.,   0.],
       [100.,  60.,   0.],
       [120.,  60.,   0.],
       [  0.,  80.,   0.],
       [ 20.,  80.,   0.],
       [ 40.,  80.,   0.],
       [ 60.,  80.,   0.],
       [ 80.,  80.,   0.],
       [100.,  80.,   0.],
       [120.,  80.,   0.],
       [  0.,  100.,   0.],
       [ 20.,  100.,   0.],
       [ 40.,  100.,   0.],
       [ 60.,  100.,   0.],
       [ 80.,  100.,   0.],
       [100.,  100.,   0.],
       [120.,  100.,   0.],
       [  0.,  120.,   0.],
       [ 20., 120.,   0.],
       [ 40., 120.,   0.],
       [ 60., 120.,   0.],
       [ 80., 120.,   0.],
       [100., 120.,   0.],
       [120., 120.,   0.]])
'''
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 20
#size_of_chessboard_squares_mm = 22
objp = objp * size_of_chessboard_squares_mm
objectpoints = []
imgp = np.array([[[239.81834 , 113.34875 ]],

       [[286.58395 , 113.48707 ]],

       [[333.0897  , 113.73705 ]],

       [[379.08224 , 114.077065]],

       [[424.5366  , 114.4696  ]],

       [[469.41132 , 115.04488 ]],

       [[513.9015  , 115.55703 ]],

       [[240.34534 , 161.3871  ]],

       [[287.04962 , 161.44261 ]],

       [[333.62314 , 161.5687  ]],

       [[379.77896 , 161.60509 ]],

       [[425.59183 , 161.64546 ]],

       [[470.95688 , 161.8505  ]],

       [[515.74396 , 162.31984 ]],

       [[240.51926 , 209.844   ]],

       [[287.3549  , 209.75308 ]],

       [[334.11288 , 209.66008 ]],

       [[380.3436  , 209.6059  ]],

       [[426.33817 , 209.47775 ]],

       [[472.18097 , 209.4479  ]],

       [[517.47296 , 209.23419 ]],

       [[240.66792 , 258.3047  ]],

       [[287.53598 , 258.1604  ]],

       [[334.32312 , 257.9656  ]],

       [[380.51633 , 257.66748 ]],

       [[426.6564  , 257.4484  ]],

       [[472.6087  , 257.26822 ]],

       [[518.4225  , 256.94168 ]],

       [[240.7989  , 306.66168 ]],

       [[287.61017 , 306.40372 ]],

       [[334.32867 , 306.0492  ]],

       [[380.5385  , 305.6519  ]],

       [[426.6026  , 305.44925 ]],

       [[472.699   , 305.12152 ]],

       [[518.5968  , 304.71912 ]],

       [[240.75195 , 354.98627 ]],

       [[287.57376 , 354.5718  ]],

       [[334.2786  , 354.1285  ]],

       [[380.46732 , 353.74722 ]],

       [[426.7192  , 353.53793 ]],

       [[472.5579  , 352.97595 ]],

       [[518.52313 , 352.53818 ]],

       [[240.65863 , 403.50653 ]],

       [[287.4804  , 402.61157 ]],

       [[334.20035 , 402.12622 ]],

       [[380.51038 , 401.71063 ]],

       [[426.55264 , 401.4689  ]],

       [[472.75977 , 401.19296 ]],

       [[518.54517 , 400.4533  ]]])
'''
for i, img_filename in enumerate(images):
    img = cv.imread(img_filename)
    for (x,y) in imgp[i]:
        radius = 2
        cv.circle(img, (x,y), radius, (0,0,255), thickness=2)
    cv.imshow('img', img)
    cv.waitKey(1000)
'''

objectpoints.append(objp)
cv.destroyAllWindows()
print((objectpoints))

ret, cameraMatrix, dist, rvecs, tvecs = cv.calibrateCamera(objectpoints, imgp, frameSize, None, None,flags=cv.CALIB_USE_INTRINSIC_GUESS)
