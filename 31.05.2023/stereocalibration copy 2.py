import numpy as np
import cv2 as cv
import glob


################ FIND CHESSBOARD CORNERS - OBJECT POINTS AND IMAGE POINTS #############################

#chessboardSize = (9,6)
chessboardSize = (7,7)

frameSize = (640,480)


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((chessboardSize[0] * chessboardSize[1], 3), np.float32)
objp[:,:2] = np.mgrid[0:chessboardSize[0],0:chessboardSize[1]].T.reshape(-1,2)

size_of_chessboard_squares_mm = 22
objp = objp * size_of_chessboard_squares_mm
#print (objp)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpointsL = [] # 2d points in image plane.
imgpointsR = [] # 2d points in image plane.


#imagesLeft = sorted(glob.glob('images/stereoLeft - Copy/*.png'))
imagesLeft = sorted(glob.glob('images/stereoimages20042023/stereoLeft(H) - 17042023_2diffcam_chess/07/*.png'))
#imagesLeft = sorted(glob.glob('images/stereoimages20042023/stereoLeft(J1) - 17042023_2samcam_chess/46/*.png'))

#imagesRight = sorted(glob.glob('images/stereoRight - Copy/*.png'))
imagesRight = sorted(glob.glob('images/stereoimages20042023/stereoRight(J) - 17042023_2diffcam_chess/07/*.png'))
#imagesRight = sorted(glob.glob('images/stereoimages20042023/stereoRight(J) - 17042023_2samcam_chess/46/*.png'))

num = 0
for imgLeft, imgRight in zip(imagesLeft, imagesRight):

    imgL = cv.imread(imgLeft)
    imgR = cv.imread(imgRight)
    grayL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    grayR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Find the chess board corners
    retL, cornersL = cv.findChessboardCorners(grayL, chessboardSize, None)
    retR, cornersR = cv.findChessboardCorners(grayR, chessboardSize, None)
    #print(retR, cornersR)

    # If found, add object points, image points (after refining them)
    if retL and retR == True:

        objpoints.append(objp)
        #print(objpoints)

        cornersL = cv.cornerSubPix(grayL, cornersL, (5,5), (-1,-1), criteria)
        imgpointsL.append(cornersL)
        #print(cornersL)
        #print("Detected center:", imgpointsL[0])

        cornersR = cv.cornerSubPix(grayR, cornersR, (5,5), (-1,-1), criteria)
        imgpointsR.append(cornersR)
        #print(cornersR)
        #print("Detected center1:", imgpointsR)

        # Draw and display the corners
        cv.drawChessboardCorners(imgL, chessboardSize, cornersL, retL)
        cv.imshow('img left', imgL)
        cv.drawChessboardCorners(imgR, chessboardSize, cornersR, retR)
        cv.imshow('img right', imgR)
        cv.waitKey(1000)
        cv.imwrite('images/stereoimages20042023/stereoLeft(H) - 17042023_2diffcam_chess/07/drawimageL' + str(num) + '.png', imgL)
        cv.imwrite('images/stereoimages20042023/stereoRight(J) - 17042023_2diffcam_chess/07/drawimageR' + str(num) + '.png', imgR)

        num += 1



cv.destroyAllWindows()




############## CALIBRATION #######################################################

retL, cameraMatrixL, distL, rvecsL, tvecsL = cv.calibrateCamera(objpoints, imgpointsL, frameSize, None, None)
heightL, widthL, channelsL = imgL.shape
newCameraMatrixL, roi_L = cv.getOptimalNewCameraMatrix(cameraMatrixL, distL, (widthL, heightL), 1, (widthL, heightL))
#print(roi_L)

print("INTRINSIC PARAMETERS:")
print("UndistortedCmaeraMatrixLeft=")
print(cameraMatrixL)


retR, cameraMatrixR, distR, rvecsR, tvecsR = cv.calibrateCamera(objpoints, imgpointsR, frameSize, None, None)
heightR, widthR, channelsR = imgR.shape
newCameraMatrixR, roi_R = cv.getOptimalNewCameraMatrix(cameraMatrixR, distR, (widthR, heightR), 1, (widthR, heightR))
print("UndistortedCmaeraMatrixRight=")
print(cameraMatrixR)
print("CameraMatrixLeft =")
print(newCameraMatrixL)
print("CameraMatrixRight =")
print(newCameraMatrixR)


########## Stereo Vision Calibration #############################################

flags = 0
flags |= cv.CALIB_FIX_INTRINSIC
# Here we fix the intrinsic camara matrixes so that only Rot, Trns, Emat and Fmat are calculated.
# Hence intrinsic parameters are the same 

criteria_stereo= (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# This step is performed to transformation between the two cameras and calculate Essential and Fundamenatl matrix
retStereo, newCameraMatrixL, distL, newCameraMatrixR, distR, rot, trans, essentialMatrix, fundamentalMatrix = cv.stereoCalibrate(objpoints, imgpointsL, imgpointsR, newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], criteria_stereo, flags)
#print(essentialMatrix, fundamentalMatrix)

print("EXTRINSIC PARAMETERS:")
print("Rotation =")
print(rot)
print("Translation =")
print(trans)
Mext = np.c_[rot, trans]
print("Mext")
print(Mext)


########## Stereo Rectification #################################################

rectifyScale= 1
rectL, rectR, projMatrixL, projMatrixR, Q, roi_L, roi_R= cv.stereoRectify(newCameraMatrixL, distL, newCameraMatrixR, distR, grayL.shape[::-1], rot, trans, rectifyScale,(0,0))

stereoMapL = cv.initUndistortRectifyMap(newCameraMatrixL, distL, rectL, projMatrixL, grayL.shape[::-1], cv.CV_16SC2)
stereoMapR = cv.initUndistortRectifyMap(newCameraMatrixR, distR, rectR, projMatrixR, grayR.shape[::-1], cv.CV_16SC2)

print("Saving parameters!")
cv_file = cv.FileStorage('stereoMap.xml', cv.FILE_STORAGE_WRITE)

cv_file.write('stereoMapL_x',stereoMapL[0])
cv_file.write('stereoMapL_y',stereoMapL[1])
cv_file.write('stereoMapR_x',stereoMapR[0])
cv_file.write('stereoMapR_y',stereoMapR[1])

cv_file.release()

