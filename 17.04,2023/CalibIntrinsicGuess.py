import numpy as np
import cv2

# define object points and image points
#objp = np.array([[0,0,0], [1,0,0], [0,1,0], [1,1,0], [0,0,-1], [1,0,-1], [0,1,-1], [1,1,-1]], dtype=np.float32)
#imgp = np.array([[10, 20], [30, 20], [10, 40], [30, 40], [15, 10], [25, 10], [15, 50], [25, 50]], dtype=np.float32)

objp = np.array([[0, 0, 0], [60, 0, 0], [120, 0, 0], [180, 0, 0], [240, 0, 0], [300, 0, 0], [360, 0, 0], [420, 0, 0], [480, 0, 0], [540, 0, 0]], dtype=np.float32)
imgp = np.array([[363,228], [395, 242], [428, 256], [461, 270], [493, 284], [526, 298], [559, 312], [591, 326], [624, 340], [657, 355]], dtype=np.float32)

# define image size
img_size = (640, 480)

# calculate initial estimate of camera matrix
fx = fy = 500 # guess for focal length
cx = img_size[0] / 2 # guess for principal point
cy = img_size[1] / 2
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

# calibrate camera
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera([objp], [imgp], img_size, camera_matrix, None,flags=cv2.CALIB_USE_INTRINSIC_GUESS)
print("Camera Matrix:")
print(mtx)
print("\nDistortion Coefficients:")
print(dist)