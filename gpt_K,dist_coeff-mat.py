import cv2
import numpy as np

# Define the size of the calibration pattern
pattern_size = (num_cols, num_rows)

# Prepare the object points
objp = np.zeros((num_cols*num_rows, 3), np.float32)
objp[:, :2] = np.mgrid[0:num_cols, 0:num_rows].T.reshape(-1, 2) * square_size

# Prepare the image points
obj_points = []
img_points = []

for img_file in calibration_image_files:
    img = cv2.imread(img_file)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    if ret:
        obj_points.append(objp)
        img_points.append(corners)

# Calculate the camera matrix and distortion coefficients
ret, K, dist_coef, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

# Save the camera matrix and distortion coefficients
np.savetxt('camera_matrix.txt', K)
np.savetxt('distortion_coefficients.txt', dist_coef)


# replace num_cols, num_rows, square_size, and calibration_image_files