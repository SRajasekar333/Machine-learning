import numpy as np
import cv2
from scipy.optimize import least_squares

# Object points in 3D coordinates
obj_points = np.array([[0, 0, 0], [0, 1.5, 0], [0, 3, 0], [0, 4.5, 0], [0, 6, 0], [0, 7.5, 0], [0, 9, 0], [0, 10.5, 0], [0, 12, 0], [0, 13.5, 0]])

# Image points in 2D coordinates
img_points = np.array([[189, 108], [189, 217], [189, 327], [189, 435], [189, 545], [189, 645], [189, 745], [189, 845], [189, 945], [189, 1045]])

# Intrinsic matrix (initial guess)
fx = fy = 500  # focal length in pixels
cx = 320  # principal point x-coordinate in pixels
cy = 240  # principal point y-coordinate in pixels
K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

# Distortion coefficients (initial guess)
dist_coeffs = np.zeros((5,))

# Flatten the intrinsic matrix and distortion coefficients into a single parameter vector
params0 = np.concatenate((K.flatten(), dist_coeffs))

# Define the residual function to be minimized by least squares
def residual_func(params, obj_points, img_points):
    # Extract the intrinsic matrix and distortion coefficients from the parameter vector
    K = np.reshape(params[:9], (3, 3))
    dist_coeffs = params[9:]
    
    # Project the object points onto the image plane using the current camera parameters
    img_points_proj, _ = cv2.projectPoints(obj_points, np.zeros((3,)), np.zeros((3,)), K, dist_coeffs)
    
    # Compute the residual between the observed image points and the projected image points
    residuals = img_points_proj.squeeze() - img_points.squeeze()
    
    return residuals.flatten()

# Perform Gauss-Newton bundle adjustment to estimate the camera parameters that minimize the residual function
result = least_squares(residual_func, params0, args=(obj_points, img_points), method='lm')

# Extract the estimated intrinsic matrix and distortion coefficients from the optimized parameter vector
K_est = np.reshape(result.x[:9], (3, 3))
dist_coeffs_est = result.x[9:]

# Print the estimated camera parameters
print("Intrinsic matrix (estimated):\n", K_est)
print("Distortion coefficients (estimated):\n", dist_coeffs_est)

