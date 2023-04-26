import numpy as np
import cv2

# Define the origin and orientation of the world coordinate system
world_origin = np.array([0, 0, 0])
world_x_axis = np.array([1, 0, 0])
world_y_axis = np.array([0, 1, 0])
world_z_axis = np.array([0, 0, 1])

# Extract the pixel coordinates of the object points from the left camera image
#left_img = cv2.imread('left.jpg')
left_obj_pts = np.array([[120, 299], [492, 314]])
right_obj_pts = np.array([[88, 149], [471, 181]])

# Use stereo correspondence to find the corresponding pixel coordinates in the right camera image
# ...

# Triangulate the object points in 3D space
'''
proj_matrix_left = np.array([[786, 0, 311, 0],
                             [0, 854, 170, 0],
                             [0, 0, 1, 0]], dtype=np.float32)
proj_matrix_right = np.array([[788, 0, 204, 5],
                              [0, 789, 204, 0],
                              [0, 0, 1, 0]], dtype=np.float32)
'''
proj_matrix_left = np.array([[3.93, 0, 1.55, 0],
                             [0, 4.27, 0.85, 0],
                             [0, 0, 1, 0]], dtype=np.float32)
proj_matrix_right = np.array([[3.94, 0, 1.02, 5],
                              [0, 3.95, 1.02, 0],
                              [0, 0, 1, 0]], dtype=np.float32)
obj_pts_3d_homogeneous = cv2.triangulatePoints(proj_matrix_left, proj_matrix_right, left_obj_pts, right_obj_pts)
print(obj_pts_3d_homogeneous)
# Transform the 3D coordinates to the world coordinate system
#decomposition = cv2.decomposeProjectionMatrix(proj_matrix_left)
_, _, translation_left, rotation_left, _, _, _ = cv2.decomposeProjectionMatrix(proj_matrix_left)

#translation_left = decomposition[0][:3, 0]
#print(translation_left)
#rotation_left = decomposition[1][:3, :3]
print(rotation_left)

R_left = rotation_left[:3, :3]
T_left = translation_left[:3]
M = np.concatenate((R_left, T_left.reshape(-1, 1)), axis=1)
M = np.vstack((M, np.array([0, 0, 0, 1])))
#print(M)

print(R_left)
#print(T_left.reshape(-1, 1))

obj_pts_3d_world_homogeneous = np.dot(M.T, obj_pts_3d_homogeneous) + translation_left.reshape(-1, 1)

# Remove the homogeneous coordinate
obj_pts_3d_world = obj_pts_3d_world_homogeneous[:3] / obj_pts_3d_world_homogeneous[3]
print(obj_pts_3d_world)
print(obj_pts_3d_world_homogeneous)