import numpy as np
from scipy.optimize import least_squares
'''
def bundle_adjustment(residuals_func, x0, observations):
    def fun(x, p):
        return np.concatenate([residuals_func(x, ob) for ob in p])

    res = least_squares(fun, x0, args=(observations,))
    return res.x

# Example usage
# -------------
# Define a simple camera model
def project(point_3d, pose, intrinsic):
    point_3d = np.array([point_3d]).T
    projection = intrinsic @ pose @ np.vstack((point_3d, [1]))
    projection = projection[:2] / projection[2]
    return projection.ravel()

# Define a simple residual function
def residuals(x, observation):
    point_3d = x[:3]
    pose = x[3:12].reshape((3, 3))
    intrinsic = x[12:].reshape((3, 3))
    proj = project(point_3d, pose, intrinsic)
    return observation - proj

# Generate some example data
point_3d = np.array([0.5, 1.0, 2.0])
pose = np.eye(3)
intrinsic = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
observations = [
    np.array([325.0, 240.0]),
    np.array([315.0, 250.0]),
    np.array([305.0, 260.0]),
    np.array([295.0, 270.0]),
    np.array([285.0, 280.0]),
    np.array([275.0, 290.0]),
]

# Call the bundle adjustment function
x0 = np.concatenate([point_3d, pose.ravel(), intrinsic.ravel()])
x_opt = bundle_adjustment(residuals, x0, observations)

# Extract the optimized parameters
point_3d_opt = x_opt[:3]
pose_opt = x_opt[3:12].reshape((3, 3))
intrinsic_opt = x_opt[12:].reshape((3, 3))

# Print the optimized parameters
print(f"Optimized 3D point: {point_3d_opt}")
print(f"Optimized pose:\n{pose_opt}")
print(f"Optimized intrinsic matrix:\n{intrinsic_opt}")

'''
import numpy as np
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define the projection function
def project_point(K, R, t, pt):
    #P = K @ np.hstack((R, t))
    #x = P @pt
    P = np.hstack((R, t))
    x = np.dot(K, np.dot(P, np.hstack((pt, 1))))
    u = x[0] / x[2]
    v = x[1] / x[2]
    return u, v

# Define the objective function to minimize
def objective_func(params, pts1, pts2, pts_3d):
    f1, cx1, cy1, k1, p1, f2, cx2, cy2, k2, p2, r11, r12, r13, r21, r22, r23, r31, r32, r33, tx, ty, tz = params
    K1 = np.array([[f1, 0, cx1], [0, f1, cy1], [0, 0, 1]])
    K2 = np.array([[f2, 0, cx2], [0, f2, cy2], [0, 0, 1]])
    dist1 = np.array([k1, k2, p1, p2, 0])
    dist2 = np.array([k1, k2, p1, p2, 0])
    R = np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]])
    t = np.array([[tx], [ty], [tz]])

    # Reproject the 3D points onto the image planes
    reprojected_pts1 = []
    reprojected_pts2 = []
    for pt in pts_3d:
        u1, v1 = project_point(K1, R, t, pt)
        u2, v2 = project_point(K2, R, t, pt)
        reprojected_pts1.append([u1, v1])
        reprojected_pts2.append([u2, v2])
    
    # Calculate the difference between the reprojected points and the observed points
    diff1 = np.array(reprojected_pts1) - pts1
    diff2 = np.array(reprojected_pts2) - pts2
    diff = np.hstack((diff1.reshape(-1), diff2.reshape(-1)))

    # Return the flattened difference array
    return diff.flatten()

# Define the reprojection error function
def reprojection_error(params, pts1, pts2, pts_3d):
    diff = objective_func(params, pts1, pts2, pts_3d)
    error = np.sqrt(np.mean(diff**2))
    return error

# Load the data
pts1 = np.array([[189, 108], [189, 217], [189, 327], [189, 435], [189, 545], [189, 645], [189, 745], [189, 845], [189, 945], [189, 1045]])
pts2 = np.array([[199, 108], [199, 217], [199, 327], [199, 435], [199, 545], [199, 645], [199, 745], [199, 845], [199, 945], [199, 1045]])
pts_3d = np.array([[0, 0, 0], [0, 1.5, 0], [0, 3, 0], [0, 4.5, 0], [0, 6, 0], [0, 7.5, 0], [0, 9, 0], [0, 10.5, 0], [0, 12, 0], [0, 13.5, 0]])

# Define the initial parameter values
f1 = 1000
cx1 = 500
cy1 = 500
k1 = 0.3
p1 = 0
f2 = 1000
cx2 = 500
cy2 = 500
k2 = 0.2
p2 = 0
r11 = 1
r12 = 0
r13 = 0
r21 = 0
r22 = 1
r23 = 0
r31 = 0
r32 = 0
r33 = 1
tx = 0
ty = 0
tz = 0
x0 = np.array([f1, cx1, cy1, k1, p1, f2, cx2, cy2, k2, p2, r11, r12, r13, r21, r22, r23, r31, r32, r33, tx, ty, tz])

result = least_squares(objective_func, x0, args=(pts1, pts2, pts_3d), method='lm')

f1, cx1, cy1, k1, p1, f2, cx2, cy2, k2, p2, r11, r12, r13, r21, r22, r23, r31, r32, r33, tx, ty, tz = result.x
print('Optimized camera parameters:')
print(f' Camera 1: f={f1}, cx={cx1}, cy={cy1}, k={k1}, p={p1}')
print(f' Camera 2: f={f2}, cx={cx2}, cy={cy2}, k={k2}, p={p2}')
print(f' Rotation matrix:\n[[{r11}, {r12}, {r13}],\n [{r21}, {r22}, {r23}],\n [{r31}, {r32}, {r33}]]')
print(f' Translation vector: [{tx}, {ty}, {tz}]')

'''
error_before = reprojection_error(x0, pts1, pts2, pts_3d)
error_after = reprojection_error(result.x, pts1, pts2, pts_3d)
print(f'Reprojection error before optimization: {error_before}')
print(f'Reprojection error after optimization: {error_after}')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts_3d[:, 0], pts_3d[:, 1], pts_3d[:, 2], c='b')
reprojected_pts1 = []
reprojected_pts2 = []
for pt in pts_3d:
    u1, v1 = project_point(np.array([[f1, 0, cx1], [0, f1, cy1], [0, 0, 1]]), np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]), np.array([[tx], [ty], [tz]]), pt)
    u2, v2 = project_point(np.array([[f2, 0, cx2], [0, f2, cy2], [0, 0, 1]]), np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]), np.array([[tx], [ty], [tz]]), pt)
    reprojected_pts1.append([u1, v1])
    reprojected_pts2.append([u2, v2])
ax.scatter(np.array(reprojected_pts1)[:, 0], np.array(reprojected_pts1)[:, 1], np.zeros(len(reprojected_pts1)), c='r')
ax.scatter(np.array(reprojected_pts2)[:, 0], np.array(reprojected_pts2)[:, 1], np.zeros(len(reprojected_pts2)), c='g')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.show()
'''

reprojected_pts1 = []
reprojected_pts2 = []
for pt in pts_3d:
    u1, v1 = project_point(np.array([[f1, 0, cx1], [0, f1, cy1], [0, 0, 1]]), np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]), np.array([[tx], [ty], [tz]]), pt)
    u2, v2 = project_point(np.array([[f2, 0, cx2], [0, f2, cy2], [0, 0, 1]]), np.array([[r11, r12, r13], [r21, r22, r23], [r31, r32, r33]]), np.array([[tx], [ty], [tz]]), pt)
    reprojected_pts1.append([u1, v1])
    reprojected_pts2.append([u2, v2])

error_before = reprojection_error(x0, pts1, pts2, pts_3d)
error_after = reprojection_error(result.x, pts1, pts2, pts_3d)
print(f'Reprojection error before optimization: {error_before}')
print(f'Reprojection error after optimization: {error_after}')

# Step 7: Visualize the results
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pts_3d[:,0], pts_3d[:,1], pts_3d[:,2])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Points')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12,6))
ax1.scatter(pts1[:,0], pts1[:,1], color='r')
ax1.scatter(reprojected_pts1[:,0], reprojected_pts1[:,1], color='b')
ax1.set_title('Camera 1')
ax2.scatter(pts2[:,0], pts2[:,1], color='r')
ax2.scatter(reprojected_pts2[:,0], reprojected_pts2[:,1], color='b')
ax2.set_title('Camera 2')
plt.show()