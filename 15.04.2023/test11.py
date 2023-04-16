import numpy as np
from scipy.optimize import least_squares

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
