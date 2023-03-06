import cv2
import time

# Define the camera IDs
cam1_id = 0
cam2_id = 1

# Create the video capture objects
cam1 = cv2.VideoCapture(cam1_id)
cam2 = cv2.VideoCapture(cam2_id)

# Set the resolution of the cameras
cam1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cam2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cam2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Define the synchronization interval
sync_interval = 10  # in milliseconds

# Capture images from both cameras
while True:
    # Capture an image from camera 1
    ret1, img1 = cam1.read()
    if not ret1:
        break
    
    # Capture an image from camera 2
    ret2, img2 = cam2.read()
    if not ret2:
        break
    
    # Get the current time in milliseconds
    timestamp = int(time.time() * 1000)
    
    # Display the synchronized images
    cv2.imshow("Camera 1", img1)
    cv2.imshow("Camera 2", img2)
    
    # Wait for the synchronization interval
    key = cv2.waitKey(sync_interval)
    if key == ord('q'):
        break

# Release the video capture objects
cam1.release()
cam2.release()

# Close all windows
cv2.destroyAllWindows()
