import cv2
import time

cap = cv2.VideoCapture(0)  # Set up camera stream

interval = 5  # Capture an image every 5 seconds
last_capture_time = time.time()

while True:
    ret, frame = cap.read()  # Read a frame from the camera stream
    
    elapsed_time = time.time() - last_capture_time
    
    if elapsed_time >= interval:
        cv2.imwrite(f"image_{time.time()}.jpg", frame)  # Save the image to a file
        last_capture_time = time.time()  # Reset the last capture time
    
    cv2.imshow("Camera Stream", frame)  # Display the frame
    
    if cv2.waitKey(1) == ord('q'):  # Exit the loop if 'q' is pressed
        break

cap.release()  # Release the camera stream
cv2.destroyAllWindows()  # Destroy all windows
