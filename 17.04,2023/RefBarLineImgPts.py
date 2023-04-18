import cv2
import numpy as np

# Define the two points
point1 = (657, 355)
point2 = (363, 228)

# Create a black image
img = cv2.imread("blue24.jpg")
cv2.circle(img, point1, 4, (0, 0, 255), -1)
cv2.circle(img, point2, 4, (0, 0, 255), -1)

# Draw a line between the two points
cv2.line(img, point1, point2, (0, 255, 0), 2)

# Calculate the distance between the two points
distance = np.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

# Calculate the distance between each of the four equidistant points
delta = distance / 9

# Calculate the x and y increments between each equidistant point
delta_x = (point2[0] - point1[0]) / 9
delta_y = (point2[1] - point1[1]) / 9

# Draw circles at each of the four equidistant points
for i in range(1, 9):
    x = int(point1[0] + i * delta_x)
    y = int(point1[1] + i * delta_y)
    cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    print(x,y)

# Display the image
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
