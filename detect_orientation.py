import cv2
import numpy as np
img = cv2.imread('cube.jpg')

#### ISOLATING THE CUBE ####

###Creating the mask ###

# get the dimensions of the input image
width = img.shape[1::-1][1]
height = img.shape[1::-1][0]
# create a black image with the same dimensions
blank_mask = np.zeros((width,height,3), np.uint8)
# create a white image with the same dimensions
all_white = np.ones((width,height,3), np.uint8)
all_white[:] = (255, 255, 255)
# change the colorspace to rgb
blank_mask = cv2.cvtColor(blank_mask, cv2.COLOR_RGB2GRAY)

### Add the lines ###

# blur the image
img_blurred = cv2.medianBlur(img, 17)
# remove noise with bilateral filter
img_noise_removed = cv2.bilateralFilter(img_blurred, 9,75,75)

# convert to grayscale
img_gray = cv2.cvtColor(img_noise_removed,cv2.COLOR_RGB2GRAY)
# use adaptive gaussian treshold to convert the image to black and white
thresh_image = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
				cv2.THRESH_BINARY, 11, 2)
                
# canny edge detection
canny_image = cv2.Canny(thresh_image,250,255)
# dilate to strengthen the edges
kernel = np.ones((9,9), np.uint8)
dilated_image = cv2.dilate(canny_image,kernel,iterations=1)

### Add the colored areas  ###

hsb_colors = {
	'blue':   ([ 37, 85, 75], [150, 230, 255]),
	'orange': ([  1, 155, 127], [ 38, 255, 255]),
	'green': ([ 30,  60, 70], [ 90,  180, 255]),
	'red': 	  ([  0, 151, 100], [ 30, 255, 255])
}

for key, (lower, upper) in hsb_colors.items():
		# create NumPy arrays from the boundaries
		lower = np.array(lower, dtype = "uint8")
		upper = np.array(upper, dtype = "uint8")
	 
		# find the colors within the specified boundaries and apply the mask
		mask = cv2.inRange(image_hsb, lower, upper)
		tmp_mask = cv2.bitwise_and(all_white, all_white, mask = mask)
		blank_mask = cv2.add(blank_mask, tmp_mask)

### Remove small areas ###
combined = cv2.add(dilated_image, blank_mask)

# Now dilate and erode to strengthen the cubes mask and remove lines and other noise #

# bring the lines and areas closer together (merge them)
kernel = np.ones((11,11), np.uint8)
combined_dilated = cv2.dilate(combined, kernel, iterations=1)

# remove everything smaller than 71x71
kernel = np.ones((71,71), np.uint8)
combined_eroded = cv2.erode(combined_dilated, kernel, iterations=1)

# increase everything in size that is left
kernel = np.ones((91,91), np.uint8)
combined_area = cv2.dilate(combined_eroded, kernel, iterations=1)

final_img = np.zeros((width,height,3), np.uint8)
final_img = cv2.bitwise_and(img, img, mask = combined_area)

# Show the result

cv2.namedWindow("final cube image", cv2.WINDOW_NORMAL)
cv2.imshow("final cube image", final_img)

















# lots of storage for findings
squares = np.array([])
lower_points = np.array([])
upper_points = np.array([])
top_lines = np.array([])
bottom_lines = np.array([])
areas = np.array([])


# reshape the numpy array to a matrix with four columns
correct_lines = correct_lines.reshape(-1, 4)

for a_x1, a_y1, a_x2, a_y2 in correct_lines:
	    line_length = np.linalg.norm(np.array([a_x1, a_y1])-np.array([a_x2, a_y2]))
		for b_x1, b_y1, b_x2, b_y2 in correct_lines:
                line_length_b = np.linalg.norm(np.array([b_x1, b_y1])-np.array([ b_x2, b_y2]))

				if 0.9 > max(line_length, line_length_b)/min(line_length, line_length_b) > 1.1:
					continue
				            dist = np.linalg.norm(np.array([ a_x1, a_y1 ]) - np.array([b_x1, b_y1]))


        # O(n^2)
        # Compare all lines with each others
        
        # only those with similar length
        
 
                
            # distance between the top points of the lines
            # lines that are too close to eachs others (or even the same line) excluded
            # also exclude those too distant
            if 20 < dist < line_length:
                dist = np.linalg.norm(np.array([ a_x2, a_y2 ]) - np.array([b_x2, b_y2]))
                if 20 < dist < line_length:
                    top_lines = np.concatenate((top_lines, np.array([a_x1,a_y1,b_x1,b_y1], \
                    dtype = "uint32")))
                    degree_top_line = abs(angle_top_line * (180 / np.pi))
                    bottom_lines = np.concatenate((bottom_lines, np.array([a_x1,a_y1,b_x1,b_y1], \
                    dtype = "uint32")))
                    angle_bottom_line = np.arctan2(int(a_y1) - int(b_y1), int(a_x1) - int(b_x1))
                    degree_bottom_line = abs(angle_bottom_line * (180 / np.pi))
                    if degree_top_line == 0 or degree_bottom_line == 0:
                        degree_top_line += 1
                        degree_bottom_line += 1
                        if 0.8 > max(degree_top_line, degree_bottom_line)/min(degree_top_line, degree_bottom_line) > 1.2:
                            print("too much difference in line degrees")
continue


              






  
    
    
    
    
    

    
      
      
			
      
        


cv2.line(img2, (int(a_x2), int(a_y2)), (int(b_x2), int(b_y2)), (0,0,255), 1)
upper_points = np.concatenate((upper_points, np.array([a_x2, a_y2], \
                   			dtype = "uint32")))
upper_points = np.concatenate((upper_points, np.array([b_x2, b_y2], \
                   			dtype = "uint32")))

cv2.line(img2, (int(a_x1), int(a_y1)), (int(b_x1), int(b_y1)), (255,0,0), 1)
lower_points = np.concatenate((lower_points, np.array([a_x1, a_y1], \
                   			dtype = "uint32")))
lower_points = np.concatenate((lower_points, np.array([b_x1, b_y1], \
                   			dtype = "uint32")))
    
area = np.array([	
					int(a_x1), int(a_y1),
					int(b_x1), int(b_y1),
					int(a_x2), int(a_y2), 
					int(b_x2), int(b_y2)
				], dtype = "int32")
areas = np.concatenate((areas, area))
    
  
					
  
