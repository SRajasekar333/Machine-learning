import cv2
import numpy as np
img = cv2.imread('cube.jpg')

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

cv2.imshow("blank mask image", blank_mask)

# blur the image
img_blurred = cv2.medianBlur(img, 17)
# remove noise with bilateral filter
img_noise_removed = cv2.bilateralFilter(img_blurred, 9,75,75)

# do the same preprocessing as before
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
noise_removal = cv2.bilateralFilter(img_gray, 9,75,75)
thresh_image = cv2.adaptiveThreshold(img_gray, 255,
	cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
	cv2.THRESH_BINARY, 11, 2)
# this dilate and erode section is not optimal and 
# the sizes of the kernels is the result multiple attempts
kernel = np.ones((10,1), np.uint8)
dilated_thresh_image = cv2.dilate(thresh_image, kernel, iterations=1)

kernel = np.ones((10,1), np.uint8)
dilated_thresh_image = cv2.erode(dilated_thresh_image, kernel, iterations=1)

kernel = np.ones((5,5), np.uint8)
dilated_thresh_image = cv2.erode(dilated_thresh_image, kernel, iterations=1)

kernel = np.ones((20,1), np.uint8)
dilated_thresh_image = cv2.dilate(thresh_image, kernel, iterations=1)

kernel = np.ones((25,1), np.uint8)
dilated_thresh_image = cv2.erode(dilated_thresh_image, kernel, iterations=1)

kernel = np.ones((5,5), np.uint8)
dilated_thresh_image = cv2.erode(dilated_thresh_image, kernel, iterations=1)

# invert the black and white image for the LineDetection
inverted_dilated_thresh_image = cv2.bitwise_not(dilated_thresh_image)

img2 = img.copy()

#Find all lines:

# Control the lines we want to find (minimum size and minimum distance between two lines)
minLineLength = 100
maxLineGap = 80

# Keep in mind that this is opencv 2.X not version 3 (the results of the api differ)
lines = cv2.HoughLinesP(inverted_dilated_thresh_image, 
		rho = 1,
		theta = 1 * np.pi/180,
		lines=np.array([]),
		threshold = 100,
		minLineLength = minLineLength,
		maxLineGap = maxLineGap)

# Now select the perpendiucalar lines:

# storage for the perpendiucalar lines
correct_lines = np.array([])

if lines is not None and lines.any():
	# iterate over every line
	for x1,y1,x2,y2 in lines[0]:
    
		# calculate angle in radian (if interesten in this see blog entry about arctan2)
		angle = np.arctan2(y1 - y2, x1 - x2)
        # convert to degree
		degree = abs(angle * (180 / np.pi))
		
        # only use lines with angle between 85 and 95 degrees 
		if 85 < degree < 95:
        	# draw the line on img2
			cv2.line(img2,(x1,y1),(x2,y2),(0,255,0),2)
            
            # correct upside down lines (switch lower and upper ends)
			if y1 < y2:
				temp = y2
				y2 = y1
				y1 = temp
				temp = x2
				x2 = x1
				x1 = temp
                
            # store the line 
			correct_lines = np.concatenate((correct_lines, np.array([x1,y1,x2,y2], \
               				dtype = "uint32")))				
			
            # draw the upper and lower end on img2
			cv2.circle(img2, (x1,y1), 2, (0,0,255), thickness=2, lineType=8, shift=0)
			cv2.circle(img2, (x2,y2), 2, (255,0,0), thickness=2, lineType=8, shift=0)
			cv2.imshow("image", img2)

#Connect the upper ends of lines and lower ends of lines
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
        
        # O(n^2)
        # Compare all lines with each others
        
        # only those with similar length
		if 0.9 > max(line_length, line_length_b)/min(line_length, line_length_b) > 1.1:
			continue
			dist = np.linalg.norm(np.array([ a_x1, a_y1 ]) - np.array([b_x1, b_y1]))
		
        # distance between the top points of the lines
        	
        
        # lines that are too close to eachs others (or even the same line) excluded
        # also exclude those too distant
		if 20 < dist < line_length:
			dist = np.linalg.norm(np.array([ a_x2, a_y2 ]) - np.array([b_x2, b_y2]))
        	
            # distance between lower points
        	#dist = np.linalg.norm(np.array([ a_x2, a_y2 ]) - np.array([b_x2, b_y2]))
            
            # if the lower points also match
			if 20 < dist < line_length:
            	# NOW: create the line between the uppder and lower ends
				top_lines = np.concatenate((top_lines, np.array([a_x1,a_y1,b_x1,b_y1], \
                   		dtype = "uint32")))
				angle_top_line = np.arctan2(int(a_y1) - int(b_y1), int(a_x1) - int(b_x1))
				degree_top_line = abs(angle_top_line * (180 / np.pi))

				bottom_lines = np.concatenate((bottom_lines, np.array([a_x1,a_y1,b_x1,b_y1], \
                   		dtype = "uint32")))
				angle_bottom_line = np.arctan2(int(a_y1) - int(b_y1), int(a_x1) - int(b_x1))
				degree_bottom_line = abs(angle_bottom_line * (180 / np.pi))
				
				# hack around 0 degree
				if degree_top_line == 0 or degree_bottom_line == 0:
					degree_top_line += 1
					degree_bottom_line += 1
					if 0.8 > max(degree_top_line, degree_bottom_line)/min(degree_top_line, 
                   			degree_bottom_line) > 1.2:
							print("too much difference in line degrees")
					continue
					
                # if the upper and lower connection have an equal angle 
                # they are interesting corners for a cube's face
                #if 0.8 > max(degree_top_line, degree_bottom_line)/min(degree_top_line, 
                   			#degree_bottom_line) > 1.2:
					#print("too much difference in line degrees")
					#continue
                    
                # draw the upper line and store its ends
				cv2.line(img2, (int(a_x2), int(a_y2)), (int(b_x2), int(b_y2)), (0,0,255), 1)
				upper_points = np.concatenate((upper_points, np.array([a_x2, a_y2], \
                   			dtype = "uint32")))
				upper_points = np.concatenate((upper_points, np.array([b_x2, b_y2], \
                   			dtype = "uint32")))
				
                # draw the lower line and store its ends
				cv2.line(img2, (int(a_x1), int(a_y1)), (int(b_x1), int(b_y1)), (255,0,0), 1)
				lower_points = np.concatenate((lower_points, np.array([a_x1, a_y1], \
                   			dtype = "uint32")))
				lower_points = np.concatenate((lower_points, np.array([b_x1, b_y1], \
                   			dtype = "uint32")))
                
                # store the spanned tetragon
				area = np.array([	
					int(a_x1), int(a_y1),
					int(b_x1), int(b_y1),
					int(a_x2), int(a_y2), 
					int(b_x2), int(b_y2)
				], dtype = "int32")
				areas = np.concatenate((areas, area))

#Cluster the ends of lines with DBSCAN

def centeroidnp(arr):
	# this method calculates the center of an array of points
	length = arr.shape[0]
	sum_x = np.sum(arr[:, 0])
	sum_y = np.sum(arr[:, 1])
	return sum_x/length, sum_y/length

# Promising results of the cluster algorithm
corners = np.array([])
lower_corners = np.array([])
upper_corners = np.array([])

# --------------------------------------------------
# Cluster the lower points
# --------------------------------------------------

# reshape the array to int32 matrix with two columns
vectors = np.int32(lower_points.reshape(-1, 2))

if vectors.any():

	db = DBSCAN(eps=75, min_samples=10).fit(vectors)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	# Number of clusters in labels, ignoring noise if present.
	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	# iterate over the clusters
	for i in set(db.labels_):
		if i == -1:
        	# -1 is noise
			continue
            
		color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
		index = db.labels_ == i
        
        # draw the members of the cluster
		for (point_x, point_y) in zip(vectors[index,0], vectors[index,1]):
			cv2.circle(img2,  (point_x, point_y), 5, color, thickness=1, lineType=8, shift=0)

		# calculate the centroid of the members
		cluster_center = centeroidnp(np.array(zip(np.array(vectors[index,0]),\
        				np.array(vectors[index,1]))))
        
        # draw the the cluster center
		cv2.circle(img2,  cluster_center, 5, color, thickness=10, lineType=8, shift=0)
			
        # store the centroid as corner
		corners = np.concatenate((corners, np.array([cluster_center[0], cluster_center[1]],\
        				dtype = "uint32")))
		lower_corners = np.concatenate((lower_corners, 
        				np.array([cluster_center[0], cluster_center[1]], dtype = "uint32")))
                        
# --------------------------------------------------
# Cluster the upper points
# = same as with lower points
# --------------------------------------------------

vectors = np.int32(upper_points.reshape(-1, 2))

if vectors.any():
	db = DBSCAN(eps=75, min_samples=10).fit(vectors)
	core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
	core_samples_mask[db.core_sample_indices_] = True
	labels = db.labels_

	n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

	for i in set(db.labels_):
		if i == -1:
			continue
		color = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
		index = db.labels_ == i
		for (point_x, point_y) in zip(vectors[index,0], vectors[index,1]):
			cv2.circle(img2,  (point_x, point_y), 5, color, thickness=1, lineType=8, shift=0)
		cluster_center = centeroidnp(np.array(zip(np.array(vectors[index,0]),\
        				np.array(vectors[index,1]))))
		cv2.circle(img2,  cluster_center, 5, color, thickness=10, lineType=8, shift=0)
		corners = np.concatenate((corners, np.array([cluster_center[0], cluster_center[1]], \
        				dtype = "uint32")))
		upper_corners = np.concatenate((upper_corners, \
        				np.array([cluster_center[0], cluster_center[1]], dtype = "uint32")))
