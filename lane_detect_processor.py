################################################################################
#   Date:    08.24.2017
#   Author:  Nathan Greco (Nathan.Greco@gmail.com)
#
#   Project:
#       CarND-Vehicle-Detection: Final project in Term 1 of Self-Driving
#           Car Nanodegree curriculum.
#
#   Module:
#       lane_detect_processor: Contains tools necessary for detecting road
#           lanes in a given image.
#
#   Repository:
#       http://github.com/nategreco/CarND-Vehicle-Detection.git
#
#   License:
#       Part of Udacity Self-Driving Car Nanodegree curriculum.
#
#   Notes:
#       Following google style guide here:
#       https://google.github.io/styleguide/pyguide.html
#      
################################################################################

#System imports
import numpy as np
import matplotlib.pyplot as plt

#3rd party imports
import cv2

#Local project imports

#Constants
ROI_SF = np.array([[(0.00, 0.95), \
                    #(0.10, 0.80), \
                    (0.43, 0.62), \
                    (0.57, 0.62), \
                    #(0.90, 0.80), \
                    (1.00, 0.95)]], \
                  dtype=np.float64)
WHITE_LOWER_THRESH_1 = np.array([0, 180, 0])
WHITE_UPPER_THRESH_1 = np.array([180, 255, 255])
WHITE_LOWER_THRESH_2 = np.array([180, 255, 255]) #Not used
WHITE_UPPER_THRESH_2 = np.array([180, 255, 255]) #Not used
YELLOW_LOWER_THRESH = np.array([10, 10, 55])
YELLOW_UPPER_THRESH = np.array([50, 255, 255])
SOBEL_X_LOWER_THRESH = 20
SOBEL_X_UPPER_THRESH = 255
SOBEL_Y_LOWER_THRESH = 255 #Not used
SOBEL_Y_UPPER_THRESH = 255 #Not used
#Calculate the birds eye view matrix transformation
SRC = np.float32([[582, 460], \
                  [698, 460], \
                  [185, 720], \
                  [1095, 720]])
DST = np.float32([[185, 0], \
                  [1095, 0], \
                  [185, 720], \
                  [1095, 720]])
BEV_MATRIX = cv2.getPerspectiveTransform(SRC, DST)
INV_MATRIX = cv2.getPerspectiveTransform(DST, SRC)
#Detect lines
NUM_WINDOWS = 15
WINDOW_WIDTH = 250
MARGIN = 50 #Convolutional only, not used
MIN_PIXELS = 50
MIN_WIDTH_PIX = 740
MAX_WIDTH_PIX = 980
#Shade lines
LINE_THICKNESS = 10
#Status
PIXEL_TO_M_X = 3.7 / 894 #3.7m / 894 pixels
PIXEL_TO_M_Y = 30 / 720 #30m / 720 pixels

#Classes
class Line():
    """
    This i sthe line class used by both the left and right line.  The class
    keeps a defined number of frames and creates a best fit 2nd order
    polynomial.
    """
    def __init__(self, num_to_keep):
        #Number of lines to average
        self.num_to_keep = num_to_keep  
        #Line detection status
        self.detected = False  
        #Polynomial coefficients for the most recent fit
        self.current_fit = np.ndarray(shape=(0,0), dtype=float)
        #Previoius points
        self.prev_pts = []
    def fit_line(self):
        pts = [item for sublist in self.prev_pts for item in sublist]
        if pts:
            y = [i[0] for i in pts]
            x = [i[1] for i in pts]
            self.current_fit = np.polyfit(y, x, 2)
        else:
            self.current_fit = np.ndarray(shape=(0,0), dtype=float)
    def update(self, y, x):
        self.prev_pts.append(list(zip(y, x)))
        if len(self.prev_pts) > self.num_to_keep:
            del self.prev_pts[0]
        if x.size != 0:
            self.detected = True
        else:
            self.detected = False
        self.fit_line()

#Functions
def extract_roi(image):
    """
    Extracts only the region of interest in the image, which is defined
    by a scale factor polygon.  This removes extra noise in the image.
    """
    #Create blank mask
    mask = np.zeros_like(image)   
    #Fill based on number of channels
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #Scale ROI vertices
    vertices = np.copy(ROI_SF)
    for i in range(0, len(vertices[0])):
        vertices[0][i][0] *= float(image.shape[1])
        vertices[0][i][1] *= float(image.shape[0])
    vertices = vertices.astype(int)
    #Fill polygon created by roi vertices
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def trim_roi(image, pixels):
    """
    This function trims the image's ROI to reduce any false gradients
    created by the edge of the ROI.
    """
    #Create blank mask
    mask = np.zeros_like(image)   
    #Fill based on number of channels
    if len(image.shape) > 2:
        channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
    #Scale ROI vertices
    vertices = np.copy(ROI_SF)
    for i in range(0, len(vertices[0])):
        vertices[0][i][0] *= float(image.shape[1])
        vertices[0][i][1] *= float(image.shape[0])
    vertices = vertices.astype(int)
    #Fill polygon created by roi vertices
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    #Erode the mask by pixel count
    assert(pixels % 2 != 0) #Check for odd kernel size
    kernel = np.ones((pixels,pixels), np.uint8)
    mask = cv2.erode(mask, kernel)
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image

def calibrate_camera(images, x_pts, y_pts):
    """
    This function uses a list of checkerboard images to develop a transformation
    matrix for undistoring all future images.
    """
    #Assert all images are same shape
    assert(all(i.shape == images[0].shape) for i in images)
    #Prepare variables
    objp = np.zeros((y_pts * x_pts, 3), np.float32)
    objp[:,:2] = np.mgrid[0:x_pts, 0:y_pts].T.reshape(-1,2)
    objpoints = []
    imgpoints = []
    #Iterate through each image
    success_count = 0
    for image in images:
        #Change color
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        #Find corners
        ret, corners = cv2.findChessboardCorners(gray, (x_pts, y_pts), None)
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)
            success_count += 1
    #Check for succesful checkerboard detections
    assert(success_count != 0)
    print(success_count, " out of ", len(images), \
        " checkerboards detected, Calibration complete!")
    #Get matrix
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera( objpoints, \
        imgpoints, \
        gray.shape[::-1], None, None )
    return mtx, dist

def hls_threshold(image):
    """
    This function performs both a white and yellow threshold to extract only
    white and yellow road markings.  It returns a binary masked so it can then
    be compared to the gradient mask.    
    """
    hls = cv2.cvtColor(cv2.blur(image, (3,3)), cv2.COLOR_BGR2HLS)
    #White threshold
    white_binary_1 = cv2.inRange(hls, WHITE_LOWER_THRESH_1, WHITE_UPPER_THRESH_1)
    white_binary_2 = cv2.inRange(hls, WHITE_LOWER_THRESH_2, WHITE_UPPER_THRESH_2)
    white_binary = cv2.bitwise_or(white_binary_1, white_binary_2)
    #Yellow_threshold
    yellow_binary = cv2.inRange(hls, YELLOW_LOWER_THRESH, YELLOW_UPPER_THRESH)
    #Combine masks
    mask = cv2.bitwise_or(white_binary, yellow_binary)
    return mask

def gradient_threshold(image):
    """
    This function performs both an x and y sobel function then compares this
    against a threshold to highlight edges in images that are potential road
    markings.
    """
    #Convert to grayscale and blur
    gray = cv2.cvtColor(cv2.blur(image, (3,3)), cv2.COLOR_BGR2GRAY)
    #Get derivative with respect to x
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    abs_sobel_x = np.absolute(sobel_x)
    scaled_sobel_x = np.uint8(255 * abs_sobel_x / np.max(abs_sobel_x))
    #Threshold
    sxbinary = cv2.inRange(scaled_sobel_x, \
                           SOBEL_X_LOWER_THRESH, \
                           SOBEL_X_UPPER_THRESH)
    #Get derivative with respect to y - maybe unecessary?
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    abs_sobel_y = np.absolute(sobel_y)
    scaled_sobel_y = np.uint8(255 * abs_sobel_y / np.max(abs_sobel_y))
    #Threshold
    sybinary = cv2.inRange(scaled_sobel_y, \
                           SOBEL_Y_LOWER_THRESH, \
                           SOBEL_Y_UPPER_THRESH)
    #Combine masks
    mask = cv2.bitwise_or(sxbinary, sybinary)
    return mask

def detect_lines_basic(image, left_line, right_line):
    """
    This function uses a sliding window to follow the most amount of pixels,
    which are then handed to the Line class for updating of the polynomial.
    """
    #Take a histogram of the bottom half of the image
    histogram = np.sum(image[image.shape[0] // 2:,:], axis=0)
    #Sliding window methodology
    #Create an output image to draw on and  visualize the result
    output_image = np.dstack((image, image, image)) * 255
    window_height = np.int(image.shape[0] / NUM_WINDOWS)
    #Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    #Create startpoints if no valid polynomial fits
    if not (len(left_line.current_fit) == 3 & \
            len(right_line.current_fit) == 3):
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_current = np.argmax(histogram[:midpoint])
        rightx_current = np.argmax(histogram[midpoint:]) + midpoint
        #Check that lane width OK
        if (rightx_current - leftx_current) < MIN_WIDTH_PIX:
            #Rely on one with most good pixels
            if histogram[leftx_current] > histogram[rightx_current]:
                rightx_current = leftx_current + MIN_WIDTH_PIX
            else:
                leftx_current = rightx_current - MIN_WIDTH_PIX
        elif (rightx_current - leftx_current) > MAX_WIDTH_PIX:
            #Rely on one with most good pixels
            if histogram[leftx_current] > histogram[rightx_current]:
                rightx_current = leftx_current + MAX_WIDTH_PIX
            else:
                leftx_current = rightx_current - MAX_WIDTH_PIX
    #Otherwise use max y position of current polynomial fits
    else:
        leftx_current = left_line.current_fit[0] * image.shape[0]**2 + \
                        left_line.current_fit[1] * image.shape[0] + \
                        left_line.current_fit[2]
        rightx_current = right_line.current_fit[0] * image.shape[0]**2 + \
                         right_line.current_fit[1] * image.shape[0] + \
                         right_line.current_fit[2]
    #Update width
    width_current = rightx_current - leftx_current
    #Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []
    #Step through the windows one by one
    for window in range(NUM_WINDOWS):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window + 1) * window_height
        win_y_high = image.shape[0] - window * window_height
        win_xleft_low = int(leftx_current - WINDOW_WIDTH / 2)
        win_xleft_high = int(leftx_current + WINDOW_WIDTH / 2)
        win_xright_low = int(rightx_current - WINDOW_WIDTH / 2)
        win_xright_high = int(rightx_current + WINDOW_WIDTH / 2)
        #Draw the windows on the visualization image
        cv2.rectangle(output_image, \
                      (win_xleft_low, win_y_low), \
                      (win_xleft_high, win_y_high), \
                      (0, 255, 0), \
                      2)
        cv2.rectangle(output_image, \
                      (win_xright_low, win_y_low), \
                      (win_xright_high, win_y_high),\
                      (0, 255, 0), \
                      2)
        #Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & \
                          (nonzeroy < win_y_high) & \
                          (nonzerox >= win_xleft_low) \
                          & (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & \
                           (nonzeroy < win_y_high) & \
                           (nonzerox >= win_xright_low) & \
                           (nonzerox < win_xright_high)).nonzero()[0]
        #Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        #If you found > MIN_PIXELS, recenter next window on their mean position
        if (len(good_left_inds) > MIN_PIXELS) & \
           (len(good_right_inds) > MIN_PIXELS):
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
        elif len(good_left_inds) > MIN_PIXELS:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            rightx_current = leftx_current + width_current
        elif len(good_right_inds) > MIN_PIXELS:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            leftx_current = rightx_current - width_current
        #Check that lane width OK
        if (rightx_current - leftx_current) < MIN_WIDTH_PIX:
            #Rely on one with most good pixels
            if len(good_left_inds) > len(good_right_inds):
                rightx_current = leftx_current + MIN_WIDTH_PIX
            else:
                leftx_current = rightx_current - MIN_WIDTH_PIX
        elif (rightx_current - leftx_current) > MAX_WIDTH_PIX:
            #Rely on one with most good pixels
            if len(good_left_inds) > len(good_right_inds):
                rightx_current = leftx_current + MAX_WIDTH_PIX
            else:
                leftx_current = rightx_current - MAX_WIDTH_PIX
        #Update width
        width_current = rightx_current - leftx_current
    #Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)
    #Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 
    #Fit a second order polynomial to each
    left_line.update(lefty, leftx)
    right_line.update(righty, rightx)
    return output_image

def detect_lines_convolution(image, left_line, right_line): #Not used
    """
    This function uses convolution methodology to find highest 
    concentrations of pixels, which centerpoints are then handed to the Line
    class for updating of the polynomial.
    """
    #Store centroid postions per each y position
    left_centroids = []
    right_centroids = []
    window = np.ones(WINDOW_WIDTH)
    window_height = np.int(image.shape[0] / NUM_WINDOWS)
    #Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(image[int(3 * image.shape[0] / 4):, \
                         :int(image.shape[1] / 2)], axis=0)
    l_center = int(np.argmax(np.convolve(window, l_sum)) - \
               WINDOW_WIDTH / 2)
    r_sum = np.sum(image[int(3 * image.shape[0] / 4):, \
                         int(image.shape[1]/2):], axis=0)
    r_center = int(np.argmax(np.convolve(window,r_sum)) - \
               WINDOW_WIDTH / 2 + \
               int(image.shape[1]/2))
    #Add what we found for the first layer
    left_centroids.append([l_center, 0])
    right_centroids.append([r_center, 0])

    #Go through each layer looking for max pixel locations
    for level in range(1, int(image.shape[0]/ window_height)):
        #Convolve the window into the vertical slice of the image
        image_layer = np.sum(image[int(image.shape[0] - \
            (level + 1) * window_height):int(image.shape[0] - \
            level * window_height), :], axis=0)
        conv_signal = np.convolve(window, image_layer)
        #Find the best left centroid by using past left center as a reference
        offset = WINDOW_WIDTH / 2
        l_min_index = int(max(l_center + offset - MARGIN, 0))
        l_max_index = int(min(l_center + offset + MARGIN, image.shape[1]))
        l_center = int(np.argmax(conv_signal[l_min_index:l_max_index]) + \
                       l_min_index - offset)
        #Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center + offset - MARGIN, 0))
        r_max_index = int(min(r_center + offset + MARGIN, image.shape[1]))
        r_center = int(np.argmax(conv_signal[r_min_index:r_max_index]) + \
                       r_min_index - offset)
        #Add what we found for that layer
        left_centroids.append([l_center, int((level + 0.5) * window_height)])
        right_centroids.append([r_center, int((level + 0.5) * window_height)])
    #Fit a second order polynomial to each
    lefty = np.array([i[1] for i in left_centroids])
    leftx = np.array([i[0] for i in left_centroids])
    left_line.update(lefty, leftx)
    righty = np.array([i[1] for i in right_centroids])
    rightx = np.array([i[0] for i in right_centroids])
    right_line.update(righty, rightx)    
    #Need to visualize data on output image - TODO
    output_image = image
    return output_image

def combine_images(bottom, top):
    """
    This function lays all nonzero pixels of the top image on the bottom image.
    """
    assert(bottom.shape == top.shape)
    #Non-zero pixel method
    nonzero = top.nonzero()
    output_image = bottom
    for i in range(len(nonzero[0])):
        output_image[nonzero[0][i]][nonzero[1][i]] = \
            top[nonzero[0][i]][nonzero[1][i]]
    return output_image

def shade_lines(image, left_line, right_line):
    """
    This function draws and shades the lane areas defined by the current_fit
    polynomial.
    """
    #Setup output image
    output_image = image.copy()
    #Define lines
    y = np.arange(image.shape[0])
    left_points = np.ndarray(shape=(0,0), dtype=int)
    right_points = np.ndarray(shape=(0,0), dtype=int)
    #Left line if polynomial defined
    if len(left_line.current_fit) == 3:
        left_fitx = lambda y: left_line.current_fit[0] * y**2 + \
                              left_line.current_fit[1] * y + \
                              left_line.current_fit[2]
        x1 = left_fitx(y)
        left_points = np.array([[[xi, yi]] for xi, yi in zip(x1, y) \
            if (0<=xi<image.shape[1] and 0<=yi<image.shape[0])]).astype(np.int32)
    #Right line if polynomial defined
    if len(right_line.current_fit) == 3:
        right_fitx = lambda y: right_line.current_fit[0] * y**2 + \
                               right_line.current_fit[1] * y + \
                               right_line.current_fit[2]
        x2 = right_fitx(y)
        right_points = np.array([[[xi, yi]] for xi, yi in zip(x2, y) \
            if (0<=xi<image.shape[1] and 0<=yi<image.shape[0])]).astype(np.int32)
    #If both lines present, shade area
    if (left_points.size != 0) & (right_points.size != 0):
        #Create polygon
        points = np.concatenate((left_points, np.flip(right_points, 0)))
        #Fill area
        shade_color = [0, 255, 0] #Green
        shade_image = np.zeros_like(image)
        cv2.fillPoly(shade_image, [points], shade_color)
        #Transform back to origianl perspective
        shade_image = cv2.warpPerspective(shade_image, \
                                          INV_MATRIX, \
                                          (image.shape[1], image.shape[0]))
        #Shade lane area
        output_image = cv2.addWeighted(image, 1.0, shade_image, 1.0, gamma=0.0)
    #Draw lines
    line_color = [255, 0, 0] #Blue
    line_image = np.zeros_like(image)
    if left_points.size != 0:
        cv2.drawContours(line_image, left_points, -1, line_color, LINE_THICKNESS)
    if right_points.size != 0:
        cv2.drawContours(line_image, right_points, -1, line_color, LINE_THICKNESS)
    #Transform back to origianl perspective
    line_image = cv2.warpPerspective(line_image, \
                                      INV_MATRIX, \
                                      (image.shape[1], image.shape[0]))
    output_image = combine_images(output_image, line_image)
    return output_image

def get_radius(line_poly, y):
    """
    This function returns the radius of a given line polynomial, independent of
    units.    
    """
    #Returns the radius created by a 2nd order polynomial
    radius = ((1 + (2 * line_poly[0] * y + line_poly[1])**2)**(3 / 2)) / \
             np.absolute(2*line_poly[0])
    return radius

def draw_status(image, left_line, right_line):
    """
    This function determines road width, vehicle offset, and road radius in real
    world units (meters) and draws the current status on the image.
    """
    #Verify both polynomials defined
    if not (len(left_line.current_fit) == 3 & \
            len(right_line.current_fit) == 3): return image
    #Define lines
    left_fitx = lambda y: left_line.current_fit[0] * y**2 + \
                          left_line.current_fit[1] * y + \
                          left_line.current_fit[2]
    right_fitx = lambda y: right_line.current_fit[0] * y**2 + \
                           right_line.current_fit[1] * y + \
                           right_line.current_fit[2]
    #Calculate road width
    road_width_pix = right_fitx(0) - left_fitx(0)
    road_width = road_width_pix * PIXEL_TO_M_X
    #Calculate offset from center
    off_center_pix = image.shape[1] / 2 - (right_fitx(0) + left_fitx(0)) / 2
    off_center = off_center_pix * PIXEL_TO_M_X
    off_center_str = "Off-center: " + "{0:.2f}".format(off_center) + " m"
    #Get line points
    y = np.arange(image.shape[0])
    leftx = left_fitx(y)
    rightx = right_fitx(y)
    #Convert to real world
    y = np.multiply(y, PIXEL_TO_M_Y, casting="unsafe")
    leftx *= PIXEL_TO_M_X
    rightx *= PIXEL_TO_M_X
    #Get real world polynomial
    left_poly = np.polyfit(y, leftx, 2)
    right_poly = np.polyfit(y, rightx, 2)
    left_raddi = get_radius(left_poly, y)
    right_raddi = get_radius(right_poly, y)
    radius = (np.average(left_raddi) + np.average(right_raddi)) / 2
    radius_str = "Radius: " + "{0:.1f}".format(radius) + " m"
    #Draw on image
    font_color = [0, 0, 255] #Red
    cv2.putText(image, \
                off_center_str, \
                (5,20), \
                cv2.FONT_HERSHEY_PLAIN, \
                1.5, \
                font_color, \
                2)
    cv2.putText(image, \
                radius_str, \
                (5,40), \
                cv2.FONT_HERSHEY_PLAIN, \
                1.5, \
                font_color, \
                2)
    return image

def process_image(image, mtx, dist, left_line, right_line):
    """
    This is the overall function that evalutes each new image to detect road
    lines and then draw the visual information on it after undistortion.
    """
    #Work with working copy
    working_image = image.copy()
    #Normalize image
    cv2.normalize(working_image, working_image, 0, 255, cv2.NORM_MINMAX)
    #Undistort image
    true_image = cv2.undistort(working_image, mtx, dist, None, mtx)
    #Extract ROI
    roi_image = extract_roi(true_image)
    #Color threshold
    color_mask = hls_threshold(roi_image)
    #Sobel threshold
    gradient_mask = gradient_threshold(roi_image)
    #Trim gradient mask to remove false edges
    gradient_mask = trim_roi(gradient_mask, 3)
    #Combine masks
    binary = cv2.bitwise_and(color_mask, gradient_mask)
    #Warp to birds-eye-view perspective
    bev_image = cv2.warpPerspective(binary, \
                                    BEV_MATRIX, \
                                    (binary.shape[1], binary.shape[0]))
    #Find lines
    visual_image = detect_lines_basic(bev_image, left_line, right_line)
    #Create output image
    output_image = shade_lines(true_image, left_line, right_line)
    output_image = draw_status(output_image, left_line, right_line)
    return output_image