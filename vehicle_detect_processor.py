################################################################################
#   Date:    08.24.2017
#   Author:  Nathan Greco (Nathan.Greco@gmail.com)
#
#   Project:
#       CarND-Vehicle-Detection: Final project in Term 1 of Self-Driving
#           Car Nanodegree curriculum.
#
#   Module:
#       vehicle_detect_processor.py: Module which applies vehicle detection
#           and tracking pipelines.
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
import time
import glob

#3rd party imports
import matplotlib.image as mpimg
import numpy as np
import cv2
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from scipy.ndimage.measurements import label

#Local project imports

#Constants
##FEATURES
SPATIAL_FEAT = False
HIST_FEAT = False
HOG_FEAT = True
###SPATIAL
###HIST
###HOG
COLOR_SPACE = 'YUV' #Can be BGR, HSV, LUV, HLS, YUV, YCrCb
SPATIAL_SIZE = (32, 32)
HIST_BINS = 32
HOG_ORIENT = 11
HOG_CELL_PIX = 16
HOG_BLOCK_PIX = 2
HOG_CHANNEL = "ALL" #Can be 0, 1, 2, or "ALL"
##Windows
START_POS = 400
STOP_POS = 656
SCALE = 1.5
##Filtering
THRESHOLD = 1
PREV_THRESHOLD = 3
##Drawing
WIN_BUFF = 20
BOX_COLOR = (0, 255, 0)

#Classes
class Vehicles():
    """
    Tracks the vehicles over a defined number of frames and does appropriate
    filtering to remove false positives
    """
    def __init__(self, num_to_keep, height, width):
        #Number of frames to keep
        self.num_to_keep = num_to_keep
        #Save image size
        self.shape = (height, width)
        #Initialize boxes to draw
        self.good_boxes = []
        #Initialize list of previous boxes
        self.prev_boxes = []
    def update(self, labels):
        #Update list of previous boxes
        self.prev_boxes.append(labels)
        if len(self.prev_boxes) > self.num_to_keep:
            del self.prev_boxes[0]
        #Create heat map from windows
        heat_image = np.zeros(self.shape, dtype=np.uint8)
        for set in self.prev_boxes:
            heat_image = add_heat_labels(heat_image, set)
        #Filter heatmap
        heat_image = apply_threshold(heat_image, PREV_THRESHOLD)
        #Update good boxes
        self.good_boxes = label(heat_image)

#Functions
def convert_color(image, color='BGR'):
    if color != 'BGR':
        if color == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color == 'LUV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif color == 'HLS':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif color == 'YUV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color == 'YCrCb':
            return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return image.copy()     
        
def bin_spatial(img, size=(32, 32)):
    """
    Compute binned color features
    """
    #Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()    
    #Return the feature vector
    return features

def color_hist(img, nbins=32, bins_range=(0, 256)):
    """
    Computes color historgram features
    """
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def get_hog_features(img,
                     orient,
                     pix_per_cell,
                     cell_per_block,
                     vis=False,
                     feature_vec=True):
    """
    Returns HOG features
    """
    #Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient,    
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block),
                                  block_norm='L1',
                                  transform_sqrt=True,    
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    #Otherwise call with one output
    else:         
        features = hog(img, orientations=orient,    
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block),
                       block_norm='L1',
                       transform_sqrt=True,    
                       visualise=vis, feature_vector=feature_vec)
        return features

def extract_features(image,
                     color_space='BGR',
                     spatial_size=(32, 32),
                     hist_bins=32,
                     orient=9,    
                     pix_per_cell=8,
                     cell_per_block=2,
                     hog_channel=0,
                     spatial_feat=True,
                     hist_feat=True,
                     hog_feat=True):
    """
    Extracts features from image
    """
    image_features = []
    #Apply color conversion
    feature_image = convert_color(image, color_space)      

    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        image_features.append(spatial_features)
    if hist_feat == True:
        #Apply color_hist()
        hist_features = color_hist(feature_image, nbins=hist_bins)
        image_features.append(hist_features)
    if hog_feat == True:
    #Call get_hog_features() with vis=False, feature_vec=True
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel],    
                                    orient, pix_per_cell, cell_per_block,    
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)           
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient,    
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #Append the new feature vector to the features list
        image_features.append(hog_features)
    #Return list of feature vectors
    return np.concatenate(image_features)

def train_model(path):
    """
    Trains a linear SVM model
    """
    #Load training data
    print('Loading training data')
    t=time.time()
    car_files = glob.glob(path + '\\' + 'vehicles' + '\\**\\' + '*.png', recursive=True)
    notcar_files = glob.glob(path + '\\' + 'non-vehicles' + '\\**\\' + '*.png', recursive=True)
    #TESTING - Reduce sample size
    #car_files = car_files[0:500]
    #notcar_files = notcar_files[0:500]
    cars = []
    for file in car_files:
        image = cv2.imread(file)
        cars.append(image)
    notcars = []
    for file in notcar_files:
        image = cv2.imread(file)
        notcars.append(image)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to load training files...')
    #Extract features
    print('Extracting features')
    t=time.time()
    car_features = []
    for image in cars:
        feature = extract_features(image,
                                   color_space=COLOR_SPACE,
                                   spatial_size=SPATIAL_SIZE,
                                   hist_bins=HIST_BINS,
                                   orient=HOG_ORIENT,    
                                   pix_per_cell=HOG_CELL_PIX,
                                   cell_per_block=HOG_BLOCK_PIX,
                                   hog_channel=HOG_CHANNEL,
                                   spatial_feat=SPATIAL_FEAT,
                                   hist_feat=HIST_FEAT,
                                   hog_feat=HOG_FEAT)
        car_features.append(feature)
    notcar_features = []
    for image in notcars:
        feature = extract_features(image,
                                   color_space=COLOR_SPACE,
                                   spatial_size=SPATIAL_SIZE,
                                   hist_bins=HIST_BINS,
                                   orient=HOG_ORIENT,    
                                   pix_per_cell=HOG_CELL_PIX,
                                   cell_per_block=HOG_BLOCK_PIX,
                                   hog_channel=HOG_CHANNEL,
                                   spatial_feat=SPATIAL_FEAT,
                                   hist_feat=HIST_FEAT,
                                   hog_feat=HOG_FEAT)
        notcar_features.append(feature)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to extract HOG features...')
    #Prepare training data
    print('Preparing training data')
    t=time.time()
    #Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                           
    #Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    #Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    #Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    #Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    print('Using:',HOG_ORIENT,'orientations',HOG_CELL_PIX,
        'pixels per cell and', HOG_BLOCK_PIX,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to prepare training data...')
    #Train
    print('Training Linear SVC')
    t=time.time()
    #Use a linear SVC    
    svc = LinearSVC()
    #Check the training time for the SVC
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    #Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    #Return model
    model = {'svc':svc, 'X_scaler':X_scaler, 'color_space':COLOR_SPACE, 
             'spatial_size':SPATIAL_SIZE, 'hist_bins':HIST_BINS, 'orient':HOG_ORIENT,
             'pix_per_cell':HOG_CELL_PIX, 'cell_per_block':HOG_BLOCK_PIX,
             'hog_channel':HOG_CHANNEL, 'spatial_feat':SPATIAL_FEAT,
             'hist_feat':HIST_FEAT, 'hog_feat':HOG_FEAT}
    return model

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    """
    Draws bounding boxes
    """
    #Make a copy of the image
    imcopy = np.copy(img)
    #Iterate through the bounding boxes
    for bbox in bboxes:
        #TEMP - Different colors help sliding windows method
        color = (np.random.randint(0, 255),
                 np.random.randint(0, 255),
                 np.random.randint(0, 255))
        #Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    #Return the image copy with boxes drawn
    return imcopy

def find_cars(img,
              ystart,
              ystop,
              scale,
              svc,
              X_scaler,
              color_space,
              spatial_size,
              hist_bins,
              orient,
              pix_per_cell,
              cell_per_block,
              hog_channel,
              spatial_feat,
              hist_feat,
              hog_feat):
    """
    Find cars
    """
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step
    
    #Compute individual channel HOG features for the entire image
    if (hog_feat):
        if hog_channel == 'ALL':
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, \
                feature_vec=False)
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, \
                feature_vec=False)
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, \
                feature_vec=False)
        elif hog_channel == 0:
            hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, \
                feature_vec=False)
        elif hog_channel == 1:
            hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, \
                feature_vec=False)
        elif hog_channel == 2:
            hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, \
                feature_vec=False)

    boxes = []
    for xb in range(nxsteps):
        for yb in range(nysteps):
            #Get window
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell
            
            #Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
            
            #Get features -- > This way ensures we use the same feature extraction
            #methods as the training, however, doesn't take advantage of a one time
            #HOG method
            """
            feature = extract_features(subimg,
                                       color_space=color_space,
                                       spatial_size=spatial_size,
                                       hist_bins=hist_bins,
                                       orient=orient,    
                                       pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel,
                                       spatial_feat=spatial_feat,
                                       hist_feat=hist_feat,
                                       hog_feat=hog_feat)
            test_features = X_scaler.transform(np.hstack(feature).reshape(1, -1))
            """

            #Get color features
            if (spatial_feat):
                spatial_features = bin_spatial(subimg, size=spatial_size)
            if (hist_feat):
                hist_features = color_hist(subimg, nbins=hist_bins)
            
            #Extract HOG for this patch
            if (hog_feat):
                if hog_channel == 'ALL':
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                elif hog_channel == 0:
                    hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()  
                    hog_features = np.hstack(hog_feat1)
                elif hog_channel == 1:
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()  
                    hog_features = np.hstack(hog_feat2)
                elif hog_channel == 2:
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()  
                    hog_features = np.hstack(hog_feat3)

            #Get features
            if (spatial_feat & hist_feat & hog_feat):
                test_features = X_scaler.transform(np.hstack((spatial_features, \
                    hist_features, hog_features)).reshape(1, -1))
            elif (spatial_feat & hist_feat):
                test_features = X_scaler.transform(np.hstack((spatial_features, \
                    hist_features)).reshape(1, -1))
            elif (spatial_feat & hog_feat):
                test_features = X_scaler.transform(np.hstack((spatial_features, \
                    hog_features)).reshape(1, -1))
            elif (hist_feat & hog_feat):
                test_features = X_scaler.transform(np.hstack((hist_features, \
                    hog_features)).reshape(1, -1))
            elif (spatial_feat):
                test_features = X_scaler.transform(np.hstack(spatial_features).reshape(1, -1))
            elif (hist_feat):
                test_features = X_scaler.transform(np.hstack(hist_features).reshape(1, -1))
            elif (hog_feat):
                test_features = X_scaler.transform(np.hstack(hog_features).reshape(1, -1))

            #Make prediction
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                boxes.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
    return boxes

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap

def add_heat_labels(heatmap, labels):
    #Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        #Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        #Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        #Define a bounding box based on min/max x and y
        box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        #Draw the box on the image
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    #Return the image
    return heatmap

def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels, color=(0,255,0)):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox) - WIN_BUFF, np.min(nonzeroy) - WIN_BUFF), \
            (np.max(nonzerox) + WIN_BUFF, np.max(nonzeroy) + WIN_BUFF))
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], color, 4)
    # Return the image
    return img

def process_image(image, model, vehicles):
    """
    Scans image with sliding window technique to detect possible vehicles
    then updates the class
    """
    #Get parameters from model
    svc = model["svc"]
    X_scaler = model["X_scaler"]
    color_space = model["color_space"]
    spatial_size = model["spatial_size"]
    hist_bins = model["hist_bins"]
    orient = model["orient"]
    pix_per_cell = model["pix_per_cell"]
    cell_per_block = model["cell_per_block"]
    hog_channel = model["hog_channel"]
    spatial_feat = model["spatial_feat"]
    hist_feat = model["hist_feat"]
    hog_feat = model["hog_feat"]
    #Work with working copy
    working_image = image.copy()
    #Find cars
    boxes = find_cars(working_image,
                      START_POS,
                      STOP_POS,
                      SCALE,
                      svc,
                      X_scaler,
                      color_space,
                      spatial_size,
                      hist_bins,
                      orient,
                      pix_per_cell,
                      cell_per_block,
                      hog_channel,
                      spatial_feat,
                      hist_feat,
                      hog_feat)
    output_image = draw_boxes(working_image, boxes)
    #Create heat map from windows
    heat_image = np.zeros_like(image[:,:,0])
    heat_image = add_heat(heat_image, boxes)
    #Filter heatmap
    heat_image = apply_threshold(heat_image, THRESHOLD)
    labels = label(heat_image)
    #Update class
    vehicles.update(labels)
    #Return working image for development
    return output_image

def draw_vehicles(image, vehicles):
    """
    Draws filtered vehicles from a given class on a given image
    """
    #Draw filtered boxes
    output_image = draw_labeled_bboxes(np.copy(image), vehicles.good_boxes, BOX_COLOR)
    return output_image