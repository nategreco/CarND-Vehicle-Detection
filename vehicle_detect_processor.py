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

#Local project imports

#Constants
##FEATURES
SPATIAL_FEAT = False
HIST_FEAT = True
HOG_FEAT = True
###SPATIAL
###HIST
###HOG
HOG_COLOR = 'YUV' #Can be BGR, HSV, LUV, HLS, YUV, YCrCb
HOG_ORIENT = 11
HOG_CELL_PIX = 16
HOG_BLOCK_PIX = 2
HOG_CHANNEL = "ALL" #Can be 0, 1, 2, or "ALL"

#Classes
class Vehicles():
    """
    Tracks the vehicles over a defined number of frames and does appropriate
    filtering to remove false positives
    """
    def __init__(self, num_to_keep):
        #Number of frames to keep
        self.num_to_keep = num_to_keep
    def update(self, boxes):
        #TODO

#Functions
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
    #Apply color conversion if other than 'BGR'
    if color_space != 'BGR':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    else: feature_image = np.copy(image)         

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
                                   color_space=HOG_COLOR,
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
                                   color_space=HOG_COLOR,
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
    return svc


def process_image(image, model, vehicles):
    """
    Scans image with sliding window technique to detect possible vehicles
    then updates the class
    """
    #Work with working copy
    working_image = image.copy()
    #TODO - sliding windows
    #TODO - get features for each windows
    #TODO - predict each window with model
    #TODO - construct heat map
    #TODO - evaluate heat map
    #TODO - update class

def draw_vehicles(image, vehicles):
    """
    Draws filtered vehicles from a given class on a given image
    """
    #TODO
    output_image = image.copy()
    return output_image