# **Vehicle Detection**

### Vehicle tracking using a Support Vector Machine

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./car.JPG "Car"
[image2]: ./not_car.JPG "Not-Car"
[image3]: ./windows_car.JPG "Window Positions"
[image3]: ./detected.JPG "Detected"
[image3]: ./heatmap.JPG "Heatmap"
[image3]: ./filtered.JPG "Filtered"
[video1]: ../project_video_edit.mp4 "Video"

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

My project includes the following files:
* [video_processor.py](../video_processor.py) containing main function which handles video file paths as arguements to process
* [video_processor_tools.py](../video_processor_tools.py) containing various functions for the editing of video files
* [vehicle_detect_processor.py](../vehicle_detect_processor.py) containing all classes and functions for vehicle detection and image manipulation
* [lane_detect_processor.py](../lane_detect_processor.py) containing all classes and functions for lane detection and image manipulation
* [project_video_edit.mp4](../project_video_edit.mp4) original video after processing
* [README.md](../writeup/README.md) the report you are reading now


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The function [get_hog_features()](../vehicle_detect_processor.py#L139) was called by both [extract_features()](../vehicle_detect_processor.py#L167) during training and [find_cars()](../vehicle_detect_processor.py#L525) during prediction.  The function was used un-modified from the course lessons and utilized the HOG module in skimage.feature.


*********************************************
![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]
*********************************************

#### 2. Explain how you settled on your final choice of HOG parameters.

Many parameters were manipulated and experimation was doen with different featuers.  The code was written so that spatial, histogram, and HOG features could be selected independent of each other.  I found after some experimentation the results were best with HOG features only.  Including spatial and histogram features often created false positives, whereas HOG seemed to be the most consistent and also performance was better.

Selecting the correct HOG parameters was done by experimentation to provide the highest test set accuracy.  The final values provided a test set result of 98.45% accuracy.  Below are the final settings:

| Parameter            | Value         | 
|:--------------------:|:-------------:| 
| Color Space          | YUV           | 
| HOG Orient           | 11            |
| HOG Pixels per cell  | 16            | 
| HOG Pixels per block | 2             |
| HOG Channels         | ALL           |


#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

Training was initiated by a conditional statement at the beginning of [video_processor.py](../video_processor.py#L62) that if a saved pickle file with the model was not present, it would call the [train_model()](../vehicle_detect_processor.py#L209).  In train_model() all the images in the training file path would be loaded, features would be extracted, and then training would occur.  Once trained the model was returned and saved as a pickle file.  Additionally, the trained SVC (support vector classifcation) was saved inside a dictionary with all the training parameters, that way when the pickle file was loaded all the correct parameters for feature extraction were used.


### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

A sliding window search was implemented in [find_cars()](../vehicle_detect_processor.py#L525).  The function is similar to what was provided in the course lessons, however it was modified to handle other features such as spatial and histogram.

Deciding the scales and overlap was balanced with the heat map filtering that was implemented.  Reason being was more overlap and more windows created a higher intensity in the heatmap.  Ideally, the heatmap threshold was lowered and then the overlap was adjusted until the cars were detect in every single frame.  Then at that point the heat map threshold was increased to remove the false positives.

Here's are the window positions used:

![Window Positions][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

