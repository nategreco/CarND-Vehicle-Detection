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
[image1]: ./cars.JPG "Cars"
[image2]: ./notcars.JPG "Not-Cars"
[image3]: ./windows_car.JPG "Window Positions"
[image4]: ./detected.JPG "Detected"
[image5]: ./heatmap.JPG "Heatmap"
[image6]: ./filtered.JPG "Filtered"
[image7]: ./result.JPG "Result"
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

The training data consisted of 17,767 64x64 .png images, classified into two categories, 'vehicle' and 'non-vehicle'.  Examples of from the training set are shown below.

Vehicles:

![Vehicles][image1]


Non-vehicles:

![Non-vehicles][image2]


#### 2. Explain how you settled on your final choice of HOG parameters.

Many parameters were manipulated and experimation was done with different featuers.  The code was written so that spatial, histogram, and HOG features could be selected independent of each other.  I found after some experimentation the results were best with HOG features only.  Including spatial and histogram features often created false positives, whereas HOG seemed to be the most consistent and also performance was better.

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

Ultimately I settled on using HOG only features, even though the code was modified to support any combintation of HOG, spatial, and histogram features.  The [process_image()](../vehicle_detect_processor.py#L710) function performed the following steps to evaluate each frame:

 - First called [find_cars()](../vehicle_detect_processor.py#L525) which used a sliding window method and returned a list of boxes which were preditcted to be cars.

![Detected][image4]

- Then called [add_heat()](../vehicle_detect_processor.py#L652) to create a heatmap from the boxes.  A value to pad the windows was also added to promote growth of the windows so adjacent boxes would merge, thus avoiding duplicates.

![Heatmap][image5]

- Afterwards labels were obtained from the heatmap by applying a threshold with the [apply_threshold()](../vehicle_detect_processor.py#L688) function.

![Filtered][image6]

- Finally an additional heatmap was applied over the previous 10 recorded frames with padding to promote merging of adjacent windows.  This helped merge the tendency to segment the vehicles by windows. The final labels were then padded additionally to fully encompass the vehicle and the result was drawn with the [draw_vehicles()](../vehicle_detect_processor.py#L658) function.

![Result][image7]


---

### Video Implementation

#### 1. Provide a link to your final video output.

Here's my results:

[Processed Video](../project_video_edit.mp4) - Minimal false positives and duplicates due to a heatmap filter applied to a 10 frame memory.


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Three methods were implemented to prevent false positives and duplicates.

### Heatmap filter method applied to original frame evaluation:

This was implemented by the [apply_threshold()](../vehicle_detect_processor.py#L688) function and called in [process_image()](../vehicle_detect_processor.py#L751).  A relatively low threshold value was used and essentially at least two windows were required to overlap in order for the function to be labeled as a vehicle.

### A class which retains the last 10 frames and applies a heatmap filter again:

The [Vehicles](../vehicle_detect_processor.py#L74) class was created to maintain the history of previous 10 frames.  The [update()](../vehicle_detect_processor.py#L88) maintained a LIFO list of the previous frames and applied an additional heatmap filter for all of the frames.  This placed a much greater weight over vehicles that were detected over multipe frames in the same location, making it much easier to filter out false positives.

### Padding to the initial and historic windows to promote merging of adjacent windows:
Additionally, the [add_heat_labels()](../vehicle_detect_processor.py#L668) function was created to handle the historic labels and apply the additional historic heatmap as described above, but also added a padding to encourage the merger of adjacent windows.  This prevented segmentation of the vehicles and instead returned a larger box that enveloped the whole vehicle as opposeod to two side by side boxes.

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The most difficult aspect of this project were the inter-dependencies of the parameters.  It was quickly determined that the HOG features were most effective in detecting vehicles.  It was also readily-obvious that the heat map could remove duplicates and false positives, and a time-weighted or historic filter of some type would also be effective, however the challenge was that manipulation of one parameter directly impacted another.  For example, changing the overlap of the sliding windows had a direct impact on the heatmap, so more overlap required a higher threshold due to more overlapping windows.  The best approach I found was to first do some experimentation to understand the behaviors of the filters, then tune each one by one in the sequence of the pipeline.
