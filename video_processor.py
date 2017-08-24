################################################################################
#   Date:    08.24.2017
#   Author:  Nathan Greco (Nathan.Greco@gmail.com)
#
#   Project:
#       CarND-Vehicle-Detection: Final project in Term 1 of Self-Driving
#           Car Nanodegree curriculum.
#
#   Module:
#       video_processor.py: Main module which handles video arguments and
#           applies processing pipeline.
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
import sys
import os

#3rd party imports
import cv2

#Local project imports
import video_processor_tools as vidtools
import lane_detect_processor as lanetools
import vehicle_detect_processor as vehtools

#Constants
CAL_PATH = '.\\camera_cal'
CAL_PTS_X = 9
CAL_PTS_Y = 6
LINES_TO_AVERAGE = 5

#Code
#Check for arguments
if len(sys.argv) < 2:
    input("No arguments passed, press ENTER to exit...")
    quit()

#Perform camera calibration
image_files = [f for f in os.listdir(CAL_PATH) \
    if os.path.isfile(os.path.join(CAL_PATH, f))]
images = []
for file in image_files:
    image = cv2.imread(CAL_PATH + '\\' + file)
    images.append(image)
cal_mtx, cal_dist = lanetools.calibrate_camera(images, CAL_PTS_X, CAL_PTS_Y)

#Perform model training
#TODO

#Iterate through video files
for i in range(1, len(sys.argv)):
    video_in = cv2.VideoCapture(sys.argv[i])
    video_out = cv2.VideoWriter(vidtools.rename_output_file(sys.argv[i]), \
        int(video_in.get(cv2.CAP_PROP_FOURCC)), \
        video_in.get(cv2.CAP_PROP_FPS), \
        (int(video_in.get(cv2.CAP_PROP_FRAME_WIDTH)), \
         int(video_in.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    #Create empty line classes
    left_line = lanetools.Line(10)
    right_line = lanetools.Line(10)
    while(video_in.isOpened()):
        result, frame = video_in.read()
        if result==False:
            print("File completed: " + sys.argv[i])
            break

        #Process frame to find and draw lines
        #output_image = lanetools.process_image(frame, \
        #                                       cal_mtx, \
        #                                       cal_dist, \
        #                                       left_line, \
        #                                       right_line)
        
        #Process frame to find and track vehicles
        #TODO
        output_image = frame
        
        #Write new frame
        video_out.write(output_image)
        #Display new frame
        cv2.imshow('Results', output_image)        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_in.release()
    video_out.release()

cv2.destroyAllWindows()