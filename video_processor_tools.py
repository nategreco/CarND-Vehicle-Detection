################################################################################
#   Date:    08.24.2017
#   Author:  Nathan Greco (Nathan.Greco@gmail.com)
#
#   Project:
#       CarND-Vehicle-Detection: Final project in Term 1 of Self-Driving
#           Car Nanodegree curriculum.
#
#   Module:
#       video_processor_tools.py: Contains tools for reading, writing, and
#           editing of video files.
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

#3rd party imports
import cv2

#Local project imports

#Functions
def rename_output_file(path):
    return path[path.rfind('\\')+1:path.rfind('.')] + \
        "_edit" + \
        path[path.rfind('.'):]