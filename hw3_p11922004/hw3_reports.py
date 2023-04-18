import math
import cv2

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utilities import \
    unsharp_masking, \
    img_expend, \
    sobel_edge_detection, \
    gaussian_filter, \
    non_maximum_suppression, \
    hysteretic_thresholding, \
    connected_component_labeling, \
    canny_edge_detection, \
    log_filtering, \
    image_coordinates_transformation, \
    black_img_copier, \
    mod_img_by_fun, \
    line_detection_non_vectorized
from plotter import img_plotter

# ============================================================
# =========== Problem 1: MORPHOLOGICAL PROCESSING ============
# ============================================================


# (a) (10 pt) Design a morphological processing to extract the objects’ boundaries in sample1.png and
# output the result as result1.png.


# (b) (10 pt) Perform hole filling on sample1.png and output the result as result2.png.


# (c) (10 pt) Design an algorithm to count the number of objects in sample1.png. Describe the steps
# in detail and specify the corresponding parameters.


# (d) (20 pt) Apply open operator and close operator to sample1.png and output the results as result3.
# png and result4.png, respectively. How will it affect the result images if you change the
# shape of the structuring element?


# (e) (Bonus) Perform Canny edge detection on result5.png and output the edge map as result6.png. Then apply
# the Hough transform to result6.png and output the Hough space as result7.png. What lines can you detect
# by this method?


# ============================================================
# ================ Problem 2: TEXTURE ANALYSIS ===============
# ============================================================


# (a) (10 pt) Perform Law’s method on sample2.png to obtain the feature vectors. Please describe
# how you obtain the feature vectors and provide the reason why you choose it in this way.


# (b) (20 pt) Use k-means algorithm to classify each pixel with the feature vectors you obtained from
# (a). Label the pixels of the same texture with the same color and output it as result5.png. Please
# describe how you use the features in k-means algorithm and all the chosen parameters in detail.


# (c) (20 pt) Based on result5.png, design a method to improve the classification result and output the
# updated result as result6.png. Describe the modifications in detail and explain the reason why.


# (d) (Bonus) TA can’t swim. Try to perform image quilting, replacing the sea in sample2.png with
# sample3.png or other texture you prefer by using the result from (c), and output it as result7.png.
# It’s allowed to utilize external libraries to help you accomplish it, but you should specify the implementation
# detail and functions you used in the report.
