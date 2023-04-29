import math
import random

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
    line_detection_non_vectorized, \
    filter_processor
# from hw4_utilities import
from plotter import img_plotter

sample1 = cv2.imread('./hw3_sample_images/sample1.png')
sample2 = cv2.imread('./hw3_sample_images/sample2.png')
sample3 = cv2.imread('./hw3_sample_images/sample3.png')

# ============================================================
# ============== Problem 1: DIGITAL HALFTONING ===============
# ============================================================

# (a) (10 pt) According to the dither matrix I2,
# please perform dithering to obtain a binary image result1.png.

# cv2.imwrite('./result1.png', )


# (b) (15 pt) Expand the dither matrix I2 to I256 (256 × 256) and use it to perform dithering.
# Output the result as result2.png.
# Compare result1.png and result2.png along with some discussions.

# cv2.imwrite('./result2.png', )


# (c) (25 pt) Perform error diffusion with Floyd-Steinberg and Jarvis’ patterns on sample1.png.
# Output the results as result3.png and result4.png, respectively.
# You may also try more patterns and show the results in your report.
# Discuss these patterns based on the results.
# You can find some masks here (from lecture slide 06. p21)

# cv2.imwrite('./result3.png', )
# cv2.imwrite('./result4.png', )

# ============================================================
# ================ Problem 2: FREQUENCY DOMAIN ===============
# ============================================================

# (a) (10 pt) By analyzing sample2.png,
# please explain how to perform image sampling on it to avoid aliasing.
# Please also perform 'inappropriate' image sampling which results in aliasing in the sampled image.
# Output the result as result5.png, specify the sampling rate you choose and discuss how it affects the resultant image.

# cv2.imwrite('./result5.png', )

# (b) (20 pt) Please perform the Gaussian high-pass filter in the frequency domain on sample2.png
# and transform the result back to the pixel domain by inverse Fourier transform. Save the resultant
# image as result6.png.

# cv2.imwrite('./result6.png', )

# (c) (20 pt) Try to remove the undesired pattern on sample3.png
# with Fourier transform and output it as result7.png.
# Please also describe how you accomplished the task.

# cv2.imwrite('./result7.png', )
