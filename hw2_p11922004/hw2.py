import math

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utilities import \
                    plot_histograms,\
                    plot_histogram, \
                    plot_ghe, \
                    get_parametric_img, \
                    low_pass_filter, \
                    median_filter, \
                    get_psnr, \
                    plot_lhe

# ============================================================
# ================ Problem 1: EDGE DETECTION =================
# ============================================================

# img = cv2.imread('./hw1_sample_images/sample1.png')
# img = np.array(img)

# (a) (10 pt) Apply Sobel edge detection to sample1.png. Output the gradient image and its corresponding edge
# map as result1.png and result2.png, respectively. Please also describe how you select the threshold and how
# it affects the result.
print()

# (b) (20 pt) Perform Canny edge detection on sample1.png and output the edge map as result3.png. Please also
# describe how you select the parameters and how they affect the result.
print()

# (c) (10 pt) Use the Laplacian of Gaussian edge detection to generate the edge map of sample1.png and output it
# as result4.png. Compare result2.png result3.png and result4.png and discuss on these three results.
print()

# (d) (10 pt) Perform edge crispening on sample2.png and output the result as result5.png. What difference can
# you observe from sample2.png and result5.png? Please specify the parameters you choose and discuss how
# they affect the result.
print()

# (e) (Bonus) Perform Canny edge detection on result5.png and output the edge map as result6.png. Then apply
# the Hough transform to result6.png and output the Hough space as result7.png. What lines can you detect
# by this method?
print()

# ============================================================
# ============ Problem 2: GEOMETRICAL MODIFICATION ===========
# ============================================================

# img2 = cv2.imread('./hw1_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
# img2 = np.array(img2)

# (a) (25 pt) The Borzoi wants to help you to get the potato chips. Please design an algorithm to make sample3.png
# become sample4.png. Output the result as result8.png with the same dimension as sample3.png. Please
# describe your method and implementation details clearly. (hint: you may perform rotation, scaling, translation,
# etc.)
print()

# (b) (25 pt) I made my own Popcat picture, although there’s no cat in it. By observing the effect shown in sample6.
# png, please design an algorithm to make sample5.png look like it as much as possible and save the output
# as result9.png. Please describe the details of your method and also provide some discussions on the designed
# method, the result, and the difference between result9.png and sample6.png, etc.
print()