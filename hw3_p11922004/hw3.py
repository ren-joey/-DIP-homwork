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
from hw3_utilities import hole_filling
from plotter import img_plotter

# ============================================================
# =========== Problem 1: MORPHOLOGICAL PROCESSING ============
# ============================================================


# (a) (10 pt) Design a morphological processing to extract the objects’ boundaries in sample1.png and
# output the result as result1.png.
sample1 = cv2.imread('./hw3_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
# img1 = np.array(sample1)
# mask = np.array([
#     [1, 1, 1],
#     [1, 1, 1],
#     [1, 1, 1]
# ])
# threshold = np.sum(mask) * 255
# extend = mask.shape[0] // 2
# G, w, h = black_img_copier(img1)
# for y in range(extend, h-extend):
#     for x in range(extend, w-extend):
#         candidate = img1[y-extend:y+extend+1, x-extend:x+extend+1]
#         sum = np.sum(candidate * mask)
#         if (sum >= threshold):
#             G[y][x] = 255
# img1 -= G
# img_plotter(
#     [sample1, G, img1],
#     ['sample1', 'sample1 content', 'result1']
# )
# cv2.imwrite('./result1.png', img1)

# (b) (10 pt) Perform hole filling on sample1.png and output the result as result2.png.

img1 = np.array(sample1)

filled_img = cv2.imread('./filled_img.png', cv2.IMREAD_GRAYSCALE)
filled_map = cv2.imread('./filled_map.png', cv2.IMREAD_GRAYSCALE)
filled_inverted_map = cv2.imread('./filled_inverted_map.png', cv2.IMREAD_GRAYSCALE)
if filled_img is None \
 or filled_map is None \
 or filled_inverted_map is None:
    filled_img, filled_map, filled_inverted_map = hole_filling(img1, 0, 0)
    cv2.imwrite('./filled_bg_sample1.png', filled_img)
    cv2.imwrite('./filled_map.png', filled_map)
    cv2.imwrite('./filled_inverted_map.png', filled_inverted_map)

img_plotter(
    [filled_img, filled_map, filled_inverted_map],
    ['filled_img', 'filled_map', 'inverted_filled_map']
)

# coord = [-1, -1]
# for y in range(content_point_finding_map.shape[0]):
#     for x in range(content_point_finding_map.shape[1]):
#         p = content_point_finding_map[y][x]
#         if p == 0:
#             coord = [y, x]
#             break
#
# res, inverted_res = hole_filling(img1, coord[1], coord[0])

# img_plotter(
#     [res, inverted_res],
#     ['res', 'inverted_res']
# )

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
