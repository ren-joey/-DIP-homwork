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
# ================ Problem 1: EDGE DETECTION =================
# ============================================================

# (a) (10 pt) Apply Sobel edge detection to sample1.png. Output the gradient image and its corresponding edge
# map as result1.png and result2.png, respectively. Please also describe how you select the threshold and how
# it affects the result.
sample1 = cv2.imread('./hw2_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
img1 = np.array(sample1)
img1_origin_processing, _ = sobel_edge_detection(img1)
img1_with_unsharp_masking = unsharp_masking(img1)
img1_with_unsharp_masking, _ = sobel_edge_detection(img1_with_unsharp_masking)
cv2.imwrite('./result1.png', img1_origin_processing)

img_plotter(
    [img1_origin_processing, img1_with_unsharp_masking],
    ['img1_origin_processing', 'img1_with_unsharp_masking']
)

sample2 = cv2.imread('./hw2_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
img2 = np.array(sample2)
img2_origin_processing, _ = sobel_edge_detection(img2)
img2 = unsharp_masking(img2)
img2_with_unsharp_masking, _ = sobel_edge_detection(img2)
cv2.imwrite('./result2.png', img2_origin_processing)

img_plotter(
    [img2_origin_processing, img2_with_unsharp_masking],
    ['img2_origin_processing', 'img2_with_unsharp_masking']
)

# (b) (20 pt) Perform Canny edge detection on sample1.png and output the edge map as result3.png. Please also
# describe how you select the parameters and how they affect the result.
sample6 = cv2.imread('./hw2_sample_images/sample6.png', cv2.IMREAD_GRAYSCALE)
img3 = np.array(sample1)
img3 = canny_edge_detection(img3, plot=True)
cv2.imwrite('./result3.png', img3)

# (c) (10 pt) Use the Laplacian of Gaussian edge detection to generate the edge map of sample1.png and output it
# as result4.png. Compare result2.png result3.png and result4.png and discuss on these three results.
img4 = np.array(sample1)
img4 = log_filtering(img4, size=9)
cv2.imwrite('./result4.png', img4)

sample8 = cv2.imread('./hw2_sample_images/sample8.png', cv2.IMREAD_GRAYSCALE)
img_b_4 = np.array(sample8)
img_b_4_size3 = log_filtering(img_b_4)
img_plotter(
    [img_b_4_size3],
    ['log_filtering size3']
)

# (d) (10 pt) Perform edge crispening on sample2.png and output the result as result5.png. What difference can
# you observe from sample2.png and result5.png? Please specify the parameters you choose and discuss how
# they affect the result.
img5 = np.array(sample2)
img5 = unsharp_masking(img5)
cv2.imwrite('./result5.png', img5)

# (e) (Bonus) Perform Canny edge detection on result5.png and output the edge map as result6.png. Then apply
# the Hough transform to result6.png and output the Hough space as result7.png. What lines can you detect
# by this method?
img6 = np.array(img5)
img6 = canny_edge_detection(img6, plot=True)
line_detection_non_vectorized(img6, num_rhos=50, num_thetas=50, t_count=80)
cv2.imwrite('./result6.png', img6)

# ============================================================
# ============ Problem 2: GEOMETRICAL MODIFICATION ===========
# ============================================================

# img2 = cv2.imread('./hw1_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
# img2 = np.array(img2)

# (a) (25 pt) The Borzoi wants to help you to get the potato chips. Please design an algorithm to make sample3.png
# become sample4.png. Output the result as result8.png with the same dimension as sample3.png. Please
# describe your method and implementation details clearly. (hint: you may perform rotation, scaling, translation,
# etc.)

# https://eeweb.engineering.nyu.edu/~yao/EL5123/lecture12_ImageWarping.pdf
sample3 = cv2.imread('./hw2_sample_images/sample3.png', cv2.IMREAD_GRAYSCALE)
img8 = np.array(sample3)
h, w = img8.shape[0], img8.shape[1]

img8 = image_coordinates_transformation(img8)
img8_log = log_filtering(img8)
img_plotter(
    [img8_log],
    ['img8_log']
)
minX, minY, maxX, maxY = math.inf, math.inf, 0, 0
for x in range(img8_log.shape[1] - 1, -1, -1):
    col = img8_log[:, x].flatten()
    if np.sum(col) == 0 and (
        minX < img8_log.shape[1]
        and minY < img8_log.shape[0]
        and maxX != 0
        and maxY != 0
    ):
        break

    for (y, pixel) in enumerate(col):
        if pixel != 0:
            if x < minX:
                minX = x
            if x > maxX:
                maxX = x
            if y < minY:
                minY = y
            if y > maxY:
                maxY = y

clip = img8[minY:maxY, minX:maxX]
img_plotter(
    [clip],
    ['clip']
)

#   x    y        u    v
# (0, 0)     -> (0, 0)
# (600, 0)   -> (600, 0)
# (600, 600) -> (600, 600)
# (550, 280) -> (550, 390)
# (0, 600)   -> (0, 600)
# u = (a0) + (a1)x + (a2)y + (a3)(x^2) + (a4)(xy) + (a5)(y^2)
# 0 = a0
# 600 = (a0) + (a1)600 + (a3)360000
# 600 = (a0) + (a1)600 + (a2)600 + (a3)360000 + (a4)360000 + (a5)360000
# 550 = (a0) + (a1)550 + (a2)280 + (a3)302500 + (a4)154000 + (a5)78400
# v = (b0) + (b1)x + (b2)y + (b3)(x^2) + (a4)(xy) + (a5)(y^2)
# 0 = (b0)
# 600 = (b0) + (b1)600 + (b2)600 + (b3)360000 + (b4)360000 + (b5)360000
# 390 = (b0) + (b1)550 + (b2)280 + (b3)302500 + (b4)154000 + (b5)78400
# 600 = (b0) + (b2)600 + (b5)360000
# a5 = -2.05089
# a4 = 0.0414463
# a3 = -0.00050421
# a2 = 1.81733e-06
# a1 = -0.00000126811
# a0 = 0.999991
#
# b5 = 4.26323e-14
# b4 = 1.04422e-08
# b3 = -5.09808e-08
# b2 = 0.000179
# b1 = 1.07056
# b0 = -471.02

def scaler(x, y):
    if x > minX:
        return img8[y][x]
    # elif 550 < y < 555 and 280 < x < 285:
    #     return 0
    # elif 550 < y < 555 and 390 < x < 395:
    #     return 0
    else:
        try:
            return img8[round(y / 1.5 + (h * 0.15))][x]
        except:
            return 255

img8_y_scaled = mod_img_by_fun(img8, scaler)

def geo_mod(x, y):
    if x > minX:
        return img8_y_scaled[y][x]
    else:
        # # u = (a0) + (a1)x + (a2)y + (a3)(x^2) + (a4)(xy) + (a5)(y^2)
        # # v = (b0) + (b1)x + (b2)y + (b3)(x^2) + (b4)(xy) + (b5)(y^2)
        # u = round(1 - 0.0000012 * x + 1.81733e-06 * y - 0.00050421 * x ** 2 + 0.0414463 * x * y - 2.05089 * y ** 2)
        # v = round(1 + 1.07056 * x + 0.000179 * y - 5.09808e-08 * x ** 2 + 1.04422e-08 * x * y + 4.26323e-14 * y ** 2)
        # u = 0 if u < 0 else (u if u < w else w - 1)
        # v = 0 if v < 0 else (v if v < h else h - 1)
        try:
            v = x - (160 * y // h) + 35
            return img8_y_scaled[y][v] if 0 <= v <= minX else 255
        except:
            return 255

img8_geo = mod_img_by_fun(img8_y_scaled, geo_mod)
img8_geo = image_coordinates_transformation(img8_geo)
cv2.imwrite('./result8.png', img8_geo)
   

# (b) (25 pt) I made my own Popcat picture, although there’s no cat in it. By observing the effect shown in sample6.
# png, please design an algorithm to make sample5.png look like it as much as possible and save the output
# as result9.png. Please describe the details of your method and also provide some discussions on the designed
# method, the result, and the difference between result9.png and sample6.png, etc.
sample5 = cv2.imread('./hw2_sample_images/sample5.png', cv2.IMREAD_GRAYSCALE)
img9 = np.array(sample5)
G = np.array(img9)
center = [148, 175]
rl = 80
density = 10000
beta = 0
for r in range(rl, 0, -1):
    beta += 1
    for i in range(density):
        x = round(math.cos(i / density * math.pi * 2) * r + center[0])
        y = round(math.sin(i / density * math.pi * 2) * r + center[1])
        if r - beta <= 0:
            G[y][x] = img9[center[1]][center[0]]
        else:
            u = round(math.cos(i / density * math.pi * 2) * (r - beta) + center[0])
            v = round(math.sin(i / density * math.pi * 2) * (r - beta) + center[1])
            G[y][x] = img9[v][u]
cv2.imwrite('./result9.png', G)