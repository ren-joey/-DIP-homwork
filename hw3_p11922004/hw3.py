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
from hw3_utilities import inverter, \
    fill, opening, closing, k_means, distance
from plotter import img_plotter

# # ============================================================
# # =========== Problem 1: MORPHOLOGICAL PROCESSING ============
# # ============================================================
#
#
# # (a) (10 pt) Design a morphological processing to extract the objects’ boundaries in sample1.png and
# # output the result as result1.png.
# sample1 = cv2.imread('./hw3_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
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
#
# # (b) (10 pt) Perform hole filling on sample1.png and output the result as result2.png.
#
# img2 = np.array(sample1)
#
# filled_img = fill(img2, (0, 0), 255)
# invert_filled_img = inverter(filled_img)
# img2 = sample1 + invert_filled_img
#
# img_plotter(
#     [filled_img, invert_filled_img, img2],
#     ['filled_img', 'invert_filled_img(b)', 'G=F-b']
# )
#
# cv2.imwrite('./result2.png', img2)
#
# # filled_img = cv2.imread('./filled_img.png', cv2.IMREAD_GRAYSCALE)
# # filled_map = cv2.imread('./filled_map.png', cv2.IMREAD_GRAYSCALE)
# # filled_inverted_map = cv2.imread('./filled_inverted_map.png', cv2.IMREAD_GRAYSCALE)
# # if filled_img is None \
# #  or filled_map is None \
# #  or filled_inverted_map is None:
# #     filled_img, filled_map, filled_inverted_map = hole_filling(img1, 0, 0)
# #     cv2.imwrite('./filled_img.png', filled_img)
# #     cv2.imwrite('./filled_map.png', filled_map)
# #     cv2.imwrite('./filled_inverted_map.png', filled_inverted_map)
# #
# # img_plotter(
# #     [filled_img, filled_map, filled_inverted_map],
# #     ['filled_img', 'filled_map', 'inverted_filled_map']
# # )
#
# # cv2.imwrite('./result2.png', filled_map)
#
# # coord = [-1, -1]
# # for y in range(filled_img.shape[0]):
# #     for x in range(filled_img.shape[1]):
# #         p = filled_img[y][x]
# #         if p == 0:
# #             coord = [y, x]
# #             break
# #
# # res, _, _ = hole_filling(inverter(filled_img), 0, 0)
# #
# # img_plotter(
# #     [res, _],
# #     ['res', 'filled_map']
# # )
#
# # (c) (10 pt) Design an algorithm to count the number of objects in sample1.png. Describe the steps
# # in detail and specify the corresponding parameters.
# n_obj = 0
#
# img3 = np.array(img2)
# blank_img = np.full(img3.shape, 0)
# for y in range(img3.shape[0]):
#     for x in range(img3.shape[1]):
#         if img3[y][x] == 255:
#             img3 = fill(img3, (x, y), 0)
#             img_plotter(
#                 [img3],
#                 ['img']
#             )
#             n_obj += 1
#
# print(f'The number of objects: {n_obj}')
#
#
#
# # (d) (20 pt) Apply open operator and close operator to sample1.png and output the results as result3.
# # png and result4.png, respectively. How will it affect the result images if you change the
# # shape of the structuring element?
# # img3, img4 = np.array(sample1), np.array(sample1)
# img3, img4 = np.array(sample1), np.array(sample1)
# origin = (1, 1)
# struct_elem = np.ones((5, 5))*255
# # struct_elem = np.array([
# #     [1, 1, 1, 1, 1],
# #     [1, 1, 1, 1, 1],
# #     [1, 1, 1, 1, 1],
# #     [1, 1, 1, 1, 1],
# #     [1, 1, 1, 1, 1]
# # ]) * 255
# img3 = opening(img3, struct_elem, origin)
# img4 = closing(img4, struct_elem, origin)
#
# img_plotter(
#     [img3, img4],
#     ['opening', 'closing']
# )
# cv2.imwrite('./result3.png', img3)
# cv2.imwrite('./result4.png', img4)
#
#
# # ============================================================
# # ================ Problem 2: TEXTURE ANALYSIS ===============
# # ============================================================
#
#
# # (a) (10 pt) Perform Law’s method on sample2.png to obtain the feature vectors. Please describe
# # how you obtain the feature vectors and provide the reason why you choose it in this way.
sample2 = cv2.imread('./hw3_sample_images/sample2.png')
# img_2a = np.array(sample2)
# laws = np.array([
#     [[1, 2, 1], #1
#      [2, 4, 2],
#      [1, 2, 1]],
#     [[1, 0, -1], #2
#      [2, 0, -2],
#      [1, 0, -1]],
#     [[-1, 2, -1], #3
#      [-2, 4, -2],
#      [-1, 2, -1]],
#     [[-1, -2, -1], #4
#      [0, 0, 0],
#      [1, 2, 1]],
#     [[1, 0, -1], #5
#      [0, 0, 0],
#      [-1, 0, 1]],
#     [[-1, 2, -1], #6
#      [0, 0, 0],
#      [1, -2, 1]],
#     [[1, -2, 1], #7
#      [-2, 4, -2],
#      [-1, -2, -1]],
#     [[-1, 0, 1], #8
#      [2, 0, -2],
#      [-1, 0, 1]],
#     [[1, -2, 1], #9
#      [-2, 4, -2],
#      [1, -2, 1]]
# ])
# selected = [0, 1, 6, 7]
# dividers = np.array([1/36, 1/12, 1/12, 1/12, 1/4, 1/4, 1/12, 1/4, 1/4])
# feats = []
#
# for (i, law) in enumerate(laws):
#     if i in selected:
#         feat = filter_processor(img_2a, law, dividers[i], single_array=False)
#         feats.append(feat)
#
# feats = np.array(feats)
#
# # (b) (20 pt) Use k-means algorithm to classify each pixel with the feature vectors you obtained from
# # (a). Label the pixels of the same texture with the same color and output it as result5.png. Please
# # describe how you use the features in k-means algorithm and all the chosen parameters in detail.
#
# print(feats.shape)
#
# feat_group = np.stack((feats[0], feats[1], feats[2]), axis=2)
# print(feat_group.shape)
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# xs = feat_group[:][:][0].flatten()
# ys = feat_group[:][:][1].flatten()
# zs = feat_group[:][:][2].flatten()
# ax.scatter(xs, ys, zs)
# plt.suptitle(f'feat: {selected}')
# plt.show()
#
# # (c) (20 pt) Based on result5.png, design a method to improve the classification result and output the
# # updated result as result6.png. Describe the modifications in detail and explain the reason why.
# xs_max = np.max(xs)
# ys_max = np.max(ys)
# zs_max = np.max(zs)
# pivots = [
#     [xs_max/5*4, ys_max/5*4, zs_max/5*4],
#     [xs_max/5*3, ys_max/5*3, zs_max/5*3],
#     [xs_max/5*2, ys_max/5*2, zs_max/5*2],
#     [xs_max/5*1, ys_max/5*1, zs_max/5*1]
# ]
# op = [
#     [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]
# ]
# groups = [[], [], [], []]
# count = 0
# points = []
# for y in range(feat_group.shape[0]):
#     for x in range(feat_group.shape[1]):
#         points.append(feat_group[y][x])
#
#
# means, idx = k_means(points, pivots, clusters=4)
# print(means, idx)
#
# for y in range(img_2a.shape[0]):
#     for x in range(img_2a.shape[1]):
#         p = feat_group[y][x]
#         dist = math.inf
#         for (i, mean) in enumerate(means):
#             _d = distance(p[0], p[1], p[2], mean[0], mean[1], mean[2])
#             if _d < dist:
#                 dist = _d
#                 img_2a[y][x] = i * 85
#
# cv2.imwrite('./result5.png', img_2a)
#
# # while op != pivots and count < 5:
# #     op = pivots
# #     for y in range(feat_group.shape[0]):
# #         for x in range(feat_group.shape[1]):
# #             mini_idx = -1
# #             dist = math.inf
# #             point = feat_group[y][x]
# #             for i in range(len(pivots)):
# #                 val = math.sqrt((point[0]-pivots[i][0])**2+(point[1]-pivots[i][1])**2+(point[2]-pivots[i][2])**2)
# #                 if val < dist:
# #                     dist = val
# #                     mini_idx = i
# #             groups[mini_idx].append(point)
# #     for (i, group) in enumerate(groups):
# #         print(group)
# #         len = len(group)
# #         if len == 0:
# #             continue
# #         for j in range(0, 3):
# #             mean = sum(group[:][j]) / len
# #             pivots[i][j] = mean
# #     count += 1
# #     print(pivots)


# (d) (Bonus) TA can’t swim. Try to perform image quilting, replacing the sea in sample2.png with
# sample3.png or other texture you prefer by using the result from (c), and output it as result7.png.
# It’s allowed to utilize external libraries to help you accomplish it, but you should specify the implementation
# detail and functions you used in the report.

sample3 = cv2.imread('./hw3_sample_images/sample3.png')
img7 = np.array(sample2)
img_2d = cv2.imread('./result5.png', cv2.IMREAD_GRAYSCALE)

for y in range(img_2d.shape[0]):
    for x in range(img_2d.shape[1]):
        if img_2d[y][x] == 255:
            val = sample3[y % sample3.shape[0]][x % sample3.shape[1]]
            img7[y][x] = val

cv2.imwrite('./result7.png', img7)
