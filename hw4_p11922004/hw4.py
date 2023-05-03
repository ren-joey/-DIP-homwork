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
    filter_processor, \
    inverter
# from hw4_utilities import
from plotter import img_plotter

sample1 = cv2.imread('./hw4_sample_images/sample1.png', cv2.IMREAD_GRAYSCALE)
sample2 = cv2.imread('./hw4_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
sample3 = cv2.imread('./hw4_sample_images/sample3.png', cv2.IMREAD_GRAYSCALE)

# ============================================================
# ============== Problem 1: DIGITAL HALFTONING ===============
# ============================================================

# (a) (10 pt) According to the dither matrix I2,
# please perform dithering to obtain a binary image result1.png.
def noise_applier(img, amplitude=64):
    img = np.array(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            noise = random.randint(-amplitude, amplitude)
            p = img[y][x] + noise
            img[y][x] = 255 if p >= 255 else p
    return img
def uniform_dithering(img, amplitude=64, threshold=128):
    img = np.array(img)
    img = noise_applier(img, amplitude)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] >= threshold:
                img[y][x] = 255
            else:
                img[y][x] = 0
    return img

def I_to_T(I):
    N = I.shape[0]
    T = np.ones(I.shape)
    for y in range(N):
        for x in range(N):
            T[y][x] = 255 * (I[y][x] + 0.5) / pow(N, 2)
    return T

def dithering(img, T, amplitude=64):
    img = np.array(img)
    img = noise_applier(img, amplitude=amplitude)
    N = T.shape[0]
    for y in range(0, img.shape[0], N):
        for x in range(0, img.shape[1], N):
            for _y in range(N):
                for _x in range(N):
                    try:
                        if img[y+_y][x+_x] >= T[_y][_x]:
                            img[y+_y][x+_x] = 255
                        else:
                            img[y+_y][x+_x] = 0
                    except IndexError:
                        continue
    return img

def I_extend_by_2 (I, _I):
    target = I.tolist()
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
            m = np.ones((2, 2))
            for _y in range(2):
                for _x in range(2):
                    m[_y][_x] = 4 * I[y][x] + _I[_y][_x]
            target[x][y] = m
    target = np.array(target)
    target = target.transpose(0, 2, 1, 3).reshape((I.shape[0] * 2, I.shape[0] * 2))
    return target

def problem_1a():
    I2 = np.array([[1, 2], [3, 0]])
    T2 = I_to_T(I2)
    img1 = np.array(sample1)
    res1_16 = dithering(img1, T2, amplitude=16)
    res1_32 = dithering(img1, T2, amplitude=32)
    res1_64 = dithering(img1, T2, amplitude=64)
    res1_96 = dithering(img1, T2, amplitude=96)
    res1_128 = dithering(img1, T2, amplitude=128)

    img_plotter(
        [sample1, res1_16, res1_32, res1_64, res1_96, res1_128],
        ['origin', 'amplitude: 16', 'amplitude: 32', 'amplitude: 64', 'amplitude: 96', 'amplitude: 128'],
        fontsize=4
    )

    cv2.imwrite('./result1.png', res1_32)

# problem_1a()

# (b) (15 pt) Expand the dither matrix I2 to I256 (256 × 256) and use it to perform dithering.
# Output the result as result2.png.
# Compare result1.png and result2.png along with some discussions.

def problem_1b():
    I2 = np.array([[1, 2], [3, 0]])
    times = int(math.log(256, 2))
    I256 = np.array(I2)
    for i in range(1, times):
        I256 = I_extend_by_2(I256, I2)
    T256 = I_to_T(I256)

    img1 = np.array(sample1)
    res2_16 = dithering(img1, T256, amplitude=16)
    res2_32 = dithering(img1, T256, amplitude=32)
    res2_64 = dithering(img1, T256, amplitude=64)
    res2_96 = dithering(img1, T256, amplitude=96)
    res2_128 = dithering(img1, T256, amplitude=128)

    img_plotter(
        [sample1, res2_16, res2_32, res2_64, res2_96, res2_128],
        ['origin', 'amplitude: 16', 'amplitude: 32', 'amplitude: 64', 'amplitude: 96', 'amplitude: 128'],
        fontsize=4
    )

    cv2.imwrite('./result2.png', res2_64)

# problem_1b()

# (c) (25 pt) Perform error diffusion with Floyd-Steinberg and Jarvis’ patterns on sample1.png.
# Output the results as result3.png and result4.png, respectively.
# You may also try more patterns and show the results in your report.
# Discuss these patterns based on the results.
# You can find some masks here (from lecture slide 06. p21)

def error_diffusion(img, mode='floyd_steinberg'):
    # floyd_steinberg | jarvis | stucki
    img = np.array(img)
    if mode == 'floyd_steinberg':
        m_ltr = np.array([
            [None, None, 7/16],
            [3/16, 5/16, 1/16]
        ])
        m_rtl = np.array([
            [7/16, None, None],
            [1/16, 5/16, 3/16]
        ])
    elif mode == 'jarvis':
        m_ltr = np.array([
            [None, None, None, 7/48, 5/48],
            [3/48, 5/48, 7/48, 5/48, 3/48],
            [1/48, 3/48, 5/48, 3/48, 1/48]
        ])
        m_rtl = np.array([
            [5/48, 7/48, None, None, None],
            [3/48, 5/48, 7/48, 5/48, 3/48],
            [1/48, 3/48, 5/48, 3/48, 1/48]
        ])
    elif mode == 'stucki':
        m_ltr = np.array([
            [None, None, None, 8/42, 4/42],
            [2/42, 4/42, 8/42, 4/42, 2/42],
            [1/42, 2/42, 4/42, 2/42, 1/42]
        ])
        m_rtl = np.array([
            [4/42, 8/42, None, None, None],
            [2/42, 4/42, 8/42, 4/42, 2/42],
            [1/42, 2/42, 4/42, 2/42, 1/42]
        ])
    else:
        raise Exception("Parameter 'mode' only accepts 'floyd_steinberg', 'jarvis' or 'stucki'")

    for y in range(img.shape[0]):
        if y % 2 == 0:
            loop_range = range(img.shape[1])
            m = m_ltr
        else:
            loop_range = range(img.shape[1]-1, -1, -1)
            m = m_rtl

        for x in loop_range:
            old_p = img[y][x]
            new_p = round(old_p / 255) * 255
            e = old_p - new_p
            img[y][x] = new_p
            for _y in range(m.shape[0]): # 0-1
                for _x in range(m.shape[1]): # 0-2
                    try:
                        ratio = m[_y][_x]
                        if ratio is None:
                            continue
                        p = img[y+_y][x+_x-1] + ratio * e
                        if p > 255:
                            p = 255
                        elif p < 0:
                            p = 0
                        img[y+_y][x+_x-1] = p
                    except IndexError:
                        continue
    return img


def problem_1c():
    img1 = np.array(sample1)
    img_floyd_steinberg = error_diffusion(img1, mode='floyd_steinberg')
    img_jarvis = error_diffusion(img1, mode='jarvis')
    img_stucki = error_diffusion(img1, mode='stucki')

    img_plotter(
        [img_floyd_steinberg, img_jarvis, img_stucki],
        ['floyd_steinberg', 'jarvis', 'stucki']
    )

    cv2.imwrite('./result3.png', img_floyd_steinberg)
    cv2.imwrite('./result4.png', img_jarvis)

# problem_1c()

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
