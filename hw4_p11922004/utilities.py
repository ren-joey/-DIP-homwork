import math
import numpy as np
from plotter import img_plotter
import matplotlib.pyplot as plt
import sys
sys.setrecursionlimit(700 * 500 * 4)

inverter = np.vectorize(lambda x: 255 if x == 0 else 0)

def black_img_copier(img):
    width = img.shape[1]
    height = img.shape[0]
    G = np.full((height, width), 0, dtype=np.uint8)
    return G, width, height


def img_expend(img, filtersize, mode='odd'):
    img = np.array(img)
    expend = math.floor(filtersize / 2)
    for i in range(expend):
        if mode == 'odd':
            img = np.append(img[1:2, :], img, axis=0)
            img = np.append(img, img[-2:-1, :], axis=0)
        elif mode == 'zero':
            arr = np.full((1, img.shape[1]), 0)
            img = np.append(arr, img, axis=0)
            img = np.append(img, arr, axis=0)
    for i in range(expend):
        if mode == 'odd':
            img = np.append(img[:, 1:2], img, axis=1)
            img = np.append(img, img[:, -2:-1], axis=1)
        elif mode == 'zero':
            arr = np.full((img.shape[0], 1), 0)
            img = np.append(arr, img, axis=1)
            img = np.append(img, arr, axis=1)
    return img, expend

# def img_continuous_filling(img, _x, _y, color=255, new_color=0):
#     def fill(x, y):
#         if img[y][x] == color:
#             img[y][x] = new_color
#
#         if y > 0 and img[y-1][x] == color:
#             fill(x, y - 1)
#         if y < img.shape[0] - 1 and img[y+1][x] == color:
#             fill(x, y + 1)
#         if x > 0 and img[y][x-1] == color:
#             fill(x - 1, y)
#         if x < img.shape[1] - 1 and img[y][x+1] == color:
#             fill(x + 1, y)
#
#     fill(_x, _y)

def filter_processor(img, filter, divider=None, single_array=False):
    filtersize = filter.shape[0]
    G, width, height = black_img_copier(img)
    img, expend = img_expend(img, filtersize=filtersize)

    if divider == None:
        divider = 1 / np.sum(filter)

    for i in range(expend, expend + height):
        for j in range(expend, expend + width):
            val = np.sum(img[i - expend:i + expend + 1, j - expend:j + expend + 1] * filter) * divider
            val = abs(val)
            G[i - expend][j - expend] = val if single_array is False else np.array([val])

    return G


def unsharp_masking(img):
    filter = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    # filter = np.array([
    #     [1, -2, 1],
    #     [-2, 4, -2],
    #     [1, -2, 1]
    # ])
    return filter_processor(img, filter)


def gaussian_kernel(size, sigma, twoDimensional=True):
    """
    Creates a gaussian kernel with given sigma and size, 3rd argument is for choose the kernel as 1d or 2d
    """
    if twoDimensional:
        kernel = np.fromfunction(lambda x, y: (1 / (2 * math.pi * sigma ** 2)) * math.e ** (
                    (-1 * ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)) / (2 * sigma ** 2)), (size, size))
    else:
        kernel = np.fromfunction(lambda x: math.e ** ((-1 * (x - (size - 1) / 2) ** 2) / (2 * sigma ** 2)), (size,))
    return kernel / np.sum(kernel)


def gaussian_kernel(size, sigma=1.4, twoDimensional=True):
    if twoDimensional:
        kernel = np.fromfunction(lambda x, y: (1 / (2 * math.pi * sigma ** 2)) * math.e ** (
                    (-1 * ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)) / (2 * sigma ** 2)), (size, size))
    else:
        kernel = np.fromfunction(lambda x: math.e ** ((-1 * (x - (size - 1) / 2) ** 2) / (2 * sigma ** 2)),
                                 (size, size))
    return kernel / np.sum(kernel)


def gaussian_filter(img, size=3, sigma=1.4):
    filter = gaussian_kernel(size=size, sigma=sigma)
    return filter_processor(img, filter, divider=1)


def sobel_edge_detection(img, mode=1):
    G, w, h = black_img_copier(img)
    derivative_map = np.full((h, w), 0.)
    img, expend = img_expend(img, 3, mode='zero')
    if mode == 1:
        xFilter = [
            [-1, -2, -1],
            [0, 0, 0],
            [1, 2, 1]
        ]
        yFilter = [
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ]
    elif mode == 2:
        xFilter = [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]
        yFilter = [
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ]
    for i in range(expend, expend + h):
        for j in range(expend, expend + w):
            xg = np.sum(
                img[i - expend:i + expend + 1, j - expend:j + expend + 1] * xFilter
            )
            yg = np.sum(
                img[i - expend:i + expend + 1, j - expend:j + expend + 1] * yFilter
            )
            derivative_map[i - expend][j - expend] = math.atan(xg / (yg if yg != 0 else 1))
            G[i - expend][j - expend] = math.sqrt(xg ** 2 + yg ** 2)
    return G, derivative_map


def non_maximum_suppression(img, d, dist=2):
    G, w, h = black_img_copier(img)
    img, expend = img_expend(img, dist * 2)
    d, _ = img_expend(d, dist * 2)
    for i in range(expend, h + expend):
        for j in range(expend, w + expend):
            angle = d[i][j]
            target = img[i][j]
            xStep = round(math.cos(angle) * dist)
            yStep = round(math.sin(angle) * dist)
            p1 = [i + yStep, j + xStep]
            p2 = [i - yStep, j - xStep]
            if target > img[p1[0]][p1[1]] and target > img[p2[0]][p2[1]]:
                G[i - expend][j - expend] = target
            else:
                G[i - expend][j - expend] = 0
    return G


def hysteretic_thresholding(img, min, max):
    G = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i][j] >= max:
                G[i][j] = 255
            elif max > img[i][j] >= min:
                G[i][j] = 100
            else:
                G[i][j] = 0
    return G


def pixel_labeling(img, x, y, candidate=100, first=True):
    if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
        return
    elif img[y][x] == 255 and first is False:
        return
    if img[y][x] == candidate or (img[y][x] == 255 and first is True):
        img[y][x] = 255
        xStart = x - 1 if x > 0 else 0
        xEnd = x + 1 if x < img.shape[1] else img.shape[1]
        yStart = y - 1 if y > 0 else 0
        yEnd = y + 1 if y < img.shape[0] else img.shape[0]
        for _y in range(yStart, yEnd + 1):
            for _x in range(xStart, xEnd + 1):
                if _x != x and _y != y:
                    pixel_labeling(img, _x, _y, candidate, False)


def connected_component_labeling(img, candidate=100):
    img = np.array(img)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] == 255:
                pixel_labeling(img, x, y, candidate=candidate)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            if img[y][x] == candidate:
                img[y][x] = 0
    return img


def canny_edge_detection(img, size=3, sigma=1.4, dist=2, threshold=[30, 150], plot=False):
    img = np.array(img)
    img_gaussian = gaussian_filter(img, size=size, sigma=sigma)
    img_sobel, derivative_table = sobel_edge_detection(img_gaussian)
    img_suppression = non_maximum_suppression(img_sobel, d=derivative_table, dist=dist)
    img_thresholding = hysteretic_thresholding(img_suppression, threshold[0], threshold[1])
    img_result = connected_component_labeling(img_thresholding)
    if plot is True:
        img_plotter(
            [img_gaussian, img_sobel, img_suppression, img_thresholding, img_result],
            ['img_gaussian', 'img_sobel', 'img_suppression', 'img_thresholding', 'img_result'],
            dpi=300,
            fontsize=6
        )
    return img_result


def log_kernel(size=3, sigma=1.4):
    # kernel = np.fromfunction(lambda x, y: \
    #                 (1 / (math.pi * sigma ** 4)) \
    #                 * (1 - ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2) / (2 * sigma ** 2))
    #                 * math.e ** \
    #                 ((-1 * ((x - (size - 1) / 2) ** 2 + (y - (size - 1) / 2) ** 2)) / (2 * sigma ** 2)),
    #                 (size, size)
    #             )
    #
    kernel = np.array([
        [-1, -1, -1],
        [-1, 8, -1],
        [-1, -1, -1]
    ])
    return kernel


def gaussian_filter(img, size=3, sigma=1.4):
    filter = gaussian_kernel(size=size, sigma=sigma)
    return filter_processor(img, filter, divider=1)


def log_filtering(img, size=3, sigma=1.4):
    filter = log_kernel(size=size, sigma=sigma)
    return filter_processor(img, filter, divider=1)


# ============================================================
# ================= GEOMETRICAL MODIFICATION =================
# ============================================================

def image_coordinates_transformation(img):
    G, w, h = black_img_copier(img)
    for y in range(w):
        for x in range(h):
            G[y][x] = img[y][w - x - 1]
    return G


def mod_img_by_fun(img, fn):
    G, w, h = black_img_copier(img)
    for y in range(h):
        for x in range(w):
            G[y][x] = fn(x, y)
    return G


def line_detection_non_vectorized(edge_image, num_rhos=180, num_thetas=180, t_count=220):
    edge_height, edge_width = edge_image.shape[:2]
    edge_height_half, edge_width_half = edge_height / 2, edge_width / 2
    #
    d = np.sqrt(np.square(edge_height) + np.square(edge_width))
    dtheta = 180 / num_thetas
    drho = (2 * d) / num_rhos
    #
    thetas = np.arange(0, 180, step=dtheta)
    rhos = np.arange(-d, d, step=drho)
    #
    cos_thetas = np.cos(np.deg2rad(thetas))
    sin_thetas = np.sin(np.deg2rad(thetas))
    #
    accumulator = np.zeros((len(rhos), len(rhos)))
    #
    figure = plt.figure(figsize=(12, 12))
    subplot = figure.add_subplot(1, 1, 1)
    subplot.set_facecolor((0, 0, 0))
    #
    for y in range(edge_height):
        for x in range(edge_width):
            if edge_image[y][x] != 0:
                edge_point = [y - edge_height_half, x - edge_width_half]
                ys, xs = [], []
                for theta_idx in range(len(thetas)):
                    rho = (edge_point[1] * cos_thetas[theta_idx]) + (edge_point[0] * sin_thetas[theta_idx])
                    theta = thetas[theta_idx]
                    rho_idx = np.argmin(np.abs(rhos - rho))
                    accumulator[rho_idx][theta_idx] += 1
                    ys.append(rho)
                    xs.append(theta)
                subplot.plot(xs, ys, color="white", alpha=0.05)

    for y in range(accumulator.shape[0]):
        for x in range(accumulator.shape[1]):
            if accumulator[y][x] > t_count:
                rho = rhos[y]
                theta = thetas[x]
                a = np.cos(np.deg2rad(theta))
                b = np.sin(np.deg2rad(theta))
                x0 = (a * rho) + edge_width_half
                y0 = (b * rho) + edge_height_half
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                subplot.plot([theta], [rho], marker='o', color="yellow")
                # subplot4.add_line(mlines.Line2D([x1, x2], [y1, y2]))

    subplot.invert_yaxis()
    subplot.invert_xaxis()

    subplot.title.set_text("Hough Space")
    plt.show()
    figure.savefig('./result7.png')
