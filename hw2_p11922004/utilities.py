import math
import numpy as np

def black_img_copier (img):
    width = img.shape[1]
    height = img.shape[0]
    G = np.full([height, width], 0)
    return G, width, height
def img_expend (img, filtersize, mode='odd'):
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


def filter_processor(img, filter):
    filtersize = filter.shape[0]
    G, width, height = black_img_copier(img)
    img, expend = img_expend(img, filtersize=filtersize)

    for i in range(expend, expend + height):
        for j in range(expend, expend + width):
            G[i - expend][j - expend] = \
                np.sum(
                    img[i - expend:i + expend + 1, j - expend:j + expend + 1] * filter
                ) / np.sum(filter)
    return G

def unsharp_masking (img):
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

def noise_reduction (img):
    filter = np.array([
        [2, 4, 5, 4, 2],
        [4, 9, 12, 9, 4],
        [5, 12, 15, 12, 5],
        [4, 9, 12, 9, 4],
        [2, 4, 5, 4, 2]
    ])
    return filter_processor(img, filter)
def sobel_edge_detection(img):
    G, w, h = black_img_copier(img)
    img, expend = img_expend(img, 3, mode='zero')
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
    for i in range(expend, expend + h):
        for j in range(expend, expend + w):
            xg = np.sum(
                img[i - expend:i + expend + 1, j - expend:j + expend + 1] * xFilter
            )
            xy = np.sum(
                img[i - expend:i + expend + 1, j - expend:j + expend + 1] * yFilter
            )
            G[i - expend][j - expend] = math.sqrt(xg ** 2 + xy ** 2)
    return G
