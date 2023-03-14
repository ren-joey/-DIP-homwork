import math

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_histograms(
    imgs=[],
    xlabel='Gray Level',
    ylabel='Count',
    titles=['Histogram'],
    filename='histograms.png',
):
    plt.figure(dpi=200)
    for (idx, img) in enumerate(imgs):
        plt.subplot(len(imgs), 1, idx + 1)
        plt.hist(img.flatten(), 256, [0, 255])
        plt.title(titles[idx])
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    plt.tight_layout(pad=0.5)
    plt.savefig(filename)
    plt.show()
    plt.close()

def plot_histogram(
    img,
    xlabel='Gray Level',
    ylabel='Count',
    title='Histogram',
    filename='histogram.png',
):
    plt.figure(dpi=200)
    plt.hist(img.flatten(), 256, [0, 255])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout(pad=0.5)
    plt.savefig(filename)
    plt.show()
    plt.close()

def get_histogram(img):
    histogram = np.full(256, 0)
    for row in img:
        for pixel in row:
            histogram[pixel] += 1
    return histogram

# References
# https://www.math.uci.edu/icamp/courses/math77c/demos/hist_eq.pdf
def plot_ghe(img):
    histogram = get_histogram(img)
    total_pixels = img.shape[0] * img.shape[1]
    G = np.full((600, 800), 0)
    p = np.full(256, 0, dtype=float)
    for (idx, num) in enumerate(histogram):
        p[idx] = num / total_pixels
    for (i, row) in enumerate(G):
        for (j, pixel) in enumerate(row):
            G[i][j] = math.floor(
                np.sum(p[0:(img[i][j] + 1)]) * 255
            )
    return G

def one_or_square(n, divider=2):
    return int(n / divider) if n > 1 else 1

def get_filter(
    size=3,
    base=2,
    pow=0
):
    side_width = math.floor(size / 2)
    filter = np.array([int(math.pow(base, pow))])
    for i in range(0, side_width):
        sqrt = int(filter[0] if filter[0] == 1 else filter[0] / base)
        filter = np.append([sqrt], filter, axis=0)
        filter = np.append(filter, [sqrt], axis=0)
    filter = np.array([filter])
    for i in range(0, side_width):
        new_filter = np.array(
            [[one_or_square(n, base) for n in filter[0]]]
        )
        filter = np.append(new_filter, filter, axis=0)
        filter = np.append(filter, new_filter, axis=0)
    return filter

def low_pass_filter(
        img,
        filtersize=3,
        base=2,
        pow=3
):
    img = np.array(img)
    expend = math.floor(filtersize/2)
    width = img.shape[1]
    height = img.shape[0]
    filter = get_filter(size=filtersize, pow=pow, base=base)
    G = np.full([height, width], 0)
    for i in range(expend):
        img = np.append(img[1:2, :], img, axis=0)
        img = np.append(img, img[-2:-1, :], axis=0)
    for i in range(expend):
        img = np.append(img[:, 1:2], img, axis=1)
        img = np.append(img, img[:, -2:-1], axis=1)

    for i in range(expend, expend + height):
        for j in range(expend, expend + width):
            G[i - expend][j - expend] = \
                np.sum(
                    img[i - expend:i + expend + 1, j - expend:j + expend + 1] * filter
                ) / np.sum(filter)
    return G

def median_filter(
    img,
    filtersize=3,
    crossmode=False
):
    expend = math.floor(filtersize / 2)
    width = img.shape[1]
    height = img.shape[0]
    G = np.full([height, width], 0)
    for i in range(expend):
        img = np.append(img[1:2, :], img, axis=0)
        img = np.append(img, img[-2:-1, :], axis=0)
    for i in range(expend):
        img = np.append(img[:, 1:2], img, axis=1)
        img = np.append(img, img[:, -2:-1], axis=1)

    for i in range(expend, expend + height):
        for j in range(expend, expend + width):
            if crossmode is False:
                candidates = img[i - expend:i + expend + 1, j - expend:j + expend + 1]
            else:
                candidates = np.append(
                    img[i - expend:i + expend + 1, j].flatten(),
                    img[i, j - expend:j + expend + 1].flatten(),
                    axis=0
                )
            flat = candidates.flatten()
            flat.sort()
            median = flat[math.ceil(len(flat) / 2)]
            G[i - expend][j - expend] = median
    return G

def get_psnr(img1, img2):
    height = img1.shape[0]
    width = img1.shape[1]
    mse = 0
    for (i, row) in enumerate(img1):
        for (j, pixel) in enumerate(row):
            mse += math.pow(int(img1[i][j]) - int(img2[i][j]), 2)
    mse /= width * height
    psnr = 10 * math.log10(math.pow(255, 2) / mse)
    return psnr