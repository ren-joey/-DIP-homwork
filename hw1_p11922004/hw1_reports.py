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
# ==================== Problem 0: Warn Up ====================
# ============================================================

oimg = cv2.imread('./hw1_sample_images/sample1.png')
oimg = cv2.cvtColor(oimg, cv2.COLOR_BGR2RGB)
oimg = np.array(oimg)

# (a) convert sample1.png to a gray-scaled image named result1.png.
# Grayscale transformation formula applied
# lightness: 			        	    (max(R, G, B) + min(R, G, B)) / 2.
# average: 			                	(R + G + B) / 3
# person’s perception of brightness:  	0.2989 * R + 0.5870 * G + 0.1140 * B
# luminosity: 			            	0.21 R + 0.72 G + 0.07 B
img = np.array(oimg)
for row in img:
    for (idx, pixel) in enumerate(row):
        row[idx] = np.sum(
            pixel * [0.2989, 0.5870, 0.1140]
        )
cv2.imwrite('./result1.png', img)

img2 = np.array(oimg)
for row in img2:
    for (idx, pixel) in enumerate(row):
        row[idx] = np.sum(
            pixel * [0.21, 0.72, 0.07]
        )

img3 = np.array(oimg)
for row in img3:
    for (idx, pixel) in enumerate(row):
        row[idx] = np.sum(pixel) / 3

img4 = np.array(oimg)
for row in img4:
    for (idx, pixel) in enumerate(row):
        max = float(np.max(pixel))
        min = float(np.min(pixel))
        row[idx] = int(math.ceil((max + min) / 2))

matplotlib.rc('font', **{'size': 6})
plt.figure(dpi=300)
plt.subplot(1, 5, 1)
plt.title('origin')
plt.imshow(oimg)
plt.axis('off')
plt.subplot(1, 5, 2)
plt.title('lightness')
plt.imshow(img4)
plt.axis('off')
plt.subplot(1, 5, 3)
plt.title('average')
plt.imshow(img3)
plt.axis('off')
plt.subplot(1, 5, 4)
plt.title('person’s perception')
plt.imshow(img)
plt.axis('off')
plt.subplot(1, 5, 5)
plt.title('luminosity')
plt.imshow(img2)
plt.axis('off')
plt.show()
plt.close()

# (b) Perform vertical flipping on result1.png and output the result as result2.png
height = img.shape[0]
G = np.full(img.shape, 0)
for i in range(math.ceil(height / 2) + 1):
    G[i], G[-i] = img[-i], img[i]
cv2.imwrite('./result2.png', G)

plt.subplot(1, 2, 1)
plt.title('origin')
plt.imshow(img)
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('vertical flipping')
plt.imshow(G)
plt.axis('off')
plt.show()
plt.close()

# ============================================================
# =============== Problem 1: IMAGE ENHANCEMENT ===============
# ============================================================

img2 = cv2.imread('./hw1_sample_images/sample2.png', cv2.IMREAD_GRAYSCALE)
img2 = np.array(img2)

# (a) (10 pt) Decrease the brightness of sample2.png
# by dividing the intensity values by 3 and output the result as result3.png.
img2a = np.array(img2)
for row in img2a:
    for (idx, pixel) in enumerate(row):
        row[idx] /= 3
cv2.imwrite('./result3.png', img2a)

# (b) (10 pt) Increase the brightness of result3.png
# by multiplying the intensity values by 3 and output the result as result4.png.
img2b = np.array(img2a)
for row in img2b:
    for (idx, pixel) in enumerate(row):
        pixel *= 3
        row[idx] = pixel if pixel <= 255 else 255
cv2.imwrite('./result4.png', img2b)

# (c) (10 pt) Plot the histograms of sample2.png, result3.png and result4.png.
# What can you observe from these three histograms?
plot_histograms(
    imgs=[img2, img2a, img2b],
    titles=['original histogram', 'P/3', 'P/3*3'],
    filename='histograms.png'
)

# (d) (10 pt) Perform global histogram equalization on sample2.png, result3.png and result4.png,
# and output the results as result5.png, result6.png and result7.png, respectively.
# Please compare these three resultant images and plot their histograms.
G5 = plot_ghe(img2)
cv2.imwrite('./result5.png', G5)

G6 = plot_ghe(img2a)
cv2.imwrite('./result6.png', G6)

G7 = plot_ghe(img2b)
cv2.imwrite('./result7.png', G7)

plot_histograms(
    imgs=[G5, G6, G7],
    titles=['original GHE', 'P/3 GHE', 'P/3*3 GHE'],
    filename='GHE_histograms.png'
)

# (e) (10 pt) Perform local histogram equalization on sample2.png, result3.png and result4.png,
# and output the results as result8.png, result9.png and result10.png, respectively.
# Please compare these three resultant images and plot their histograms.

G8 = plot_lhe(img2)
cv2.imwrite('./result8.png', G8)

G9 = plot_lhe(img2a)
cv2.imwrite('./result9.png', G9)

G10 = plot_lhe(img2b)
cv2.imwrite('./result10.png', G10)

plot_histograms(
    imgs=[G8, G9, G10],
    titles=['original LHE', 'P/3 LHE', 'P/3*3 LHE'],
    filename='LHE_histograms.png'
)

# (f) (10 pt) Design a transfer function to enhance sample2.png and output the result as result11.png.
# Try your best to obtain the most appealing result by adjusting the parameters.
# Show the parameters, the best resultant image and its corresponding histogram.
# Provide some discussions on the result as well.
parametric_img = get_parametric_img(G5)
cv2.imwrite('./result11.png', parametric_img)

# ============================================================
# ================= Problem 2: NOISE REMOVAL =================
# ============================================================

# (a) (20 pt) Design proper filters to remove the noise in sample4.png and sample5.png.
# Output the clean images as result12.png and result13.png, respectively.
# Write down details of your noise removal process in the report,
# including the filters and parameters you use.
# Please also provide some discussions about the reason why those filters and parameters are chosen.

img3 = cv2.imread('./hw1_sample_images/sample3.png', cv2.IMREAD_GRAYSCALE)
img4 = cv2.imread('./hw1_sample_images/sample4.png', cv2.IMREAD_GRAYSCALE)
img5 = cv2.imread('./hw1_sample_images/sample5.png', cv2.IMREAD_GRAYSCALE)
img5 = np.array(img5)

result12 = low_pass_filter(img4, filtersize=3, base=2, pow=2)
result12a = low_pass_filter(img4, filtersize=3, base=2, pow=3)
result12b = low_pass_filter(img4, filtersize=5, base=2, pow=4)
result12c = low_pass_filter(img4, filtersize=7, base=2, pow=5)
result12d = median_filter(img4, filtersize=5, crossmode=True)
cv2.imwrite('./result12.png', result12b)

matplotlib.rc('font', **{'size': 4})
plt.figure(dpi=300)
plt.subplot(1, 5, 1)
plt.title('3x3-2^2')
plt.imshow(result12, cmap='gray')
plt.axis('off')
plt.subplot(1, 5, 2)
plt.title('3x3-2^3')
plt.imshow(result12a, cmap='gray')
plt.axis('off')
plt.subplot(1, 5, 3)
plt.title('5x5-2^4')
plt.imshow(result12b, cmap='gray')
plt.axis('off')
plt.subplot(1, 5, 4)
plt.title('7x7-2^5')
plt.imshow(result12c, cmap='gray')
plt.axis('off')
plt.subplot(1, 5, 5)
plt.title('5x5-cross')
plt.imshow(result12d, cmap='gray')
plt.axis('off')
plt.show()
plt.close()

result12e = low_pass_filter(img4, filtersize=3, base=3, pow=3)
result12f = low_pass_filter(img4, filtersize=5, base=3, pow=3)
result12g = low_pass_filter(img4, filtersize=5, base=4, pow=3)
result12h = low_pass_filter(img4, filtersize=5, base=5, pow=3)

matplotlib.rc('font', **{'size': 4})
plt.figure(dpi=300)
plt.subplot(1, 4, 1)
plt.title('3x3-3^3')
plt.imshow(result12e, cmap='gray')
plt.axis('off')
plt.subplot(1, 4, 2)
plt.title('5x5-3^3')
plt.imshow(result12f, cmap='gray')
plt.axis('off')
plt.subplot(1, 4, 3)
plt.title('5x5-4^3')
plt.imshow(result12g, cmap='gray')
plt.axis('off')
plt.subplot(1, 4, 4)
plt.title('5x5-5^3')
plt.imshow(result12h, cmap='gray')
plt.axis('off')
plt.show()
plt.close()

result13 = median_filter(img5, filtersize=3, crossmode=False)
result13a = median_filter(img5, filtersize=5, crossmode=True)
result13b = median_filter(img5, filtersize=5, crossmode=False)
result13c = median_filter(img5, filtersize=7, crossmode=True)
result13d = median_filter(img5, filtersize=7, crossmode=False)
result13e = median_filter(img5, filtersize=9, crossmode=True)
result13f = low_pass_filter(img5, filtersize=5, base=2, pow=4)
cv2.imwrite('./result13.png', result13)

matplotlib.rc('font', **{'size': 4})
plt.figure(dpi=300)
plt.subplot(2, 4, 1)
plt.title('3x3')
plt.imshow(result13, cmap='gray')
plt.axis('off')
plt.subplot(2, 4, 5)
plt.title('5x5-cross')
plt.imshow(result13a, cmap='gray')
plt.axis('off')
plt.subplot(2, 4, 2)
plt.title('5x5')
plt.imshow(result13b, cmap='gray')
plt.axis('off')
plt.subplot(2, 4, 6)
plt.title('7x7-cross')
plt.imshow(result13c, cmap='gray')
plt.axis('off')
plt.subplot(2, 4, 3)
plt.title('7x7')
plt.imshow(result13d, cmap='gray')
plt.axis('off')
plt.subplot(2, 4, 7)
plt.title('9x9-cross')
plt.imshow(result13e, cmap='gray')
plt.axis('off')
plt.subplot(2, 4, 4)
plt.title('5x5-2^4')
plt.imshow(result13f, cmap='gray')
plt.axis('off')
plt.show()
plt.close()

# (b) (10 pt) In noise removal problems,
# PSNR is a widely used metric to present the quality of your recovered image.
# Please compute PSNR values of result12.png and result13.png, respectively,
# and provide some discussions.

# psnr12 = get_psnr(img3, result12)
# psnr13 = get_psnr(img3, result13)

psnr1 = get_psnr(img3, result12)
psnr2 = get_psnr(img3, result12a)
psnr3 = get_psnr(img3, result12b)
psnr4 = get_psnr(img3, result12c)
psnr5 = get_psnr(img3, result12d)

print(psnr1, psnr2, psnr3, psnr4, psnr5)

psnr6 = get_psnr(img3, result12e)
psnr7 = get_psnr(img3, result12f)
psnr8 = get_psnr(img3, result12g)
psnr9 = get_psnr(img3, result12h)

print(psnr6, psnr7, psnr8, psnr9)

print(
    get_psnr(img3, result13),
    get_psnr(img3, result13a),
    get_psnr(img3, result13b),
    get_psnr(img3, result13c),
    get_psnr(img3, result13d),
    get_psnr(img3, result13e),
    get_psnr(img3, result13f),
)

print(
    get_psnr(img3, img4),
    get_psnr(img3, img5)
)

print(
    # evaluate PSNR
    # clean image & uniform noise image
    get_psnr(img3, img4),
    # clean image & Salt-and-pepper image
    get_psnr(img3, img5)
)

print(
    # evaluate PSNR
    # clean image & low-pass-filter image
    get_psnr(img3, result12),
    # clean image & median-filter image
    get_psnr(img3, result13),
)