import math

import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from utilities import \
                    plot_histograms,\
                    plot_histogram, \
                    plot_ghe, \
                    low_pass_filter, \
                    median_filter, \
                    get_psnr


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

histogram2, histogram2a, histogram2b = \
                                    np.full(256, 0), \
                                    np.full(256, 0), \
                                    np.full(256, 0)

# (d) (10 pt) Perform global histogram equalization on sample2.png,
# result3.png and result4.png, and output the results as result5.png, result6.png and result7.png, respectively.
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

print()

# (f) (10 pt) Design a transfer function to enhance sample2.png and output the result as result11.png.
# Try your best to obtain the most appealing result by adjusting the parameters.
# Show the parameters, the best resultant image and its corresponding histogram.
# Provide some discussions on the result as well.

print()

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

result12 = low_pass_filter(img4, filtersize=3, base=4, pow=4)
cv2.imwrite('./result12.png', result12)

result13 = median_filter(img5, filtersize=5, crossmode=True)
cv2.imwrite('./result13.png', result13)

psnr12 = get_psnr(img3, result12)
psnr13 = get_psnr(img3, result13)


# (b) (10 pt) In noise removal problems,
# PSNR is a widely used metric to present the quality of your recovered image.
# Please compute PSNR values of result12.png and result13.png, respectively,
# and provide some discussions.

print()
