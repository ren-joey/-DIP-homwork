import math

import numpy as np
inverter = np.vectorize(lambda x: 255 if x == 0 else 0)

def fill(img, xy, value, thresh=0, show_border=False):
    img = np.array(img)
    x, y = xy
    try:
        background = img[y][x]
        img[y][x] = value
    except (ValueError, IndexError):
        return  # seed point outside image
    edge = {(x, y)}
    # use a set to keep record of current and previous edge pixels
    # to reduce memory consumption
    full_edge = set()
    while edge:
        new_edge = set()
        for x, y in edge:  # 4 adjacent method
            for s, t in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                # If already processed, or if a coordinate is negative, skip
                if (s, t) in full_edge or s < 0 or t < 0:
                    continue
                try:
                    p = img[t][s]
                except (ValueError, IndexError):
                    pass
                else:
                    full_edge.add((s, t))
                    fill = abs(p - background) <= thresh
                    if fill:
                        img[t][s] = value
                        new_edge.add((s, t))
                    elif show_border is True:
                        img[t][s] = 100
        full_edge = edge  # discard pixels processed
        edge = new_edge
    return img

# def hole_filling(img, startX, startY):
#     f = np.array(img)
#     fc = inverter(f)
#     filter = np.array([
#         [0, 1, 0],
#         [1, 1, 1],
#         [0, 1, 0]
#     ])
#     extend = filter.shape[0] // 2
#     G = np.full(f.shape, 0)
#     labelImg = np.array(G)
#     checkedMap = np.full(f.shape, 0)
#
#     # 第一步在 labelImg 第一點畫上顏色
#     labelImg[startY][startX] = 1
#
#     count = 0
#     while np.array_equal(labelImg, G) is False and count <= 1000:
#         count += 1
#
#         # 將 G 變成當前 labelImg 的樣子
#         G = np.array(labelImg)
#
#         # 計算所有為 1 點的數量
#         totalPoints = (labelImg == 1).sum()
#         proceededPoints = 0
#
#         # 將 labelImg 初始化
#         for y in range(G.shape[0]):
#             for x in range(G.shape[1]):
#
#                 # 如果查看過的點的數量已經到達上限，則跳出這次迴圈
#                 if proceededPoints >= totalPoints:
#                     break
#
#                 # 如果該點有顏色，且沒有被上色過，就用 filter 對這些顏色作延伸
#                 if G[y][x] == 1:
#                     proceededPoints += 1
#
#                     if checkedMap[y][x] == 0:
#                         # 記錄這個點已經被上色過
#                         checkedMap[y][x] = 1
#
#                         for filter_y in range(filter.shape[0]):
#                             for filter_x in range(filter.shape[1]):
#                                 filter_val = filter[filter_y][filter_x]
#
#                                 # 查看 filter 對應到 image 中的元素，如果 filter 值為 1 就貼到 image 上
#                                 if filter_val == 1:
#                                     _y = y - filter_y + extend
#                                     _x = x - filter_x + extend
#
#                                     # 如果 _x, _y 沒有超出圖片範圍就進行標記
#                                     if 0 <= _y < labelImg.shape[0] \
#                                             and 0 <= _x < labelImg.shape[1]:
#                                         labelImg[_y][_x] = 1
#
#         # 把 G 跟 labelImg 作聯集
#         labelImg &= fc
#
#     G = labelImg * 255
#     return (G | f), inverter(G), G

def erosion(image, struct_elem, origin):
    img_list = []
    image = np.pad(image, int(len(struct_elem)/2), 'edge')
    for img_row in range(origin[0],  len(image)-len(struct_elem)+origin[0]+1):
        for img_col in range(origin[1],  len(image[0])-len(struct_elem[0])+origin[1]+1):
            img_list.append([255 if np.array_equal(image[img_row-origin[0]:img_row-origin[0]+len(struct_elem), img_col-origin[1]:img_col-origin[1]+len(struct_elem[0])], struct_elem) else 0 ])
    return np.array(img_list).reshape(-1, len(image[0])-len(struct_elem[0])+1)

def dilation(image,  struct_elem,  origin):
    img_list = []
    image = np.pad(image, int(len(struct_elem)/2), 'edge')
    for img_row in range(origin[0],  len(image)-len(struct_elem)+origin[0]+1):
        for img_col in range(origin[1],  len(image[0])-len(struct_elem[0])+origin[1]+1):
            img_list.append([0 if np.array_equal(image[img_row-origin[0]:img_row-origin[0]+len(struct_elem), img_col-origin[1]:img_col-origin[1]+len(struct_elem[0])], np.logical_not(struct_elem)) else 255 ])
    return np.array(img_list).reshape(-1, len(image[0])-len(struct_elem[0])+1)

def opening(image, struct_elem, origin):
    image_eroded = erosion(image, struct_elem, origin)
    return dilation(image_eroded, struct_elem, origin)

# Function to perform closing on image, given image, structuring element and origin
def closing(image, struct_elem, origin):
    image_dilated = dilation(image, struct_elem, origin)
    return erosion(image_dilated, struct_elem, origin)


def distance(x1, y1, z1, x2, y2, z2):
    dist = (x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2
    dist = math.sqrt(dist)

    return dist


def k_means(points, means, clusters):
    iterations = 10  # the number of iterations
    m, n = len(points), len(points[0])

    # these are the index values that
    # correspond to the cluster to
    # which each pixel belongs to.
    index = np.zeros(m)

    # k-means algorithm.
    while (iterations > 0):

        for j in range(len(points)):
            # initialize minimum value to a large value
            minv = 1000
            temp = None

            for k in range(clusters):
                x1 = points[j][0]
                y1 = points[j][1]
                z1 = points[j][2]
                x2 = means[k][0]
                y2 = means[k][1]
                z2 = means[k][2]

                dist = distance(x1, y1, z1, x2, y2, z2)
                if dist < minv:
                    minv = dist
                    temp = k
                    index[j] = k

        for k in range(clusters):
            sumx = 0
            sumy = 0
            sumz = 0
            count = 0

            for j in range(len(points)):
                if (index[j] == k):
                    sumx += points[j][0]
                    sumy += points[j][1]
                    sumz += points[j][2]
                    count += 1

            if (count == 0):
                count = 1

            means[k][0] = float(sumx / count)
            means[k][1] = float(sumy / count)
            means[k][2] = float(sumz / count)

        iterations -= 1

    return means, index