import numpy as np
inverter = np.vectorize(lambda x: 255 if x == 0 else 0)

def hole_filling(img, startX, startY):
    f = np.array(img)
    fc = inverter(f)
    filter = np.array([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ])
    extend = filter.shape[0] // 2
    G = np.full(f.shape, 0)
    labelImg = np.array(G)
    checkedMap = np.full(f.shape, 0)

    # 第一步在 labelImg 第一點畫上顏色
    labelImg[startY][startX] = 1

    count = 0
    while np.array_equal(labelImg, G) is False and count <= 1000:
        count += 1

        # 將 G 變成當前 labelImg 的樣子
        G = np.array(labelImg)

        # 計算所有為 1 點的數量
        totalPoints = (labelImg == 1).sum()
        proceededPoints = 0

        # 將 labelImg 初始化
        for y in range(G.shape[0]):
            for x in range(G.shape[1]):

                # 如果查看過的點的數量已經到達上限，則跳出這次迴圈
                if proceededPoints >= totalPoints:
                    break

                # 如果該點有顏色，且沒有被上色過，就用 filter 對這些顏色作延伸
                if G[y][x] == 1:
                    proceededPoints += 1

                    if checkedMap[y][x] == 0:
                        # 記錄這個點已經被上色過
                        checkedMap[y][x] = 1

                        for filter_y in range(filter.shape[0]):
                            for filter_x in range(filter.shape[1]):
                                filter_val = filter[filter_y][filter_x]

                                # 查看 filter 對應到 image 中的元素，如果 filter 值為 1 就貼到 image 上
                                if filter_val == 1:
                                    _y = y - filter_y + extend
                                    _x = x - filter_x + extend

                                    # 如果 _x, _y 沒有超出圖片範圍就進行標記
                                    if 0 <= _y < labelImg.shape[0] \
                                            and 0 <= _x < labelImg.shape[1]:
                                        labelImg[_y][_x] = 1

        # 把 G 跟 labelImg 作聯集
        labelImg &= fc

    G = labelImg * 255
    return (G | f), inverter(G), G