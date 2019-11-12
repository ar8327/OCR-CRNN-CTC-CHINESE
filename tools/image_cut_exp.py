import numpy as np
import tools.util_yao as util_yao
import os
import cv2
def img_precut(img):
    bg_color = util_yao.getBgColor(img)
    print("img_precut:bg_color = " + str(bg_color))
    img_bin = img.copy()

    "二值化"
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if util_yao.is_similar_color(bg_color, img[i][j], 10):
                img_bin[i][j] = np.array([0]) * 3
            else:
                img_bin[i][j] = np.array([255]) * 3

    "x方向投影"
    pixel_count = []
    print(img.shape[1])
    for i in range(img.shape[1]):
        im_t = img_bin[:, i]
        c = 0
        for j in range(img.shape[0]):
            if im_t[j][0] != 0:
                c += 1
        pixel_count.append(c)
    print("IMG PIXEL_COUNT = " + str(pixel_count))
    slices = find_slice(pixel_count, 15, 5)
    print("IMG SLICE = " + str(slices))

    img_cut = []

    for i in slices:
        img_cut.append(img[:, i[0]:i[1]])

    # for i in img_cut:
    #     cv2.imshow('A',i)
    #     cv2.waitKey(0)
    return img_cut


def find_slice(pixel_count, thresh, count_thresh):  # thresh : 最小连续阈值 #count_thresh：有效像素阈值
    prev_ptr = 0
    ptr = 0
    result = []
    for i in range(len(pixel_count)):
        if pixel_count[i] <= count_thresh and prev_ptr != ptr and abs(ptr - prev_ptr >= thresh):  # make slice
            result.append((prev_ptr, ptr))
            prev_ptr = i
        ptr = i
    if prev_ptr < len(pixel_count):
        result.append((prev_ptr, len(pixel_count)))
    return result

data_for_training = 0 #0 For test , 1 for training

if data_for_training > 0:
    dir_walk = './data_rects_train/images/'
else:
    dir_walk = './data_rects_valid/images/'

for file in os.listdir(dir_walk):
    print(dir_walk+file)
    img = cv2.imread(dir_walk+file)
    cv2.imshow('A',img)
    cv2.waitKey(0)
    result = img_precut(img)

    for slices in result:
        cv2.imshow("A",slices)
        cv2.waitKey(0)