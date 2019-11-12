#Data load utils
import os
import numpy as np
import core.meta as meta
import cv2

DEBUG = False
sampleFormat = '.jpg'

def getFilesInDirect(path, str_dot_ext):
    file_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.splitext(file_path)[1] == str_dot_ext:
            file_list.append(file_path)
    return file_list


def load_data(dir_data):
    #
    dir_images = dir_data + '/images'
    # dir_contents = dir_data + '/contents'
    #
    list_imgs = getFilesInDirect(dir_images, sampleFormat)
    #
    data = []
    labels = []
    #
    for img_file in list_imgs:
        #
        # label
        label_string = os.path.basename(img_file).split("_")[1]
        #
        try:
            list_chars = list(map(meta.mapChar2Order, label_string))
        except BaseException:
            print('Unknown char in file: %s' % img_file)
            continue
        if DEBUG:
            print(img_file)
            print(label_string)
            print(list_chars)
        #
        labels.append(list_chars)  #
        #
        # img_data
        img = cv2.imdecode(np.fromfile(img_file,dtype=np.uint8),cv2.IMREAD_UNCHANGED)
        img = np.array(img, dtype=np.float32) / 255 #Normalization
        img = img[:, :, 0:3]
        data.append(img)
    #
    return {'x': data, 'y': labels}
    #

if __name__ == '__main__':
    load_data('C:\\Users\\05481A\\PycharmProjects\\crnn-ctc-ocr\\tools\\data_rects_valid\\')