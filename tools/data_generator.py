# -*- coding: utf-8 -*-
#This file is a part of OCR-CRNN-CTC demo project on GitHub.
#Modified to increase sample variety - AR8327 191004


import os
import random
from PIL import Image,ImageDraw,ImageFont 

import re
from collections import Counter

import sys


#
# change this value to change the default purpose of data-generating
#
data_for_training = 1                # 1 for training, 0 for test  
# 
#
dir_images_base = './images_base'
#
if data_for_training > 0:
    dir_data_generated = './data_train'
    NumPerImage = 2000
else:
    dir_data_generated = './data_valid'
    NumPerImage = 5

dir_images_gen = dir_data_generated + '/images'
dir_contents_gen = dir_data_generated + '/contents'
#
str_dot_img_ext = '.jpg'
#
ratio_digital = 0
#
ratio_noise_line = 0.1
ratio_line_horizontal = 0.05
ratio_noise_bbox = 0.05
ratio_noise_patch = 0.05
#
sep_vertical = 1
sep_horizontal = 1
sep_div = 20
#
margin_hor = 1
margin_ver =1
#

#
words_file = '../dictionaries/numbers.txt'
#
list_sizes = [9,10,11,12,14,16,20]
#



#
if len(sys.argv) >= 2:
    #
    data_for_training = int(sys.argv[1])
    #
    if data_for_training != 0 and data_for_training != 1:
        print('The parameter should be 0 or 1.')
        print('Set to 0 by default.')
        data_for_training = 1
        #

#
#


#
list_fonts = [#'C:\Windows\fonts\Arial.ttf',
              #'C:\Windows\fonts\Arialbd.ttf',
              #'C:\Windows\fonts\courbd.ttf',
              #'C:\Windows\fonts\courbi.ttf',
              #'C:\Windows\fonts\CENTAUR.TTF',
              #'C:\Windows\fonts\BASKVILL.TTF',
              'fonts/simsun.ttc',
              'fonts/simhei.ttf',
              'fonts/simkai.ttf',
              'fonts/simfang.ttf',
              'fonts/msyh.ttc',
              'fonts/msyhbd.ttc',
              'fonts/msyhl.ttc']
#



#
if not os.path.exists(dir_data_generated): os.mkdir(dir_data_generated)
if not os.path.exists(dir_images_gen): os.mkdir(dir_images_gen)
if not os.path.exists(dir_contents_gen): os.mkdir(dir_contents_gen)
#

#
def extractWords(text): return text.split()
#
WORDS = Counter(extractWords(open(words_file, encoding="utf-8").read()))
#
#Add new word
#WORDS['history'] = 100
#WORDS['bed/room'] = 100
#
# for key in WORDS: print(key)
#
#
def generateRandomDigital():
    #
    s = ''
    a = random.randint(0, 10)
    b = random.randint(0, 999)
    c = random.randint(0, 999)
    d = random.randint(0, 99)
    # if(random.random() < 0.25):
    #     s += str(a)+"%"
    # elif(random.random() > 0.25 and random.random() <= 0.5):
    #     s += str(float(b/100))
    # else:
    #     s += str(c+d)




    #
    s = "0"


    if random.random() < 0.5: d = 0
    #
    if a > 0 and random.random() < 0.1:
        a = '%d' % a
        b = '%03d' % b
        c = '%03d' % c
        d = '%02d' % d
        #
        s = a + ',' + b + ',' + c
        #
        if random.random() < 0.5: s += '.' + d
        #
    elif b > 0 and random.random() < 0.4:
        b = '%d' % b
        c = '%03d' % c
        d = '%02d' % d
        #
        s = b + ',' + c
        #
        if random.random() < 0.5: s += '.' + d
        #
    else:
        c = '%d' % c
        d = '%02d' % d
        #
        s = c
        #
        if random.random() < 0.5: s += '.' + d
        #

    return s
    #    
#

def getRandomColor():
    r = random.randint(0, 255)
    b = random.randint(0, 255)
    g = random.randint(0, 255)
    if random.random() <= 0.5: #Fix Add Black Color
        return (0,0,0)
    else:
        return (r,g,b)

def addDigitalWords(words_dict, num):
    #
    for i in range(int(num)):
        #
        words_dict[generateRandomDigital()] = 100
        #
    #
    return words_dict
    #
#
WORDS = addDigitalWords(WORDS, len(WORDS) * ratio_digital)
#
list_words = list(WORDS.keys())
#

#
def drawLineNoise(draw, width, height):
    #
    for i in range(100):
        if random.random() > ratio_noise_line: continue
        #
        x0 = random.randint(0, width)
        x1 = random.randint(0, width)
        y0 = random.randint(0, height)
        y1 = random.randint(0, height)
        #
        if random.random() < ratio_line_horizontal: y1 = y0
        #
        line_width = random.randint(1,4)
        #
        draw.line([(x0,y0),(x1,y1)], fill = getRandomColor(), width = line_width)
        #
#
def drawPatchNoise(draw, x0, y0):
    #
    x1 = x0 + random.randint(20, sep_div)
    y1 = y0 + random.randint(20, sep_div)
    ym = (y0+y1)/2
    #
    draw.line([(x0,ym),(x1,ym)], fill = getRandomColor(), width = y1-y0)
    #
    return [x1, y1]
#
#


def generateDataSample(draw, width, height):
    #
    drawLineNoise(draw, width, height)
    #
    width -= margin_hor
    height -= margin_ver
    #
    # text
    list_bbox_text = []
    #
    x0 = margin_hor + random.randint(0, sep_div)
    y0 = margin_ver + random.randint(0, sep_div)
    #
    x1 = x0
    y1 = y0
    #
    max_th = 0
    #
    while 1:
        #
        if random.random() < ratio_noise_patch:
            ep = drawPatchNoise(draw, x0, y0)
            x0 = ep[0] + sep_horizontal + random.randint(0, sep_div)
            y1 = max(y1, ep[1])            
        #
        font_file = random.choice(list_fonts)
        text_size = random.choice(list_sizes)
        text_str = random.choice(list_words)
        #字体
        font = ImageFont.truetype(font_file, text_size)
        #文字大小
        tw,th = draw.textsize(text_str,font)
        #
        if th > max_th: max_th = th
        #
        xs = x0 # - round(tw/50.0)
        ys = y0 # + round(th/20.0)
        ym = ys + th/2.0 #round(th/2.0)
        #
        xe = x0 + tw
        ye = ys + th
        #
        x1 = xe
        y1 = max(y1, ye)
        #
        #print('chosen: %d %s' % (text_size, text_str))
        #print('%d %d %d %d' % (xs, ys, xs+tw, ys+th))
        #
        if y1 > height: break
        if x1 > width:
            x0 = margin_hor + random.randint(0, sep_div)
            y0 = y1 + sep_vertical + random.randint(0, sep_div)
            #
            x1 = x0
            y1 = y0
            #
            continue
        #        
        #清空背景
        # draw.line([(x0,ym),(x0+tw,ym)], fill = (255,255,255), width = th)
        #画文字, 设置文字位置/内容/颜色/字体
        draw.text((x0,y0), text_str, getRandomColor(), font=font)
        #
        #画方框
        if random.random() < ratio_noise_bbox:
            line_width = 1 # round(text_size/10.0)
            draw.line([(xs,ys),(xs,ye),(xe,ye),(xe,ys),(xs,ys)],
                      width=line_width, fill=getRandomColor())
        #
        list_bbox_text.append([[xs, ys,
                                xs+tw, ys+th + round(th/10.0)], text_str])
        #
        #更新
        x0 = x1 + sep_horizontal + random.randint(0, sep_div)
        #
    #
    print('max text_height: %d' % max_th)
    #
    return list_bbox_text
#

#
def getFilesInDirect(path, str_dot_ext):
    file_list = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)  
        if os.path.splitext(file_path)[1] == str_dot_ext:  
            file_list.append(file_path)
            #print(file_path)
        #
    return file_list;
    #

#
#
img_list = getFilesInDirect(dir_images_base, str_dot_img_ext)
#
NumImages = len(img_list)
count = 0
#
for imageFile in img_list:
    count += 1
    #
    filename = os.path.basename(imageFile)
    arr_split = os.path.splitext(filename)
    #
    for num in range(NumPerImage):
        #
        print('current: %d / %d, %d / %d' %(count, NumImages, num+1, NumPerImage))
        #
        #打开图片，画图
        img_draw = Image.open(imageFile)        
        #print(np.array(img_draw))
        #
        img_size = img_draw.size  # (width, height)
        draw = ImageDraw.Draw(img_draw)
        #生成样本图片
        list_bbox_text = generateDataSample(draw, img_size[0], img_size[1])
        #删除画笔
        del draw
        #
        print('generated')
        #
        #另存图片
        imageTextFile = arr_split[0] + '_generated_' + str(num) + arr_split[1]
        imageTextFile = os.path.join(dir_images_gen, imageTextFile)
        img_draw.save(imageTextFile)
        #
        #保存内容
        contentFile = arr_split[0] + '_generated_' + str(num) + '.txt'
        contentFile = os.path.join(dir_contents_gen, contentFile)
        with open(contentFile, 'w') as fp:
            for item in list_bbox_text:
                #
                fp.write('%d-%d-%d-%d|%s\n' %(item[0][0],item[0][1],item[0][2],item[0][3],item[1]))
                #
        #
    #
    
    
