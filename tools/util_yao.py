

import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import spline
import math


def is_similar_color(color1,color2,thresh): #颜色对比，3通道独立
    try:
        if abs(color1[0]-color2[0]) <= thresh and \
                abs(color1[1] - color2[1]) <= thresh and \
                abs(color1[2] - color2[2]) <= thresh:
            return True
        else:
            return False
    except:
        print("is_similar_color:color data presentation error!")



def area(box):
    """获取box的面积"""
    return abs((box[0]-box[2])*(box[1]-box[3]))

def is_rect_intersect(box1,box2):
    '''判断两个文本框是否相交'''
    zx = abs(box1[0] + box1[2] - box2[0] - box2[2])
    x = abs(box1[0] - box1[2]) + abs(box2[0] - box2[2])
    zy = abs(box1[1] + box1[3] - box2[1] - box2[3])
    y = abs(box1[1] - box1[3]) + abs(box2[1] - box2[3])
    if (zx <= x and zy <= y):
        cx = max(box1[0],box2[0])
        cy = max(box1[1],box2[1])
        ex = min(box1[2], box2[2])
        ey = min(box1[3], box2[3])
        return abs((cx-ex)*(cy-ey))
    else:
        return 0

def remove_overlap_box(boxes,insect_area_ratio = 0.6):
    removedboxes = []
    for box1 in boxes:
        for box2 in boxes:
            if box1 is not box2:
                area1 = is_rect_intersect(box1, box2)
                if area1 > 0:
                    if area1 > insect_area_ratio * min(area(box1), area(box2)):
                        if area(box1) > area(box2):
                            removedboxes.append(box2)
                        else:
                            removedboxes.append(box1)
    for box in removedboxes:
        if box in boxes:
            boxes.remove(box)
    re = np.array(boxes).astype(int).tolist()
    return re,removedboxes

def getMaxRange(projection_hist):
    """获取文本框投影的局部最大值坐标范围"""
    peaks = []
    start = 0
    for i in range(1, len(projection_hist)):
        if projection_hist[i - 1] < projection_hist[i]:
            start = i
        elif projection_hist[i - 1] > projection_hist[i]:
            end = i
            if start > 0:
                peaks.append((start, end,projection_hist[i - 1]))
                start = -1
    return peaks

def getHorizontalMaxRange(projection_hist,threshol_box_min_hight= 8,threshold_allbox_width_ratio = 0.2):
    """获取文本框投影的局部最大值坐标范围,对一些文本框之间有重合的去除一些毛刺"""
    max_val = max(projection_hist)
    peaks = []
    start = 0
    for i in range(1, len(projection_hist)):
        if int(projection_hist[i - 1]) != int(projection_hist[i]):
            if i-start < threshol_box_min_hight: #文本框的宽度太窄，则不可能，直接扔掉该最大区域，跟边上的合并
                start = i
                continue
            if projection_hist[i - 1] > max_val*threshold_allbox_width_ratio:
                peaks.append((start, i,projection_hist[i - 1]))
            start = i
    return peaks


def textbox_vertical_projection(boxes,img_width):
    '''文本框的垂直投影'''
    horizonProjection = [0] * img_width
    for box in boxes:
        for x in range(int(box[0]), int(box[2])):
            horizonProjection[x] += 1
    # plt.plot(horizonProjection, color='b')
    # plt.xlim([0, w-1])
    # plt.show()
    return horizonProjection

def verify_YAxis_textbox_by_width_consistency(textboxes,thre = 3):
    """通过坐标刻度值文本框的平均宽度去除上下的可能存在的噪声,比如图题被作为y轴坐标刻度值的。
    确保textboxes是一个list并且是从上到下顺序存储"""
    if  textboxes == None:
        return None
    if len(textboxes) <= 2:
        return None,textboxes
    textboxes.sort(key = lambda x:x[1])
    mean_width_list = [ x[2]-x[0] for x in textboxes]
    mean_width = sum(mean_width_list)/len(textboxes)
    popboxes =[]
    if mean_width_list[0] > mean_width*thre:
        popboxes.append(textboxes.pop(0))
    if mean_width_list[-1] > mean_width*thre:
        popboxes.append(textboxes.pop())
    return textboxes,popboxes

def verify_YAxis_textbox_by_gap_consistency(textboxes):
    """去除上下的可能存在的噪声,比如图题被作为y轴坐标刻度值的。
    确保textboxes是一个list并且是从上到下顺序存储"""
    if len(textboxes) <= 2:
        return None,textboxes
    textboxes.sort(key = lambda x:x[1])
    gaplist =[]
    for i in range(len(textboxes)-1):
        gaplist.append(textboxes[i+1][1]-textboxes[i][3])
    meangap = sum(gaplist)/len(gaplist)
    diff2 = [abs(x-meangap) for x in gaplist ]
    mean_diff =  sum(diff2)/len(gaplist)

    popboxes = []
    #print("meangap",meangap)
    if gaplist[0] > (meangap+5)*2:
        a=textboxes.pop(0)
        popboxes.append(a)
        #print("pop first",gaplist[0])
    if gaplist[-1] > (meangap+5)*2:
        a = textboxes.pop()
        popboxes.append(a)
        #print("pop last",gaplist[-1])
    return textboxes,popboxes

def textbox_horizontal_projection(boxes,img_width,img_height):
    '''文本框的垂直投影获得每个水平位置上的文本框的长度'''
    hist = [0]*img_height
    for box in boxes:
        for i in range(int(box[1]),int(box[3])):
            hist[i] += box[2] - box[0]
    # plt.plot(hist, color='b')
    # plt.xlim([0, img_height-1])
    # plt.show()
    return hist

def get_YAxis_textbox_group(boxes,img,ratio_thresholod = 0.9):
    '''获取Y轴刻度的文本框的列表'''
    h, w, _ = img.shape
    ver_projection = textbox_vertical_projection(boxes, w)
    # 取局部极大值区域
    peak_range = getMaxRange(ver_projection)

    def fun(begin,end,val): #begin ，end为分为左中右三个区域的时候，每个区的起止x方向的位置
        text_range  = None
        for peak in peak_range:
            if peak[2] == val and  begin < (peak[0]+peak[1])//2 < end:
                text_range = peak
        if text_range:
            yaxis_textbox = []
            mid_x = (text_range[0]+text_range[1])//2
            for box in boxes:
                if box[0] < mid_x < box[2]:
                    yaxis_textbox.append(box)
            return yaxis_textbox
        return None
    # """划分三个区域，如果左中右三个区域中的最大值差别很小就认为不存在坐标轴的文本框
    #    textbox_projection是一个直方图，length直方图的bin数,ratio 是中间区间小于ratio*左边文本框个数"""
    step = w // 3
    left_num = max(ver_projection[0:step])
    mid_num = max(ver_projection[step:2 * step])
    right_num = max(ver_projection[2 * step:w-1])
    y_textbox = []
    if mid_num < left_num * ratio_thresholod and mid_num < right_num * ratio_thresholod:
        y_textbox.append(fun(0,step,left_num))
        y_textbox.append(fun(2*step,w-1, right_num))
    elif mid_num < left_num * ratio_thresholod:
        y_textbox.append(fun(0, step, left_num))
    elif mid_num < left_num * ratio_thresholod:
        y_textbox.append(fun(2 * step, w - 1, right_num))

    #文本框过滤，去除y轴上下存在的一些噪声文本框，主要适合图题等
    ret = []
    popboxes = []
    for boxes in y_textbox:
        filterbox,popbox = verify_YAxis_textbox_by_gap_consistency(boxes)
        popboxes.extend(popbox)
        if filterbox:
            temp_boxes,pop1 = verify_YAxis_textbox_by_width_consistency(filterbox)
            popboxes.extend(pop1)
            if temp_boxes != None:
                ret.append(filterbox)
    # 去掉boxes中的y轴文本框
    for box in ret:
        boxes.remove(box)
    return ret #,popboxes

def get_horizontal_line_by_vertical_gradient(img,threshold_diff = 20,threshold_width_ratio = 0.7):
    """返回水平线，但是不能保证起点终点的x坐标是否正确
    主要用于确定x轴的位置和Y轴刻度的对应位置（由于设置了比较严格的对比度，可能部分的平行刻度线没有检测出来"""
    if len(img.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        return  None
    (iH, iW) = img.shape[:2]
    output = np.zeros((iH, iW), dtype="uint8")
    for y in np.arange(1, iH):
        for x in np.arange(iW):
            output[y][x] = math.fabs(int(img[y][x]) - int(img[y-1][x]))
    vertical_count = [0]*iH
    vertical_startx = [0] * iH
    vertical_endx = [0] * iH
    for y in range(iH):
        count = 0
        startx=0
        endx =0
        for x in range(iW):
            if output[y][x] > threshold_diff:
                count += 1
                if startx == 0:
                    startx = x
                endx = x
        vertical_count[y] = count

        vertical_startx[y] = startx
        vertical_endx[y] = endx
    lines = []
    for i in range(iH):
        #print("i:", i, vertical_count[i])
        if vertical_count[i] > iH*threshold_width_ratio:
            lines.append(((vertical_startx[i],i),(vertical_endx[i],i)))
            #print(output[i][vertical_startx[i]:vertical_endx[i]])
    #删除连续的线
    if len(lines) == 0:
        return None
    gourp = []
    temp = []
    temp.append(lines[0])
    for i in range(1,len(lines)):
        if lines[i][0][1] - lines[i-1][0][1] <= 2:
            temp.append(lines[i])
        else:
            gourp.append(temp)
            temp = []
            temp.append(lines[i])
    else:
        gourp.append(temp)
    ret = []
    for g in gourp:
        if len(g) == 1:
            ret.append(g[0])
        else:
            tempy = (g[0][0][1]+g[-1][0][1])//2
            ret.append(((g[0][0][0],tempy),(g[0][1][0],tempy)))
    return ret

def get_xAxis(img):
    image = img.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)  # apertureSize参数默认其实就是3,边缘白色，其他黑色
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 160, minLineLength=20, maxLineGap=5)

    x_group = []
    # 找到所有包括x轴在内平行于x轴的线段
    for line in lines:
        x1, y1, x2, y2 = line[0]
        radius = 180 * math.atan2(y2 - y1, x2 - x1) / math.pi
        if -10 < radius < 10:
            x_group.append(list(line[0]))

    for x1, y1, x2, y2 in x_group:
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255, 255), 2)

    # x_group.sort()
    print('x_group的值：', x_group)
    cv2.imshow("x_axis_result", edges)
    cv2.waitKey(0)
    # # y值最大的那一条才是x轴
    # x_axis = [0, 0, 0, 0]
    # for temp in x_group:
    #     if (temp[1] > x_axis[1]):
    #         x_axis = temp
    #
    # # x1,y1,x2,y2=x_group[0]
    # x1, y1, x2, y2 = x_axis
    # left = min(x1, x2) - 30
    # right = max(x1, x2) + 30
    # top = min(y1, y2) - 40
    # bottom = max(y1, y2) + 40
    # roi = edges[top:bottom + 1, left:right + 1]
    # # cv2.imshow("x_axis",roi
    # # cv2.imwrite("x_axis.png",roi)
    # template = cv2.imread("x.png")
    # template = cv2.cvtColor(template, cv2.COLOR_RGB2GRAY)
    # points = []
    # w, h = template.shape
    # res = cv2.matchTemplate(roi, template, cv2.TM_CCORR_NORMED)
    # threshold = 0.81
    # loc = np.where(res >= threshold)
    # for pt in zip(*loc[::-1]):
    #     points.append((left + pt[0] + w // 2, top + pt[1] + h // 2))
    # for pt in points:
    #     cv2.circle(image, pt, 10, (255), 2)
        # print(pt)
    # points就是所有的x轴点的坐标组成的list
    # cv2.imshow("x_axis_result",image)
    #cv2.imwrite("x_axis_result.png",image)
    #return points

def get_XAxis_textbox_group(boxes,img,ratio_thresholod = 0.9):
    """获取x轴的文本框(以从左往右的方式排列)，采用投影方法"""
    h, w, _ = img.shape
    ver_projection = textbox_horizontal_projection(boxes, w,h)
    # 取局部极大值区域
    peak_range = getHorizontalMaxRange(ver_projection) # list 元素为（起始y位置,终止y位置，联合的文本宽度值)
    #由于x轴处于底部区域，所有去掉上面的peak
    removedPeak = []
    for peak in peak_range:
        if peak[1] < h//2 or peak[1]-peak[0]<8:
            removedPeak.append(peak)
    for peak in removedPeak:
        peak_range.remove(peak)
    #根据水平投影的局部峰值，将所有textbox进行分组
    groups = []
    for peak in peak_range:
        text_boxes = []
        for box in boxes:
            if peak[0]-1<(box[1]+box[3])/2< peak[1]+1 : # 在peak区间内
                text_boxes.append(box)
        if len(text_boxes) > 0: #最少有1个文本框(主要是存在合并的情况)
            groups.append(text_boxes)
    #判断group中文本框的外接矩形的中心在图像的中间区域
    removedgroup = []
    for group in groups:
        left = w
        right = 0
        for box in group:
            if box[0] < left:
                left = box[0]
            if box[2] > right:
                right = box[2]
        if not (w/3 < (left+right)/2 < 2*w/3):
            removedgroup.append(group)
    for group in removedgroup:
        groups.remove(group)
    #group中的文本框的个数和y位置排序，取最大的那个
    if len(groups) >= 1:
        groups.sort(key = lambda x:(len(x),x[0][3]))
        return groups[-1]
    else:
        return []

def verify_XAxis_textbox_group_byXAxis(img,boxes,threshold_BoxLineGap_ratio = 1.5):
    """通过获得的X轴刻度文本框和X轴，根据其相对位置关系，
    确认X轴刻度文本框和X轴是否正确匹配，正确匹配表示两者都获取正确"""
    lines = get_horizontal_line_by_vertical_gradient(img)
    xAxis_boxes = get_XAxis_textbox_group(boxes,img)
    top = img.shape[0]
    for box in xAxis_boxes:
        if box[1] < top:
            top = box[1]
    match_line = lines[0]
    min_gap = img.shape[0]
    for line in lines: #line的结构是((x1,y1),(x2,y2))
        diff = top-line[0][1]
        if diff > 0 and diff < min_gap:
            min_gap = diff
            match_line = line
    if min_gap < threshold_BoxLineGap_ratio*(xAxis_boxes[0][3]-xAxis_boxes[0][1]):
        #去掉boxes中的x轴文本框
        for box in xAxis_boxes:
            boxes.remove(box)
        return xAxis_boxes, match_line
    else:
        return [] #,None

def gauss_smooth(data,weights):
    '''对输入的一维数据（list)进行平滑，从而便于处理一些局部极大值点，便于发现波峰'''
    w_len = len(weights)
    ret = []
    for i in range(len(data)-w_len):
        ret.append(np.dot(data[i:i+w_len],weights))
    return ret

def getPeakValleyValue(data,threshold=0.01):
    '''对平滑后的数据求取局部极大值和极小值所在的位置，作为其前景色的色度值，和对应的划分区间的位置'''
    colors = []
    temp = sum(data)
    for i in range(1,len(data)-1):
        if data[i-1]<data[i]>data[i+1] and data[i]>temp*threshold:
            colors.append(i)
    if len(colors) == 0:
        return colors
    if len(colors) == 1:
        return [(colors[0]-10,colors[0],colors[0]+10)]
    ret = []
    tempMin = Min = max(data)
    thelastrightindx = 0
    for i,hue in enumerate(colors):
        tempMin = Min
        left_minIdx = 0
        if i == 0:
            for k in range(hue):
                if data[k] < tempMin:
                    tempMin = data[k]
                    left_minIdx = k
            for k in range(176,colors[-1],-1):
                if data[k] < tempMin:
                    tempMin = data[k]
                    left_minIdx = k
            if left_minIdx > hue:
                thelastrightindx = left_minIdx
                left_minIdx -= 180
            else:
                thelastrightindx = left_minIdx + 180
        else:
            for k in range(colors[i-1],colors[i]):
                if data[k] < tempMin:
                    tempMin = data[k]
                    left_minIdx = k
        if i == len(colors)-1:#最后一个
            right_minIdx = thelastrightindx
        else:
            tempMin = Min
            right_minIdx = 0
            for k in range(colors[i],colors[i+1]):
                if data[k] < tempMin:
                    tempMin = data[k]
                    right_minIdx = k
        ret.append((left_minIdx,hue,right_minIdx))
    return ret

def getFrColorByHSL(img,sat_thr = 25):
    '''sat_thr 现在还不能用，需要调试一下，整个函数用于通过H分量来获取前景的颜色值，
    通过HSL中的S值来判断背景，这里需要假设背景是白色或者黑色等灰度颜色'''
    hsl_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,w,_ = hsl_img.shape
    hsl_img_long = hsl_img.reshape((w * h), 3)
    fg_mask = np.asarray([ 255 if hsl[1] > 25 else 0 for hsl in hsl_img_long]).reshape((h,w,1))
    fg_mask = fg_mask.astype(np.uint8)
    hist = cv2.calcHist([hsl_img],[0],
        fg_mask,
        [180],
        [0,179]
    )
    hist = hist.reshape(-1)
    smooth_hist = gauss_smooth(hist,[0.3,0.4,0.3])
    return getPeakValleyValue(smooth_hist)
    # plt.plot(hist, color='b')
    # plt.xlim([0, 179])
    # plt.show()

def getManhattanDistance(c1,c2):
    '''计算两个颜色的曼哈顿距离'''
    return abs(int(c1[0])-c2[0]) + abs(int(c1[1])-c2[1]) + abs(int(c1[2])-c2[2])


class ColorCenter:
    '''用于简单的颜色聚类用的类中心对象,采用RGB颜色通道来表示类中心'''
    def __init__(self,clr):
        self._B = clr[0]*1.0
        self._G = clr[1]*1.0
        self._R = clr[2]*1.0
        self._count = 1
        self._color = clr
    def appendPoint(self,clr):
        self._B += clr[0]
        self._G += clr[1]
        self._R += clr[2]
        self._count += 1
        self._color = [int(self._B/self._count),int(self._G/self._count),int(self._R/self._count)]
    def getDistance(self,clr):
        return getManhattanDistance(self._color,clr)
    def __str__(self):
        return "B{0} G{1} R{2}".format(self._color[0],self._color[1],self._color[2])

def getBgColor(img):
    '''用于获取图像的背景颜色，三通道独立处理，假定区域最大的颜色值为背景'''
    color = ('b', 'g', 'r')
    def getMaxIndex(list1):
        list1 = list1.reshape(-1)
        max_value = 0
        max_index = 0
        for i,item in enumerate(list1):
            if item > max_value:
                max_value = item
                max_index = i
        return max_index
    max_bgr = []
    for i, col in enumerate(color):
        histr = cv2.calcHist([img], [i], None, [256], [0, 256])
        max_bgr.append(getMaxIndex(histr))
    #     plt.plot(histr, color=col)
    #     plt.xlim([0, 256])
    # plt.show()
    return max_bgr

def getFrColorByKmean(img, bgColor,threshold = 50):
    '''用于获取图像img中的前景颜色，返回颜色类中心'''
    height, width, _ = img.shape
    colors = []
    def  nearColor(color_centers,clr):
        '''对clr与每个颜色类中心的距离，然后确定最接近的颜色类中心下标和距离值'''
        min_distance = 512000000
        min_index = -1
        for i, center in  enumerate(color_centers):
            dist = center.getDistance(clr)
            if  dist < min_distance:
                min_distance = dist
                min_index = i
        return min_index, min_distance

    for i in range(height):
        for j in range(width):
            if getManhattanDistance(bgColor,img[i,j]) < threshold:
                continue
            idx,distance = nearColor(colors,img[i,j])
            if distance < threshold:
                colors[idx].appendPoint(img[i,j])
            else:
                colors.append(ColorCenter(img[i,j]))

    ret = []
    for  center in colors:
        if center._count > min(width,height)*2:
            ret.append(center)
    return ret

def getFrImage(img, bgColor,dist_threshold = 10):
    '''把背景设置成黑色，看看前景提取的效果
    @dist_threshold 是曼哈顿距离的阈值，用于判断两个颜色的相似性'''
    height,width,_ = img.shape
    ret_img = img.copy()
    for i in range(height):
        for j in range(width):
            if getManhattanDistance(img[i,j], bgColor) < dist_threshold:
                ret_img[i,j] = [0,0,0]
    return ret_img

def get_caption_textbox(boxes,img,bgColor):
    '''获得图像中的图题，可能包括多个文本框（比如来源声明）。
    主要可能在上面或者下面,主要特征是文本的高度（也就是字体）
    另外，caption一般是比较长的文本框，
    img参数主要是为了获取图像的大小等信息'''
    h,w = img.shape[:2]
    area_top = (0,h//3)
    area_middle = (h//3,2*h // 3-1)
    area_bottom = (2*h // 3, h-1)
    top_boxes = []
    mid_boxes =[]
    bottom_boxes = []
    for box in boxes: #将文本框划分到不同的区域（上，中，下）
        if area_top[0] <= box[1] <= area_top[1]:
            top_boxes.append(box)
        elif area_bottom[0] <= box[1] <= area_bottom[1]:
            bottom_boxes.append(box)
        else:
            mid_boxes.append(box)
    #统计中间区域文本框的平均高度
    max_w, max_h = 0, 0
    for box in mid_boxes:
        if box[3]-box[1] > max_h:
            max_h += box[3]-box[1]
        if box[2]-box[0] > max_w:
            max_w = box[2]-box[0]
    mean_h = max_h/len(mid_boxes)
    #根据大小获取候选box
    canidate_top_box = []
    canidate_bottom_box = []
    for box in top_boxes:
        if box[2]-box[0]> 3*(box[3]-box[1]) and box[2]-box[0] > mean_h*1.2:
            canidate_top_box.append(box)
    for box in bottom_boxes:
        if box[2]-box[0]> 3*(box[3]-box[1]) and box[2]-box[0] > mean_h*1.2:
            canidate_bottom_box.append(box)
    # for box in bottom_boxes:
    #     if  box[3]-box[1] > max_h and box[2] - box[0] > 3 * (box[3] - box[1]): #and box[2]-box[0] > max_w+5
    #         canidate_bottom_box.append(box)
    #假设caption文本框是水平的，而且是单一的灰度颜色，如果是彩色则认为不是
    # startx, endx = 0,w
    # for box in canidate_top_box[:]:
    #     colors = getFrColorByHSL(img[int(box[1]):int(box[3]),startx:endx],sat_thr=40)
    #     if len(colors) > 0:
    #         canidate_top_box.remove(box)
    # for box in canidate_bottom_box[:]:
    #     colors = getFrColorByHSL(img[int(box[1]):int(box[3]),startx:endx],sat_thr=40)
    #     if len(colors) > 0:
    #         canidate_bottom_box.remove(box)
    return canidate_top_box, canidate_bottom_box




