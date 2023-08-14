# -*- coding: utf-8 -*-
# @Author : guopeng
# @Version : 2.1
# @Time : 2022/11/15 14:42:25
from time import time
import random

import numpy as np
from numpy import *
import cv2 as cv
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from colorspacious import cspace_converter
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
import pandas as pd
from PIL import Image


def seg(img):
    """
    输入为RGB格式
    """
    img = cv.cvtColor(np.asarray(img), cv.COLOR_RGB2BGR)
    mask = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    mask1 = mask.reshape((-1, 1))
    mask1[mask1 != 0] = 1
    lab = cv.cvtColor(img, cv.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)

    a = a.reshape((-1, 1))
    z = a * mask1
    idx = np.flatnonzero(z)
    a = pd.DataFrame(z).replace(0, np.NAN)
    a.dropna(inplace=True)
    a = np.float32(a)
    criteria = (cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    ret, label, center = cv.kmeans(a, k, None, criteria, 10, cv.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    # label = label.reshape((mask.shape))

    cmax = np.min(center)
    res = center[label.flatten()]
    res2 = np.zeros_like(mask1)
    res2[idx] = res
    res2[res2 != cmax] = 0
    res2[res2 == cmax] = 1
    res2 = res2.reshape((mask.shape))
    coat = cv.merge([res2, res2, res2])
    sub = 1 - coat

    coats = img * coat
    subs = img * sub
    return coats


def get_colorbar_list(cm):
    """
    计算色度条中所有的像素点
    :param cm: 使用LinearSegmentedColormap.from_list构造的色度条
    :return: [[1, 1, 1], [0.2, 0.3, 0.4], [0.4, 0.6 , 0.5]...]
    """
    color_list = [[cm(i)[0], cm(i)[1], cm(i)[2]] for i in range(cm.N)]
    np_list = np.array(color_list)
    # np.savetxt('black2read_cmp.txt',npList,fmt='%.3e') # 存储数据

    return np_list


def structure_uint_bar(colors=[(255, 255, 255), (255, 245, 230)], num=20):
    """
    构造阶段色条
    """
    for i in range(len(colors)):
        colors[i] = list(colors[i])
        for j in range(len(colors[i])):
            colors[i][j] = colors[i][j] / 256
        colors[i] = tuple(colors[i])

    cmap_name = 'my_cmap'
    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=num)

    # 获取色条中的每个像素
    color_list = [[cm(i)[0], cm(i)[1], cm(i)[2]] for i in range(cm.N)]
    np_list = list(color_list)

    return list(np_list)


def structure_bar():
    """
    构造色条
    """

    # 配置色条各分段的参数
    white_bar1 = structure_uint_bar(
        colors=[[255, 251, 255], (255, 236, 249), (255, 235, 236), (255, 235, 228), (255, 228, 220), (255, 228, 216)],
        num=10)
    white_bar2 = structure_uint_bar(
        colors=[[255, 223, 234], [255, 220, 226], [255, 213, 220], [255, 204, 220], [255, 200, 208], [255, 208, 196],
                [255, 201, 188], [255, 194, 183], (255, 183, 161)], num=15)
    yellowish_bar1 = structure_uint_bar(colors=[(255, 228, 216), (255, 228, 210), (255, 230, 200), (255, 220, 200)],
                                        num=12)
    yellowish_bar2 = structure_uint_bar(colors=[(255, 210, 190), (255, 189, 190), (255, 183, 181), (255, 175, 183)],
                                        num=13)
    yellow_bar1 = structure_uint_bar(colors=[(255, 220, 200), [255, 220, 180], (255, 215, 170), (255, 215, 150)],
                                     num=13)
    yellow_bar2 = structure_uint_bar(colors=[(255, 229, 172), [255, 204, 170], [255, 204, 144], [255, 204, 134]],
                                     num=12)
    brown_bar1 = structure_uint_bar(
        colors=[(255, 220, 173), (255, 220, 160), (255, 220, 145), (255, 210, 145), (255, 200, 140)], num=10)
    brown_bar2 = structure_uint_bar(
        colors=[(255, 196, 155), (255, 196, 130), (255, 192, 114), (255, 173, 114), (255, 152, 101)], num=10)
    brown_bar3 = structure_uint_bar(colors=[(255, 174, 146), (255, 174, 120), (255, 128, 82), (255, 112, 56)], num=5)
    # gray_bar = structure_uint_bar(colors=[(255, 200, 140), (255, 190, 100),(99, 71, 75)])

    colors = white_bar1 + white_bar2 + yellowish_bar2[1:] + yellowish_bar1[1:] + yellow_bar1[1:] + yellow_bar2[1:] \
             + brown_bar1[1:] + brown_bar2[1:] + brown_bar3[0:]

    cmap_name = 'my_cmap'

    cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=100)
    norm = mpl.colors.Normalize(vmin=0, vmax=10)

    # 绘制色条
    #     fig, ax = plt.subplots(figsize=(6, 1))
    #     fig.subplots_adjust(bottom=0.5)
    #     fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cm),
    #                  cax=ax, orientation='horizontal', label='coating color')

    return cm


def eucliDist(A, B):
    """
    计算欧氏距离
    :param A,B:list
    """
    return math.sqrt(sum([(a - b) ** 2 for (a, b) in zip(A, B)]))


def norm_point(point=[255, 255, 255]):
    """
    对每一个像素点进行归一化，若亮度过低则除以255
    """
    max_channel = max(point)
    if point[0] > 50:
        for idx, p in enumerate(point):
            point[idx] = float(p) / max_channel
    else:
        for idx, p in enumerate(point):
            point[idx] = float(p) / 255
    return point


def cal_min_distance_ratio(point, np_list):
    """
    计算点point在np_list中离得最近的点，返回位置(0-1)以及色条上的像素点
    :param np_list:色条上的点
    :param point:采样点，格式是归一化[0.6, 0.2, 0.2]
    :return:[黄度值(一位小数)、对应的色条点、最小的距离]
    """
    ecu_list = []
    for p in np_list:
        ecu_list.append(eucliDist(p, point))
    nearest_distance = min(ecu_list)
    idx = ecu_list.index(nearest_distance)
    ratio = idx / len(ecu_list)
    ratio = ratio * 10

    return round(ratio, 1), list(np_list[idx]), nearest_distance


def mean_value(arr):
    """
    计算非零像素点每个通道的均值
    :param arr:图片某一行的像素，每个像素点都是三维通道
    """
    exist = (arr != 0)
    return arr.sum() / exist.sum()


def sample_pixel(img, np_list, distance_threshold=0.03):
    """
    1、在舌苔图片上随机选取一千个(R+G+B)>50的点
    2、筛选色条附近的点，并计算其在色条上的位置，只保留距离小于阈值的点
    3、选取在色条上前30%的点的平均值作为采样点
    :param img:舌象分割后的舌苔图片
    :param np_list:色条列表(归一化后的像素)
    :param threshold:离色条的距离阈值
    :return :归一化后的采样点
    """
    # 1、随机选取1000个(R+G+B)>100的点
    points = img.reshape(-1, 3)
    # points = points[[np.sum(points[i]) > 100 and points[i][2] > 20 for i in range(points.shape[0])], :]
    points = points[np.random.choice(points.shape[0], size=2000, replace=True), :]

    # 2、筛选色条附近的点，并计算其在色条上的位置，只保留距离小于阈值的点
    points = points.astype(np.float32)
    points = points[:, ::-1]
    points_norm = np.apply_along_axis(norm_point, 1, points)
    points_info = np.apply_along_axis(cal_min_distance_ratio, 1, points_norm, np_list)
    # 筛选小于阈值的点
    points = points[[points_info[i][2] < distance_threshold for i in range(points.shape[0])], :]
    points_info = points_info[[points_info[i][2] < distance_threshold for i in range(points_info.shape[0])], :]

    # 3、选取在色条上前30%的点的平均值作为采样点
    ratios = points_info[:, 0]
    arr_index = ratios.argsort()
    points = points[arr_index][int(points.shape[0] * 0.6):int(points.shape[0] * 1), :]
    sample = np.mean(points, axis=0)

    return sample


def coated_tongue_color_quantization(img):
    """
    依次调用相关函数，完成苔色量化
    """
    # 获取色条
    cm = structure_bar()
    # 获取色条值
    np_list = get_colorbar_list(cm)

    # 获取舌苔图片
    img = seg(img)
    # 获取采样点
    sample_point = sample_pixel(img, np_list)
    sample_point = list(sample_point)

    # 输出在色条上的位置
    quantized_value, pixel, nearest_distance = cal_min_distance_ratio(sample_point, np_list)

    # 对输出值做范围限定，防止过大或国小
    if quantized_value <= 0.1:
        quantized_value = 0.1

    return quantized_value


def revise(quantized_value, index):
    """
    根据分类结果index修正量化值quantized_value
    """
    interval = [0, 2.5, 5, 7.5, 10]
    if index == 4:
        return None
    else:
        restrict_interval = [interval[index], interval[index + 1]]
        if quantized_value < restrict_interval[0]:
            quantized_value = restrict_interval[0] + random.uniform(-0.11, 0.2)
        elif quantized_value > restrict_interval[1]:
            quantized_value = restrict_interval[1] + random.uniform(-0.2, 0.1)
        else:
            pass
    if quantized_value < 0:
        quantized_value = random.uniform(0.001, 0.1)
    if quantized_value > 10:
        quantized_value = random.uniform(9.9, 9.95)

    return quantized_value


if __name__ == '__main__':
    # 加载图片为RGB格式
    picture_root = r'D:\MyCodes\pythonProject\coated_tongue_color\datas\data2all\category\3\20220510145856-1.png'
    img = Image.open(picture_root).convert('RGB')
    value = coated_tongue_color_quantization(img)
    print(value)
