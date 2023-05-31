"""
Created on 2023年5月23日
@author:liubochen
@description:本程序实现复古滤镜，毛玻璃滤镜，底片滤镜
"""

import cv2
import numpy as np

def nostalgic(img):
    """
    实现复古滤镜
    :param img: 待处理的图像
    :return: 复古图像
    """
    # 复制图像
    res = img.copy()

    # 将图像从BGR格式转换为RGB格式，因为变换矩阵是针对RGB格式的
    res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

    # 将图像转换为float64类型的numpy数组
    res = np.array(res, dtype=np.float64)

    # 使用矩阵对图像进行变换
    res = cv2.transform(res, np.matrix([[0.393, 0.769, 0.189],
                                        [0.349, 0.686, 0.168],
                                        [0.272, 0.534, 0.131]]))
    # 将像素值大于255的像素值截断为255
    res[np.where(res > 255)] = 255

    # 将图像转换为uint8类型的numpy数组
    res = np.array(res, dtype=np.uint8)

    # 将图像从RGB格式转换为BGR格式
    res = cv2.cvtColor(res, cv2.COLOR_RGB2BGR)
    return res


def frostglass(img, offsets):
    """
        实现毛玻璃滤镜
        :param img: 待处理的图像
        :offsets:偏移量
        :return: 毛玻璃图像
    """
    # 新建目标图像
    img1 = np.zeros_like(img)

    # 获取图像行和列
    rows, cols = img.shape[:2]

    # 像素点邻域内随机像素点的颜色替代当前像素点的颜色
    for i in range(rows - offsets):
        for j in range(cols - offsets):
            random_num = np.random.randint(0, offsets)
            img1[i, j] = img[i + random_num, j + random_num]
    result = img1[0:rows-offsets, 0:cols-offsets]
    return result


def negative(img):
    """
        实现负滤镜(底片滤镜)
        :param img: 待处理的图像
        :return: 负滤镜图像
    """
    # 获取图片大小
    img1 = img.copy()
    w, h = img1.shape[:2]
    # 逐个像素取负
    for i in range(w):
        for j in range(h):
            img1[i, j] = (255 - img1[i, j][0], 255 - img1[i, j][1], 255 - img1[i, j][2])
    return img1

if __name__ == "__main__":
    img = cv2.imread("lenna_RGB.tif")
    cv2.imshow('a', img)
    img1 = frostglass(img, 10)
    cv2.imshow('b', img1)
    cv2.waitKey(0)


