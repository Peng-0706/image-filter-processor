"""
Created on 2023年5月21日
@author:liubochen
@description:本程序实现图像的马赛克效果
"""

import cv2
import numpy as np

def mosaic(img, size):
    """
    图片马赛克处理函数，使用均值填充
    :param img: 待处理的图像
    :param size: 马赛克块大小
    :return: 马赛克图像
    """
    img1 = img


    # 定义马赛克大小
    mosaic_size = size
    if mosaic_size > 30:
        mosaic_size = 30

    # 获取图像的宽度和高度
    height, width = img1.shape[:2]

    # 将图片划分成若干个马赛克块
    num_blocks_y = height // mosaic_size  # 竖向马赛克块数
    num_blocks_x = width // mosaic_size  # 横向马赛克块数

    # 滑动马赛克块，对其进行模板卷积运算，实现颜色填充
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            # 马赛克块的左上角和右下角坐标
            block_left = x * mosaic_size
            block_top = y * mosaic_size
            block_right = (x + 1) * mosaic_size
            block_bottom = (y + 1) * mosaic_size

            # 获取马赛克块内的像素
            temp_block = img1[block_top:block_bottom, block_left:block_right]

            # 对马赛克块内的像素进行颜色填充，用马赛克块的均值填充
            block_color = cv2.mean(temp_block)[:3]

            img1[block_top:block_bottom, block_left:block_right] = block_color
    return img1

def mosaic1(img, size):
    """
    图片马赛克处理函数使用模板运算
    :param img: 待处理的图像
    :param size: 马赛克块大小
    :return: 马赛克图像
    """
    img1 = img

    # 定义马赛克大小
    mosaic_size = size
    if mosaic_size > 30:
        mosaic_size = 30

    # 获取图像的宽度和高度
    height, width = img1.shape[:2]

    # 将图片划分成若干个马赛克块
    num_blocks_y = height // mosaic_size
    num_blocks_x = width // mosaic_size

    # 滑动马赛克块，对其进行模板卷积运算，实现颜色填充
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            block_left = x * mosaic_size
            block_top = y * mosaic_size
            block_right = (x + 1) * mosaic_size
            block_bottom = (y + 1) * mosaic_size

            # 获取马赛克块内的像素
            temp_block = img1[block_top:block_bottom, block_left:block_right]

            # 进行卷积运算
            kernel = np.ones([15, 15], np.float32) / 200
            # temp_block:马赛克块
            # 目标图像深度-1，即与原图像一致
            # kernel：卷积核
            dst = cv2.filter2D(temp_block, -1, kernel=kernel)

            img1[block_top:block_bottom, block_left:block_right] = dst
    return img1

if __name__ == '__main__':
    img = cv2.imread('lenna_RGB.tif')
    cv2.imshow('a', img)
    img1 = mosaic1(img,20)

    cv2.imshow('b', img1)
    cv2.waitKey(0)