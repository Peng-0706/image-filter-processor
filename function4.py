"""
Created on 2023年5月21日
@author:liubochen
@description:本程序实现图像的马赛克效果以及颗粒效果
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
    img1 = img.copy()

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

            # 不需要透明度通道
            img1[block_top:block_bottom, block_left:block_right] = block_color
    return img1


def mosaic1(img, size):
    """
    图片马赛克处理函数，使用模板运算
    :param img: 待处理的图像
    :param size: 马赛克块大小
    :return: 马赛克图像
    """
    img1 = img.copy()

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

            # 进行卷积运算
            kernel = np.ones([15, 15], np.float32) / 200
            # temp_block:马赛克块
            # 目标图像深度-1，即与原图像一致
            # kernel：卷积核
            dst = cv2.filter2D(temp_block, -1, kernel=kernel)

            img1[block_top:block_bottom, block_left:block_right] = dst

    return img1


def Grain(src, level):
    """
    图片颗粒处理函数
    :param src: 待处理的图像
    :param size: 颗粒明显程度
    :return: 颗粒化图像
    """
    # 获取图像的行数、列数和通道数
    row, col, ch = src.shape

    # 如果level大于50，则将其设置为50；如果level小于0，则将其设置为0
    if level > 50:
        level = 50
    if level < 0:
        level = 0

    # 复制源图像
    result = src.copy()
    # 遍历每个像素
    for i in range(row):
        t = result[i]
        for j in range(col):
            for k in range(ch):
                # 获取当前像素的颜色值
                temp = result[i][j][k]
                # 添加随机噪声
                temp += np.random.randint(-level, level)
                # 如果像素值小于0，则将其设置为0；如果像素值大于255，则将其设置为255
                temp = max(0,min(temp, 255))
                # 更新像素的颜色值
                t[j][k] = temp
    # 返回添加噪声后的图像
    return result


def Grain1(img, level):
    """
        图片颗粒处理函数
        :param img: 待处理的图像
        :param level: 颗粒明显程度
        :return: 颗粒化图像
        """
    # 定义卷积核
    kernel = np.ones((5, 5), np.float32) / 25  # 5x5 的平均滤波器核：

    # 对图像进行卷积,使图像变得更加平滑，添加随机噪声时效果更加自然。
    img_conv = cv2.filter2D(img, -1, kernel)

    # 随机产生一些噪音
    noise = np.random.normal(-30, 40, img.shape)
    result = img_conv + noise - 30*level
    return result


if __name__ == '__main__':
    img = cv2.imread('lenna_RGB.tif')
    cv2.imshow('a', img)
    # img1 = mosaic1(img,20)

    img1 = Grain(img,40)
    # img1 = edge_sketch(img)
    cv2.imshow('b', img1)
    cv2.waitKey(0)