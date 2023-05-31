"""
Created on 2023年5月26日
@author:liubochen
@description:本程序实现图像描边
"""
import cv2

def outline(img):
    """ 描边函数
    img：待处理的图像
    返回值：result：处理后的图像
    """
    # 灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Canny算法检测图像边缘
    edges = cv2.Canny(gray, 100, 200)

    # 进行二值化处理获取掩模
    _, mask = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    # 对原始图像进行滤波
    filtered = cv2.boxFilter(img, -1, (5, 5))

    # 对滤波后的图像和二值化掩膜进行与操作
    result = cv2.bitwise_and(filtered, mask)
    return result
# 显示并保存处理结果
if __name__ == '__main__':
    img = cv2.imread('lenna_RGB.tif')
    cv2.imshow('a', img)

    img1 = outline(img)
    cv2.imshow('b', img1)
    cv2.waitKey(0)
