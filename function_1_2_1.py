import cv2
import numpy as np
from matplotlib import pyplot as plt


# 图像锐化处理(范围：0-10)
def sharp(img, deep):
    if deep == 0:
        return img
    para = deep * 0.1
    kernel = np.array([[0, -1, 0],
                       [-1, 5 + para, -1],
                       [0, -1, 0]])

    result = cv2.filter2D(img, -1, kernel)
    return result


# 图像亮度调整(范围：-10-10)
def bright(img, deep):
    if deep == 0:
        return img
    rows, cols, channels = img.shape
    dst = img.copy()
    b = deep * 20
    for i in range(rows):
        for j in range(cols):
            for c in range(3):
                color = img[i, j][c] + b
                if color > 255:  # 防止像素值越界（0~255）
                    dst[i, j][c] = 255
                elif color < 0:  # 防止像素值越界（0~255）
                    dst[i, j][c] = 0
    return dst


# 图像对比度调整(范围：0-300)
def contrast(img, deep):
    if deep == 0:
        return img
    a = deep * 0.01
    return np.uint8(np.clip((a * img), 0, 255))


# 图像边缘检测处理
def edge(img):
    # 先用高斯滤波降噪
    gray = cv2.GaussianBlur(img, (5, 5), 0)
    result = cv2.Canny(gray, 100, 200)
    return result


# 返回轮廓图像
def lunkuo(img):
    temp1 = sharp(img)
    return edge(temp1)


#  返回素描图像
def sumiao(img):
    # 灰度映射
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 通过高斯滤波过滤噪声
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0)
    # 通过canny算法提取图像轮廓
    canny = cv2.Canny(gaussian, 50, 140)
    # 对轮廓图像进行反二进制阈值化处理
    ret, result = cv2.threshold(canny, 90, 255, cv2.THRESH_BINARY_INV)
    return result


if __name__ == '__main__':
    img = cv2.imread('dog.jpg')
    # 输出显示
    plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('origin')
    plt.subplot(1, 3, 2), plt.imshow(lunkuo(img), 'gray'), plt.title('contour')
    plt.subplot(1, 3, 3), plt.imshow(sumiao(img), 'gray'), plt.title('sketch')
    plt.show()
