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


# 图像浮雕处理
def fudiao(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[0:2]
    # 定义空白图像，存放图像浮雕处理之后的图片
    img1 = np.zeros((h, w), dtype=gray.dtype)
    # 通过对原始图像进行遍历，通过浮雕公式修改像素值，然后进行浮雕处理
    for i in range(h):
        for j in range(w - 1):
            # 前一个像素值
            a = gray[i, j]
            # 后一个像素值
            b = gray[i, j + 1]
            # 新的像素值,防止像素溢出
            img1[i, j] = min(max((int(a) - int(b) + 160), 0), 255)
    return img1


# 图像水彩画效果处理
def shuicai(img):
    result = cv2.stylization(img, sigma_s=60, sigma_r=0.6)
    return result



if __name__ == '__main__':
    img = cv2.imread('dog.jpg')
    # 输出显示
    plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title('origin')
    plt.subplot(1, 3, 2), plt.imshow(lunkuo(img), 'gray'), plt.title('contour')
    plt.subplot(1, 3, 3), plt.imshow(sumiao(img), 'gray'), plt.title('sketch')
    plt.show()



