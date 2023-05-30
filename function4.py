import cv2


def masaike(img):

    img1 = img
    # 定义马赛克大小
    mosaic_size = 5
    # 获取图像的宽度和高度
    height, width= img1.shape[:2]
    # 将图片划分成若干个马赛克块
    num_blocks_y = height // mosaic_size  # 竖向的马赛克块数目
    num_blocks_x = width // mosaic_size  # 横向的马赛克块数目

    # 遍历每个马赛克块，对其进行模板卷积运算，实现颜色填充
    for y in range(num_blocks_y):
        for x in range(num_blocks_x):
            # 马赛克块的左上角和右下角坐标
            block_left = x * mosaic_size
            block_top = y * mosaic_size
            block_right = (x + 1) * mosaic_size
            block_bottom = (y + 1) * mosaic_size

            # 获取马赛克块内的像素
            block_pixels = img1[block_top:block_bottom, block_left:block_right]

            # 对马赛克块内的像素进行颜色填充，这里我们用马赛克块的均值来进行填充
            block_color = cv2.mean(block_pixels)[:3]
            # 不需要透明度通道
            img1[block_top:block_bottom, block_left:block_right] = block_color
    return img1
if __name__ == '__main__':
    img = cv2.imread('lenna_RGB.tif')
    cv2.imshow('a', img)
    img1 = masaike(img)

    cv2.imshow('b', img1)
    cv2.waitKey(0)