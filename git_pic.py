import cv2
import numpy as np
from PIL import Image


class conversion(object):
    def __init__(self, image):
        self.image = image

    def PIL_opencv(self):
        img = cv2.cvtColor(np.asarray(self.image), cv2.COLOR_RGB2BGR)
        return img

    def opencv_PIL(self):
        img = Image.fromarray(cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB))
        return img


class git_pic(conversion):
    def GrayScale(self):
        img = git_pic.PIL_opencv(self)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.image = gray
        return git_pic.opencv_PIL(self)

    def overturn(self):     # 图片反转
        img = git_pic.PIL_opencv(self)
        height, width, deep = img.shape
        dst1 = np.zeros((height, width, deep), np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                (b, g, r) = img[i, j]
                dst1[i, j] = (255 - b, 255 - g, 255 - r)
        self.image = dst1
        return git_pic.opencv_PIL(self)

    def Binarization(self):     # 二值化
        img = git_pic.PIL_opencv(self)
        GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        GrayImage = cv2.medianBlur(GrayImage, 5)
        img = cv2.adaptiveThreshold(GrayImage, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)
        self.image = img
        return git_pic.opencv_PIL(self)

    def midwave(self):     #中值滤波
        img = git_pic.PIL_opencv(self)
        img_medianBlur = cv2.medianBlur(img, 5)
        self.image = img_medianBlur
        return git_pic.opencv_PIL(self)

    def avewave(self):        #均值滤波
        img = git_pic.PIL_opencv(self)
        img_Blur = cv2.blur(img, (5, 5))
        self.image = img_Blur
        return git_pic.opencv_PIL(self)

    def gauss(self):        #高斯双边滤波
        img = git_pic.PIL_opencv(self)
        img_bilateralFilter = cv2.bilateralFilter(img, 40, 75, 75)
        self.image = img_bilateralFilter
        return git_pic.opencv_PIL(self)

    def Canny(self):   #Canny边缘检测
        img = git_pic.PIL_opencv(self)
        lenna = cv2.GaussianBlur(img, (5, 5), 0)
        canny = cv2.Canny(lenna, 50, 150)
        self.image = canny
        return git_pic.opencv_PIL(self)




