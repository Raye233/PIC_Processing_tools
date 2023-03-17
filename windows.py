# -*- coding:utf-8 -*-
import tkinter as tk
from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import *
import cv2
from git_pic import git_pic
from PIL import Image, ImageTk
import pytesseract.pytesseract
import numpy as np
import os
import sys
import dlib
import argparse
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import winsound
from Stitcher import Stitcher


class windows:
    path = None
    myWindow = tk.Tk()
    lb = tk.Label(myWindow)
    lb1 = tk.Label(myWindow)
    lb_pic = tk.Label(myWindow)

    w_box = 500
    h_box = 500
    img_open = None
    img_real = None

    @staticmethod
    def Grayscale():  # 灰度图
        image = windows.img_open
        im = git_pic(image)
        image = im.GrayScale()
        windows.img_open = image
        image = windows.win_zoom(image)
        windows.win_show(image)

    @staticmethod
    def overturn():  # 图片反转
        image = windows.img_open
        im = git_pic(image)
        image = im.overturn()
        windows.img_open = image
        image = windows.win_zoom(image)
        windows.win_show(image)

    @staticmethod
    def Binarization():  # 二值化
        image = windows.img_open
        im = git_pic(image)
        image = im.Binarization()
        windows.img_open = image
        image = windows.win_zoom(image)
        windows.win_show(image, name='二值化')

    @staticmethod
    def midwave():  # 中值滤波
        image = windows.img_open
        im = git_pic(image)
        image = im.midwave()
        windows.img_open = image
        image = windows.win_zoom(image)
        windows.win_show(image, name='中值滤波')

    @staticmethod
    def avewave():  # 均值滤波
        image = windows.img_open
        im = git_pic(image)
        image = im.avewave()
        windows.img_open = image
        image = windows.win_zoom(image)
        windows.win_show(image, name='均值滤波')

    @staticmethod
    def gauss():  # 高斯双边滤波
        image = windows.img_open
        im = git_pic(image)
        image = im.gauss()
        windows.img_open = image
        image = windows.win_zoom(image)
        windows.win_show(image, name='二值化')

    @staticmethod
    def Canny():  # Canny边缘检测
        image = windows.img_open
        im = git_pic(image)
        image = im.Canny()
        windows.img_open = image
        image = windows.win_zoom(image)
        windows.win_show(image, name='Canny边缘检测')

    @staticmethod
    def win_pic():
        path = tk.filedialog.askopenfilename()
        windows.img_real = Image.open(path)
        windows.img_open = Image.open(path)
        img_open = windows.img_open

        img = windows.win_zoom(img_open)

        windows.lb.configure(image=img, width=windows.w_box, height=windows.h_box)
        windows.lb.image = img

    @staticmethod
    def win_resize(w, h, w_box, h_box, pil_image):
        f1 = 1.0 * w_box / w
        f2 = 1.0 * h_box / h
        factor = min([f1, f2])
        res_width = int(w * factor)
        res_height = int(h * factor)
        return pil_image.resize((res_width, res_height), Image.ANTIALIAS)

    @staticmethod
    def win_zoom(image):
        w, h = image.size
        pil_image_resized = windows.win_resize(w, h, windows.w_box, windows.h_box, windows.img_open)
        img = ImageTk.PhotoImage(pil_image_resized)
        return img

    @staticmethod
    def win_show(image, **txt):
        windows.lb.configure(image=image, width=windows.w_box, height=windows.h_box)
        windows.lb.image = image
        if len(txt) == 1:
            windows.lb_pic.config(text=txt['name'], font=('黑体', 14), bg='white', fg='black')
        elif len(txt) == 2:
            windows.lb_pic.config(text=(txt['name'] + txt['value']), font=('黑体', 14), bg='white', fg='black')
        else:
            return

    @staticmethod
    def save():
        filename = asksaveasfilename(title=u'保存文件', filetypes=[("PNG", ".png"), ("JPEG", '.jpg'), ("GIF", '.gif')])
        windows.img_open.save(filename)

    @staticmethod
    # 预处理
    # 过滤轮廓唯一
    def contour_demo(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1)
        ref, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        kernel = np.ones((9, 9), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=4)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    @staticmethod
    def capture(img):
        contours = windows.contour_demo(img)
        # 轮廓唯一，以后可以扩展
        contour = contours[0]
        contour = np.float32(contour)
        img_copy = img.copy()
        approx = cv2.approxPolyDP(contour, 22, True)
        n = []
        # 生产四个角的坐标点
        for x, y in zip(approx[:, 0, 0], approx[:, 0, 1]):
            n.append((x, y))
        p1 = np.array(n, dtype=np.float32)
        # 对应点
        p2 = np.array([(0, 0), (0, 1500), (1000, 1500), (1000, 0)], dtype=np.float32)
        M = cv2.getPerspectiveTransform(p1, p2)
        result = cv2.warpPerspective(img_copy, M, (0, 0))
        # 重新截取
        result = result[:1501, :1001]
        return result

    @staticmethod
    def ocr():
        path = tk.filedialog.askopenfilename()
        src = cv2.imread(path)
        res = windows.capture(src)
        gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        txts = pytesseract.image_to_string(gray)
        print(txts)

    @staticmethod
    def connect():
        img_path1 = tk.filedialog.askopenfilename()
        img_path2 = tk.filedialog.askopenfilename()
        imageA = cv2.imread(img_path1)
        imageB = cv2.imread(img_path2)
        stitcher = Stitcher()
        (result, vis) = stitcher.stitch([imageA, imageB], showMatches=True)
        cv2.imshow("Result", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    @staticmethod
    def objecttracking():
        # 配置参数
        ap = argparse.ArgumentParser()
        ap.add_argument("-v", "--video", type=str,
                        help="path to input video file")
        ap.add_argument("-t", "--tracker", type=str, default="kcf",
                        help="OpenCV object tracker type")
        args = vars(ap.parse_args())

        # opencv已经实现了的追踪算法
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }

        # 实例化OpenCV's multi-object tracker
        trackers = cv2.MultiTracker_create()
        # vs = cv2.VideoCapture(args["video"])
        image = tk.filedialog.askopenfilename(title='选择文件')
        vs = cv2.VideoCapture(image)

        # 视频流
        while True:
            # 取当前帧
            frame = vs.read()
            # (true, data)
            frame = frame[1]
            # 到头了就结束
            if frame is None:
                break

            # resize每一帧
            (h, w) = frame.shape[:2]
            width = 600
            r = width / float(w)
            dim = (width, int(h * r))
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

            # 追踪结果
            (success, boxes) = trackers.update(frame)

            # 绘制区域
            for box in boxes:
                (x, y, w, h) = [int(v) for v in box]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 显示
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(100) & 0xFF

            if key == ord("s"):
                # 选择一个区域，按s
                box = cv2.selectROI("Frame", frame, fromCenter=False,
                                    showCrosshair=True)

                # 创建一个新的追踪器
                tracker = OPENCV_OBJECT_TRACKERS[args["tracker"]]()
                trackers.add(tracker, frame, box)

            # 退出
            elif key == 27:
                break
        vs.release()
        cv2.destroyAllWindows()

    @staticmethod
    def wink():
        def eye_aspect_ratio(eye):
            # 计算两组垂直眼界标（x，y）坐标之间的欧几里得距离
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])

            # 计算水平眼界标（x，y）坐标之间的欧几里得距离
            C = dist.euclidean(eye[0], eye[3])

            # 计算眼睛长宽比
            ear = (A + B) / (2.0 * C)

            # 返回眼睛长宽比
            return ear

        # 构造参数解析并解析参数
        ap = argparse.ArgumentParser()
        ap.add_argument("-p", "--shape-predictor", required=True,
                        help="path to facial landmark predictor")
        ap.add_argument("-v", "--video", type=str, default="",
                        help="path to input video file")
        args = vars(ap.parse_args())

        EYE_AR_THRESH = 0.25
        EYE_AR_CONSEC_FRAMES = 2

        # 初始化帧计数器和闪烁总数
        COUNTER = 0
        TOTAL = 0

        # 初始化dlib的面部检测器（基于HOG），然后创建面部界标预测器
        print("[INFO] loading facial landmark predictor...")
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(args["shape_predictor"])

        # 分别获取左眼和右眼的面部标志的索引
        (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        #  启动视频流线程
        print("[INFO] starting video stream thread...")
        vs = FileVideoStream(args["video"]).start()
        fileStream = True
        time.sleep(1.0)

        #  循环播放视频流中的帧
        while True:
            # 如果这是文件视频流，那么我们需要检查缓冲区中是否还有剩余的帧要处理
            if fileStream and not vs.more():
                break
            # 从线程视频文件流中抓取帧，调整其大小，然后将其转换为灰度通道）

            frame = vs.read()
            if frame is not None:
                frame = imutils.resize(frame, width=450)  # 调节显示的像素大小
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 0)

                # 循环人脸检测
                for rect in rects:
                    # 确定面部区域的面部界标，然后将面部界标（x，y）坐标转换为NumPy数组
                    shape = predictor(gray, rect)
                    shape = face_utils.shape_to_np(shape)

                    # 提取左眼和右眼坐标，然后使用坐标计算两只眼睛的眼睛纵横比
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEAR = eye_aspect_ratio(leftEye)
                    rightEAR = eye_aspect_ratio(rightEye)

                    # 将两只眼睛的眼睛纵横比平均在一起
                    ear = (leftEAR + rightEAR) / 2.0

                    # 计算左眼和右眼的凸包，然后可视化每只眼睛
                    leftEyeHull = cv2.convexHull(leftEye)
                    rightEyeHull = cv2.convexHull(rightEye)
                    cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
                    cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

                    # 检查眼睛宽高比是否低于眨眼阈值，如果是，则增加眨眼帧计数器
                    if ear < EYE_AR_THRESH:
                        COUNTER += 1

                    # 否则，眼睛纵横比不低于眨眼阈值
                    else:
                        # 如果闭上眼睛的次数足够多，则增加眨眼的总数
                        if COUNTER >= EYE_AR_CONSEC_FRAMES:
                            TOTAL += 1

                        # 重置眼框计数器
                        COUNTER = 0

                    # 绘制帧上眨眼的总数以及计算出的帧的眼睛纵横比
                    cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "COUNTER: {}".format(COUNTER), (140, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Frame", frame)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("q"):
                    break

        cv2.destroyAllWindows()
        vs.stop()

    @staticmethod
    def win_main():
        windows.myWindow.title('图像处理工具')
        # 窗口尺寸
        width = 1080
        height = 720
        windows.myWindow.geometry('1080x720')
        # 设置窗口不可变
        windows.myWindow.resizable(width=True, height=True)
        # 获取屏幕尺寸以计算布局参数，使窗口居屏幕中央
        screenwidth = windows.myWindow.winfo_screenwidth()
        screenheight = windows.myWindow.winfo_screenheight()
        alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
        windows.myWindow.geometry(alignstr)

        var = '请添加图片'
        windows.lb.config(
            text=var,
            fg='black',
            font=('黑体', 25),
            width=30,
            height=12,
        )

        var1 = '特殊功能'
        windows.lb1.config(
            text=var1,
            fg='black',
            font=('黑体', 25),
            width=10,
            height=12,
        )

        windows.lb.place(x=420, y=30)

        windows.lb1.place(x=20, y=150)

        btn = display('选择文件路径', 570, 600)
        btn_open = btn.display()

        btn = display('灰度图', 30, 30)
        btn_gray = btn.display()

        btn = display('二值化', 200, 30)
        btn_Binarization = btn.display()

        btn = display('反转', 30, 100)
        btn_overturn = btn.display()

        btn = display('中值滤波', 200, 100)
        btn_midwave = btn.display()

        btn = display('均值滤波', 30, 170)
        btn_avewave = btn.display()

        btn = display('高斯双边滤波', 200, 170)
        btn_gauss = btn.display()

        btn = display('Canny边缘检测', 30, 240)
        btn_canny = btn.display()

        btn = display('另存为', 200, 240)
        btn_save = btn.display()

        btn = display('ocr文本检测', 30, 380)
        btn_ocr = btn.display()

        btn = display('图像拼接特征匹配', 200, 380)
        btn_connect = btn.display()

        btn = display('深度学习目标追踪', 30, 450)
        btn_learn1 = btn.display()

        btn = display('深度学习眨眼检测', 200, 450)
        btn_learn2 = btn.display()

        btn_open.config(command=windows.win_pic)
        btn_gray.config(command=windows.Grayscale)
        btn_Binarization.config(command=windows.Binarization)
        btn_overturn.config(command=windows.overturn)
        btn_midwave.config(command=windows.midwave)
        btn_avewave.config(command=windows.avewave)
        btn_gauss.config(command=windows.gauss)
        btn_canny.config(command=windows.Canny)
        btn_save.config(command=windows.save)
        btn_ocr.config(command=windows.ocr)
        btn_connect.config(command=windows.connect)
        btn_learn1.config(command=windows.objecttracking)
        btn_learn2.config(command=windows.wink)

        windows.myWindow.mainloop()


class display(windows):
    def __init__(self, txt, x, y):
        super().__init__()
        self.txt = txt
        self.y = y
        self.x = x

    def display(self):
        btn = tk.Button(text=self.txt,
                        font=('黑体', 12),
                        fg='black',
                        bg='white',
                        activebackground='#1E90FF',
                        activeforeground='white',
                        )
        btn.place(x=self.x, y=self.y, width=150, height=50)
        return btn
