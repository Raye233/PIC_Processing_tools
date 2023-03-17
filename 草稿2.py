import tkinter as tk
import cv2
import numpy
import numpy as np
from PIL import Image
import tkinter.filedialog


class picture:

    def __init__(self):
        self.root = None
        self.img = None

    def set(self, path):
        self.root = path

    def get(self):
        return self.root

    def function1(self):    #灰度图
        path = self.root
        img = cv2.imread(path)
        img = cv2.resize(img, None, fx=0.25, fy=0.25)
        width, height = img.shape[:2][::-1]
        img_resize = cv2.resize(img, (int(width * 1), int(height * 1)), interpolation=cv2.INTER_CUBIC)
        self.img = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        cv2.imshow("img_gray", self.img)
        cv2.waitKey(0)

    def function2(self):
        pass

    def function3(self):
        pass

    def function4(self):
        pass

    def function5(self):
        pass

    def function6(self):
        pass

    def function7(self):
        pass

    def function8(self):
        selectFileName = tk.filedialog.askopenfilename(title='选择文件')
        self.set(selectFileName)

    def function9(self):
        pass


def show(abc):
    img = cv2.imread(abc)
    img = cv2.resize(img, None, fx=0.25, fy=0.25)
    cv2.imshow("PICTURE", img)
    cv2.waitKey(0)


def windows():

    PIC = picture()
    tx = tk.Tk()
    tx.title("IPT图像处理工具")
    tx.geometry("600x450")

    button1 = tk.Button(tx, text="灰度图", height=2, width=15, bg="white", fg="black", command=PIC.function1)
    button1.place(x=10, y=50)

    button2 = tk.Button(tx, text="二值化", height=2, width=15, bg="white", fg="black" )
    button2.place(x=10, y=100)

    button3 = tk.Button(tx, text="反转", height=2, width=15, bg="white", fg="black")
    button3.place(x=10, y=150)

    button4 = tk.Button(tx, text="中值率波", height=2, width=15, bg="white", fg="black")
    button4.place(x=10, y=200)

    button5 = tk.Button(tx, text="均值滤波", height=2, width=15, bg="white", fg="black")
    button5.place(x=10, y=250)

    button6 = tk.Button(tx, text="高斯双边滤波", height=2, width=15, bg="white", fg="black")
    button6.place(x=10, y=300)

    button7 = tk.Button(tx, text="Canny边缘检测", height=2, width=15, bg="white", fg="black")
    button7.place(x=10, y=350)

    button8 = tk.Button(tx, text="另存为", height=2, width=15, bg="white", fg="black")
    button8.place(x=10, y=400)

    button9 = tk.Button(tx, text="选择文件路径", height=2, width=15, bg="white", fg="black", command=PIC.function8)
    button9.place(x=375, y=375)

    button10 = tk.Button(tx, text="查看图片", height=2, width=15, bg="white", fg="black", command=lambda: show(PIC.get()))
    button10.place(x=200, y=375)

    tx.mainloop()


windows()