import tkinter as tk
import tkinter.filedialog
import cv2
import git_pic as gp


def choose_file():  # 选择文件
    selectFileName = tk.filedialog.askopenfilename(title='选择文件')
    e.set(selectFileName)


def show(e_entry):  # 显示图片
    img = cv2.imread(e_entry.get())
    img = cv2.resize(img, None, fx=0.25, fy=0.25)
    cv2.imshow("PICTURE", img)
    cv2.waitKey(0)


def window():
    myWindow = tk.Tk()
    myWindow.title('图像处理工具')
    # 窗口尺寸
    width = 1080
    height = 720
    myWindow.geometry('1080x720')
    # 设置窗口不可变
    myWindow.resizable(width=True, height=True)
    # 获取屏幕尺寸以计算布局参数，使窗口居屏幕中央
    screenwidth = myWindow.winfo_screenwidth()
    screenheight = myWindow.winfo_screenheight()
    alignstr = '%dx%d+%d+%d' % (width, height, (screenwidth - width) / 2, (screenheight - height) / 2)
    myWindow.geometry(alignstr)

    global e
    e = tk.StringVar()  # 文本输入框
    e_entry = tk.Entry(myWindow, width=68, textvariable=e)

    b1 = tk.Button(myWindow, text='灰度图', relief='groove', width=25, height=5)
    b1.grid(row=0, column=1, padx=5, pady=5)

    b2 = tk.Button(myWindow, text='二值化', relief='groove', width=25, height=5)
    b2.grid(row=1, column=1, padx=5, pady=5)

    b3 = tk.Button(myWindow, text='反转', relief='groove', width=25, height=5)
    b3.grid(row=2, column=1, padx=5, pady=5)

    b4 = tk.Button(myWindow, text='中值滤波', relief='groove', width=25, height=5)
    b4.grid(row=3, column=1, padx=5, pady=5)

    b5 = tk.Button(myWindow, text='均值滤波', relief='groove', width=25, height=5)
    b5.grid(row=0, column=2, padx=5, pady=5)

    b6 = tk.Button(myWindow, text='高斯双边滤波', relief='groove', width=25, height=5)
    b6.grid(row=1, column=2, padx=5, pady=5)

    b7 = tk.Button(myWindow, text='Canny边缘检测', relief='groove', width=25, height=5)
    b7.grid(row=2, column=2, padx=5, pady=5)

    b8 = tk.Button(myWindow, text='另存为', relief='groove', width=25, height=5)
    b8.grid(row=3, column=2, padx=5, pady=5)

    # 选择文件控件
    sumbit_btn = tk.Button(myWindow, text='选择图片路径', command=choose_file, relief='groove', width=25, height=5)
    sumbit_btn.grid(row=5, column=1, padx=5, pady=5)

    # 展示文件控件
    show_btn = tk.Button(myWindow, text='查看图片', bg='pink', command=lambda: show(e_entry))
    show_btn.grid(row=7, column=1, padx=5, pady=5)

    myWindow.mainloop()


window()