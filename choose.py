import tkinter as tk
import cv2
import tkinter.filedialog

class chose:

    def show(e_entry):
        img = cv2.imread(e_entry)
        cv2.imshow("PICTURE", img)
        cv2.waitKey(0)










