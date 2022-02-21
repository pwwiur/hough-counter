import os
import traceback

import cv2
import matplotlib.pyplot as plt
import xml.etree.cElementTree as ET
from PIL import Image, ImageTk
import numpy as np

import tkinter as tk
from tkinter.filedialog import askdirectory

label = "circle"
files = []
file_index = -1
data_path = ""
config = {}
to_be_saved = {}


def init():
    global config
    config = {
        "img_default_width": 600,
        "img_default_height": 400,
        "minDist": 22,
        "minRadius": 22,
        "maxRadius": 50,
        "gaussian_default": 5,
        "median_default": 7,
        "threshold": 0
    }


def filter(img):
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, (config["gaussian_default"], config["gaussian_default"]), 0)
    gray_image = cv2.medianBlur(gray_image, config["median_default"])
    # gray_image= cv2.GaussianBlur(gray_image, (7, 7), 0)
    ret, th1 = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 5)
    th3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 5)

    # https://www.pyimagesearch.com/2021/04/28/opencv-morphological-operations/#:~:text=Morphological%20operations%20are%20simple%20transformations,and%20structures%20inside%20of%20images.
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(th2, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations=1)

    imgray = cv2.Canny(erosion, 30, 100)

    circles = cv2.HoughCircles(imgray, method=cv2.HOUGH_GRADIENT, dp=1, minDist=config["minDist"], param1=50, param2=30,
                               minRadius=config["minRadius"], maxRadius=config["maxRadius"])

    if circles is not None:
        circles = np.uint16(np.around(circles))
        blue = (0, 0, 255)
        green = (0, 255, 0)
        for i in circles[0, :]:
            # draw the outer circle
            # cv2.circle(img,(i[0],i[1]),i[2], green, 2)
            cv2.rectangle(img, (i[0] - i[2] - config["threshold"], i[1] - i[2] - config["threshold"]),
                          (i[0] + i[2] + config["threshold"], i[1] + i[2] + config["threshold"]), green, 2)
            # draw the center of the circle
            cv2.circle(img, (i[0], i[1]), 2, blue, 3)

        # plt.title(title)
        # plt.imshow(img, cmap='gray', interpolation='bicubic')
        # plt.show()
        # plt.title(title)
        # plt.imshow(imgray, cmap='gray', interpolation='bicubic')
        # plt.show()

    return img, circles


def set_image(img):
    img_box.configure(image=img)
    img_box.image = img


def load_config(directory, file):
    global config
    try:

        xml_file = directory + "/" + os.path.splitext(file)[0] + ".xml"
        if os.path.exists(xml_file):
            root = ET.parse(xml_file).getroot()
            settings = root.find("config")
            if settings:

                for setting in settings:
                    config[setting.tag] = int(setting.text)
            else:
                init()

    except Exception as e:
        print("Error in loading config", e)


def load_image():
    global to_be_saved
    file_path = data_path + "/" + files[file_index]
    print("Loading: ", file_path)
    load_config(data_path, files[file_index])
    arr_img = cv2.imread(file_path)
    if arr_img.shape[0] > arr_img.shape[1]:
        arr_img = cv2.resize(arr_img, (
        config["img_default_width"], int((config["img_default_width"] / arr_img.shape[0]) * arr_img.shape[1])))
    else:
        arr_img = cv2.resize(arr_img, (
        int((config["img_default_height"] / arr_img.shape[1]) * arr_img.shape[0]), config["img_default_height"]))
    title = "g" + str(config["gaussian_default"]) + "m" + str(config["median_default"])
    show_info("Processing " + title, False)
    try:
        to_be_saved["img"] = np.copy(arr_img)
        cv_img, circles = filter(arr_img)
        tk_img = ImageTk.PhotoImage(image=Image.fromarray(cv_img))
        show_info(title + " Circles found: " + str(0 if circles is None else len(circles[0, :])))
        set_image(tk_img)
        to_be_saved["file"] = files[file_index]
        to_be_saved["circles"] = circles
    except Exception as e:
        show_info("Error occurred for " + file_path + "(" + title + ")", False)
        raise e


def scales_onchange(event):
    global config
    config["gaussian_default"] = sld_gaussian.get()
    config["median_default"] = sld_median.get()
    config["minRadius"] = sld_minsize.get()
    config["maxRadius"] = sld_maxsize.get()
    config["minDist"] = sld_mindist.get()
    config["threshold"] = sld_threshold.get()
    load_image()


def show_info(msg, remove_image=True):
    if remove_image:
        set_image("")
    info_box.config(text=msg)


def save():
    if "circles" in to_be_saved and "file" in to_be_saved and "img" in to_be_saved and to_be_saved["circles"] is not None:
        file = to_be_saved["file"]
        img = to_be_saved["img"]
        circles = to_be_saved["circles"]

        directory = data_path
        if not os.path.exists(directory):
            os.makedirs(directory)

        path = directory + "/" + file

        cv2.imwrite(path, img)

        annotation = ET.Element("annotation")
        ET.SubElement(annotation, "folder").text = directory
        ET.SubElement(annotation, "filename").text = file
        ET.SubElement(annotation, "path").text = path

        source = ET.SubElement(annotation, "source")
        ET.SubElement(source, "database").text = "Real_World_Shapes_v1.0"

        size = ET.SubElement(annotation, "size")
        ET.SubElement(size, "width").text = str(img.shape[0])
        ET.SubElement(size, "height").text = str(img.shape[1])
        ET.SubElement(size, "depth").text = str(img.shape[2])

        for i in circles[0, :]:
            object = ET.SubElement(annotation, "object")
            ET.SubElement(object, "name").text = "pvc_pipe"
            ET.SubElement(object, "pose").text = "Unspecified"
            ET.SubElement(object, "truncated").text = "0"
            ET.SubElement(object, "difficult").text = "0"
            bndbox = ET.SubElement(object, "bndbox")
            ET.SubElement(bndbox, "xmin").text = str(i[0] - i[2] - config["threshold"])
            ET.SubElement(bndbox, "ymin").text = str(i[1] - i[2] - config["threshold"])
            ET.SubElement(bndbox, "xmax").text = str(i[0] + i[2] + config["threshold"])
            ET.SubElement(bndbox, "ymax").text = str(i[1] + i[2] + config["threshold"])

        ET.SubElement(annotation, "segmented").text = 0

        settings = ET.SubElement(annotation, "config")
        ET.SubElement(settings, "img_default_width").text = str(config["img_default_width"])
        ET.SubElement(settings, "img_default_height").text = str(config["img_default_height"])
        ET.SubElement(settings, "minDist").text = str(config["minDist"])
        ET.SubElement(settings, "minRadius").text = str(config["minRadius"])
        ET.SubElement(settings, "maxRadius").text = str(config["maxRadius"])
        ET.SubElement(settings, "gaussian_default").text = str(config["gaussian_default"])
        ET.SubElement(settings, "median_default").text = str(config["median_default"])
        ET.SubElement(settings, "threshold").text = str(config["threshold"])

        tree = ET.ElementTree(annotation)
        tree.write(directory + "/" + os.path.splitext(file)[0] + ".xml")


def is_valid(ext):
    return ext.lower().endswith(('.png', '.jpg', '.jpeg'))


def nextImage():
    global file_index

    file_index += 1
    while file_index < len(files) and not is_valid(files[file_index]):
        file_index += 1


def load_next():
    nextImage()
    safe = file_index < len(files)
    try:
        if safe:
            load_image()
        else:
            show_info("All images are done!")
    except BaseException as e:
        print(e)
        traceback.print_exc()
        print("Trying to load next...")
        if safe:
            load_next()


def save_and_load_next():
    save()
    load_next()


def reset():
    init()
    sld_gaussian.set(config["gaussian_default"])
    sld_median.set(config["median_default"])
    sld_minsize.set(config["minRadius"])
    sld_maxsize.set(config["maxRadius"])
    sld_mindist.set(config["minDist"])
    sld_threshold.set(config["threshold"])


init()
window = tk.Tk()
window.title("cvLabels")

window.rowconfigure(3, minsize=800, weight=1)
window.columnconfigure(0, minsize=800, weight=1)

info_box = tk.Label(window)
img_box = tk.Label(window)
fr_buttons = tk.Frame(window)
fr2_buttons = tk.Frame(window)

lbl_directory = tk.Label(fr_buttons, text="Directory: Not set.")
lbl_gaussian = tk.Label(fr_buttons, text="Gaussian: ")
sld_gaussian = tk.Scale(fr_buttons, from_=0, to=20, orient=tk.HORIZONTAL, command=scales_onchange)
sld_gaussian.set(config["gaussian_default"])
lbl_median = tk.Label(fr_buttons, text="Median: ")
sld_median = tk.Scale(fr_buttons, from_=0, to=20, orient=tk.HORIZONTAL, command=scales_onchange)
sld_median.set(config["median_default"])
btn_reset = tk.Button(fr_buttons, text="Reset", command=reset)
btn_next = tk.Button(fr_buttons, text="Next", command=load_next)
btn_open = tk.Button(fr_buttons, text="Save & Next", command=save_and_load_next)

lbl_minsize = tk.Label(fr2_buttons, text="Min size: ")
sld_minsize = tk.Scale(fr2_buttons, from_=0, to=100, orient=tk.HORIZONTAL, command=scales_onchange)
sld_minsize.set(config["minRadius"])
lbl_maxsize = tk.Label(fr2_buttons, text="Max size: ")
sld_maxsize = tk.Scale(fr2_buttons, from_=0, to=100, orient=tk.HORIZONTAL, command=scales_onchange)
sld_maxsize.set(config["maxRadius"])
lbl_mindist = tk.Label(fr2_buttons, text="Min dist.: ")
sld_mindist = tk.Scale(fr2_buttons, from_=0, to=400, orient=tk.HORIZONTAL, command=scales_onchange)
sld_mindist.set(config["minDist"])
lbl_threshold = tk.Label(fr2_buttons, text="Threshold.: ")
sld_threshold = tk.Scale(fr2_buttons, from_=-30, to=30, orient=tk.HORIZONTAL, command=scales_onchange)
sld_threshold.set(config["threshold"])

lbl_minsize.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
sld_minsize.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
lbl_maxsize.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
sld_maxsize.grid(row=0, column=3, sticky="ew", padx=5, pady=5)
lbl_mindist.grid(row=0, column=4, sticky="ew", padx=5, pady=5)
sld_mindist.grid(row=0, column=5, sticky="ew", padx=5, pady=5)
lbl_threshold.grid(row=0, column=6, sticky="ew", padx=5, pady=5)
sld_threshold.grid(row=0, column=7, sticky="ew", padx=5, pady=5)

lbl_directory.grid(row=0, column=0, sticky="ew", padx=10, pady=5)
lbl_gaussian.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
sld_gaussian.grid(row=0, column=2, sticky="ew", padx=5, pady=5)
lbl_median.grid(row=0, column=3, sticky="ew", padx=5, pady=5)
sld_median.grid(row=0, column=4, sticky="ew", padx=5, pady=5)
btn_reset.grid(row=0, column=5, sticky="ew", padx=5, pady=5)
btn_next.grid(row=0, column=6, sticky="ew", padx=5, pady=5)
btn_open.grid(row=0, column=7, sticky="ew", padx=5, pady=5)

fr_buttons.grid(row=0, column=0, sticky="ns")
fr2_buttons.grid(row=1, column=0, sticky="ns")
info_box.grid(row=2, column=0, sticky="nsew")
img_box.grid(row=3, column=0, sticky="nsew")

data_path = askdirectory()
lbl_directory.config(text="Directory: " + data_path)
files = os.listdir(data_path)

nextImage()
load_image()

window.mainloop()

# https://www.programmersought.com/article/78194233146/
# https://stackoverflow.com/questions/43841210/how-to-detect-circle-in-a-binary-image/43844556
