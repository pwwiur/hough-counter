# Hough circle counter
Circle counter application using computer vision hough circle transform algorithm. Another capabillity of this application is to save bounding boxes for object detection tasks for deep learning porpuses. It saves a `xml` file named the same as image name in opened directory for labels. The structure of the xml file inherited from [labelImg](https://github.com/qaprosoft/labelImg) project which is suitable for tensorflow. You can also open the same directory with labelImg to correct labels.

## Installation

First clone project to your directory, Then install following packages:

    pip install numpy opencv-python matplotlib pillow

After installation run it by this command:

    python main.py

## Demo

![hough-counter-demo](https://user-images.githubusercontent.com/22914652/133030972-395a9b71-da7b-45e2-93d4-db14d1708989.png)

## Config

You can change default config using `init` function in `main.py` file:

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
