import requests
import os
import cv2
import numpy as np

model_zoo = {'yolov7': "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt",
             'yolov7x': "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7x.pt",
             'yolov7-w6': "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6.pt",
             'yolov7-e6': "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6.pt",
             'yolov7-d6': "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-d6.pt",
             'yolov7-e6e': "https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-e6e.pt"}


def download_model(name, models_folder):
    URL = model_zoo[name]
    print("Downloading model for {}".format(name))
    response = requests.get(URL)
    with open(os.path.join(models_folder, name + ".pt"), "wb") as f:
        f.write(response.content)


def letterbox_costum(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    
    return im, r, (dw, dh)