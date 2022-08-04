import requests
import os

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

