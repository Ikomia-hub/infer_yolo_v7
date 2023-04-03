# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import utils, core, dataprocess
from infer_yolo_v7.ikutils import letterbox_costum
from infer_yolo_v7.yolov7.utils.general import non_max_suppression, scale_coords
from infer_yolo_v7.yolov7.utils.datasets import letterbox
import copy
import os
import torch
import numpy as np
import random
import cv2
import onnxruntime as ort
import yaml
import onnx
import ast

# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferYoloV7Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.weights = ""
        self.imgsz = 640
        self.thr_conf = 0.25
        self.iou_conf = 0.5
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.weights = str(param_map["weights"])
        self.imgsz = int(param_map["imgsz"])
        self.thr_conf = float(param_map["thr_conf"])
        self.iou_conf = float(param_map["iou_conf"])
        self.update = True

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["weights"] = str(self.weights)
        param_map["imgsz"] = str(self.imgsz)
        param_map["thr_conf"] = str(self.thr_conf)
        param_map["iou_conf"] = str(self.iou_conf)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYoloV7(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add object detection output
        self.addOutput(dataprocess.CObjectDetectionIO())

        self.obj_detect_output = None
        self.model = None
        self.device = torch.device("cpu")
        self.stride = 32
        self.classes = None
        self.colors = None
        self.session = None
        self.providers = ['CPUExecutionProvider']
        self.coco_data =  os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                        "yolov7", "data", "coco.yaml")
        # Create parameters class
        if param is None:
            self.setParam(InferYoloV7Param())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1


    def infer(self, img0):
        # Get parameters :
        param = self.getParam()

        # Padded resize
        img = cv2.cvtColor(img0, cv2.COLOR_BGR2RGB)
        h, w = np.shape(img)[:2]
        img, ratio, dwdh = letterbox(img, int(param.imgsz), stride=self.stride)

        # Convert
        img = img.transpose(2, 0, 1)  # HxWxC, to CxHxW
        img = np.expand_dims(img, 0)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0  # 0 - 255 to 0.0 - 1.0

        output_names = [x.name for x in self.session.get_outputs()]

        y = self.session.run(output_names, {self.session.get_inputs()[0].name: img})

        pred, *others, proto = [torch.tensor(i, device="cpu") for i in y] # return to torch
        y = (pred, (others, proto))

        pred = non_max_suppression(pred,
                                param.thr_conf,
                                param.iou_conf,
                                classes=None,
                                agnostic=False
                                )[0]

        index = 0
        pred[:, :4] = scale_coords(img.shape[2:], pred, img0.shape)[:, :4]

        for p in pred:
            box_score_cls = [e for e in p.detach().cpu().numpy()]
            box = box_score_cls[:4]
            box = [int(b) for b in box]
            cls = int(box_score_cls[5])
            conf = box_score_cls[4]
            w = float(box[2] - box[0])
            h = float(box[3] - box[1])
            x = float(box[0])
            y = float(box[1])
            self.obj_detect_output.addObject(index,
                                             self.classes[cls],
                                             float(conf),
                                             x, y, w, h,
                                             self.colors[cls])
            index += 1

    def run(self):
        # Call beginTaskRun for initialization
        self.beginTaskRun()
        # Get input :
        input = self.getInput(0)
        # Get image from input/output (numpy array):
        srcImage = input.getImage()
        # Get outputs :
        self.obj_detect_output = self.getOutput(1)
        self.obj_detect_output.init("YoloV7", 0)
        # Forward input image
        self.forwardInputImage(0, 0)
        # Get parameters :
        param = self.getParam()


        with open(self.coco_data) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
            names = data["names"]

        if param.update or self.session is None:
            self.device = torch.device("cpu")
            print("Will run on {}".format(self.device.type))
            self.session = ort.InferenceSession(param.weights, providers=self.providers)
            onnx_model = onnx.load(param.weights)
            
            if len(onnx_model.metadata_props) > 0:
                names = ast.literal_eval(onnx_model.metadata_props[0].value)
            else:
                with open(self.coco_data) as f:
                    data = yaml.load(f, Loader=yaml.FullLoader)
                    names = data["names"]
                    
            self.classes = names
            random.seed(0)
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classes]
            # remove added path in pythonpath after loading model
            param.update = False

        # Call to the process main routine
        with torch.no_grad():
            self.infer(srcImage)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYoloV7Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolo_v7"
        self.info.shortDescription = "YOLOv7 object detection models."
        self.info.description = "This plugin proposes inference on YOLOv7 object detection models."\
                                "Models should be in .onnx format."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.1.0"
        self.info.iconPath = "icons/icon.png"
        self.info.authors = "Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark"
        self.info.article = "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"
        self.info.journal = "arxiv"
        self.info.year = 2022
        self.info.license = "GPL-3.0"
        # URL of documentation
        self.info.documentationLink = ""
        # Code source repository
        self.info.repository = "https://github.com/WongKinYiu/yolov7"
        # Keywords used for search
        self.info.keywords = "yolo, v7, object, detection, real-time, coco"

    def create(self, param=None):
        # Create process object
        return InferYoloV7(self.info.name, param)
