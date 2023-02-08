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
import copy
import os
from infer_yolo_v7.ikutils import download_model
import torch
from infer_yolo_v7.yolov7.models.experimental import attempt_load
from infer_yolo_v7.yolov7.utils.datasets import letterbox
from infer_yolo_v7.yolov7.utils.general import check_img_size, non_max_suppression, scale_coords, xyxy2xywh
from infer_yolo_v7.yolov7.models.yolo import Model
from infer_yolo_v7.yolov7.utils.torch_utils import torch_load
import numpy as np
import random


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferYoloV7Param(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.img_size = 640
        self.custom_train = False
        self.pretrain_model = 'yolov7'
        self.cuda = torch.cuda.is_available()
        self.fp16 = False
        self.thr_conf = 0.25
        self.iou_conf = 0.5
        self.custom_model = ""
        self.update = False

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.img_size = int(param_map["img_size"])
        self.custom_train = utils.strtobool(param_map["custom_train"])
        self.pretrain_model = str(param_map["pretrain_model"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.cuda = utils.strtobool(param_map["fp16"])
        self.thr_conf = float(param_map["thr_conf"])
        self.iou_conf = float(param_map["iou_conf"])
        self.custom_model = param_map["custom_model"]
        self.update = True

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        param_map["custom_train"] = str(self.custom_train)
        param_map["img_size"] = str(self.img_size)
        param_map['pretrain_model'] = str(self.pretrain_model)
        param_map["thr_conf"] = str(self.thr_conf)
        param_map["iou_conf"] = str(self.iou_conf)
        param_map["cuda"] = str(self.cuda)
        param_map["fp16"] = str(self.fp16)
        param_map["custom_model"] = str(self.custom_model)
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
        self.weights = ""
        self.device = torch.device("cpu")
        self.stride = 32
        self.imgsz = 640
        self.thr_conf = 0.25
        self.iou_conf = 0.45
        self.classes = None
        self.colors = None
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
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img.transpose(2, 0, 1)  # HxWxC, to CxHxW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        if param.fp16 is True:
            img = img.half() if self.device.type == 'cuda' else img.float()  # uint8 to fp16/32
        else:
            img = img.float()  # uint8 to fp16/32

        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, 
                                   self.thr_conf,
                                   self.iou_conf,
                                   classes=None,
                                   agnostic=False)[0]

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
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get input :
        input = self.getInput(0)

        # Get outputs :
        self.obj_detect_output = self.getOutput(1)
        self.obj_detect_output.init("YoloV7", 0)

        # Forward input image
        self.forwardInputImage(0, 0)
        # Get parameters :
        param = self.getParam()

        if param.update or self.model is None:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            self.iou_conf = param.iou_conf
            self.thr_conf = param.thr_conf
            print("Will run on {}".format(self.device.type))

            if param.custom_train:
                ckpt = torch_load(param.custom_model, device=self.device)
                # custom model trained with ikomia
                if "yaml" in ckpt:
                    cfg = ckpt["yaml"]
                    self.classes = ckpt["names"]
                    state_dict = ckpt["state_dict"]
                    self.model = Model(cfg=cfg, ch=3, nc=len(self.classes), anchors=None)
                    self.model.load_state_dict(state_dict)
                    self.model.float().fuse().eval().to(self.device)
                    del ckpt
                # other
                else:
                    del ckpt
                    self.model = attempt_load(param.custom_model,
                                              map_location=self.device)  # load FP32 model
                    self.classes = self.model.names
            else:
                weights_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
                if not os.path.isdir(weights_folder):
                    os.mkdir(weights_folder)

                self.weights = os.path.join(weights_folder, param.pretrain_model + '.pt')
                if not os.path.isfile(self.weights):
                    download_model(param.pretrain_model, weights_folder)

                self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
                self.classes = self.model.names
            random.seed(0)
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classes]
            # remove added path in pythonpath after loading model
            self.stride = int(self.model.stride.max())  # model stride
            self.imgsz = check_img_size(param.img_size, s=self.stride)  # check img_size

            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                    next(self.model.parameters())))  # run once

            if param.fp16 is True:
                half = self.device.type != 'cpu'  # half precision only supported on CUDA
                if half:
                    self.model.half()  # to FP16

            param.update = False

        # Get image from input/output (numpy array):
        srcImage = input.getImage()

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
        self.info.description = "This plugin proposes inference on YOLOv7 object detection models."
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
