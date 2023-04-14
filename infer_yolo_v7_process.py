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
        self.input_size = 640
        self.use_custom_model = False
        self.pretrain_model = 'yolov7'
        self.cuda = torch.cuda.is_available()
        self.conf_thres = 0.25
        self.iou_conf = 0.5
        self.custom_model = ""
        self.model_name_or_path = ""
        self.update = False

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.input_size = int(param_map["input_size"])
        self.use_custom_model = utils.strtobool(param_map["use_custom_model"])
        self.pretrain_model = str(param_map["pretrain_model"])
        self.cuda = utils.strtobool(param_map["cuda"])
        self.conf_thres = float(param_map["conf_thres"])
        self.iou_conf = float(param_map["iou_conf"])
        self.custom_model = param_map["custom_model"]
        self.model_name_or_path = str(param_map["model_name_or_path"])
        self.update = True

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {}
        param_map["use_custom_model"] = str(self.use_custom_model)
        param_map["input_size"] = str(self.input_size)
        param_map['pretrain_model'] = str(self.pretrain_model)
        param_map["conf_thres"] = str(self.conf_thres)
        param_map["iou_conf"] = str(self.iou_conf)
        param_map["cuda"] = str(self.cuda)
        param_map["custom_model"] = str(self.custom_model)
        param_map["model_name_or_path"] = str(self.model_name_or_path)
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferYoloV7(dataprocess.CObjectDetectionTask):

    def __init__(self, name, param):
        dataprocess.CObjectDetectionTask.__init__(self, name)
        # Add object detection output

        self.model = None
        self.weights = ""
        self.device = torch.device("cpu")
        self.stride = 32
        self.imgsz = 640
        self.conf_thres = 0.25
        self.iou_conf = 0.45
        self.classes = None
        self.colors = None

        # Create parameters class
        if param is None:
            self.set_param_object(InferYoloV7Param())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def infer(self, img0):
        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]
        # Convert
        img = img.transpose(2, 0, 1)  # HxWxC, to CxHxW
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.device.type == 'cuda' else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_conf, classes=None, agnostic=False)[0]

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
            self.add_object(index, cls, float(conf), x, y, w, h)
            index += 1

    def run(self):
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get input :
        input = self.get_input(0)

        # Forward input image
        self.forward_input_image(0, 0)
        # Get parameters :
        param = self.get_param_object()

        if param.update or self.model is None:
            self.device = torch.device("cuda") if param.cuda else torch.device("cpu")
            self.iou_conf = param.iou_conf
            self.conf_thres = param.conf_thres
            print("Will run on {}".format(self.device.type))

            if param.model_name_or_path != "":
                if os.path.isfile(param.model_name_or_path):
                    param.use_custom_model = True
                    param.custom_model = param.model_name_or_path
                else:
                    param.pretrain_model = param.model_name_or_path

            if param.use_custom_model:
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
                    self.model = attempt_load(param.custom_model, map_location=self.device)  # load FP32 model
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
            self.set_names(self.classes)
            # remove added path in pythonpath after loading model
            self.stride = int(self.model.stride.max())  # model stride
            self.imgsz = check_img_size(param.input_size, s=self.stride)  # check img_size

            if self.device.type != 'cpu':
                self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(
                    next(self.model.parameters())))  # run once

            half = self.device.type != 'cpu'  # half precision only supported on CUDA
            if half:
                self.model.half()  # to FP16

            param.update = False

        # Get image from input/output (numpy array):
        srcImage = input.get_image()

        # Call to the process main routine
        with torch.no_grad():
            self.infer(srcImage)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferYoloV7Factory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_yolo_v7"
        self.info.short_description = "YOLOv7 object detection models."
        self.info.description = "This plugin proposes inference on YOLOv7 object detection models."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Detection"
        self.info.version = "1.2.0"
        self.info.icon_path = "icons/icon.png"
        self.info.authors = "Wang, Chien-Yao and Bochkovskiy, Alexey and Liao, Hong-Yuan Mark"
        self.info.article = "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors"
        self.info.journal = "arxiv"
        self.info.year = 2022
        self.info.license = "GPL-3.0"
        # URL of documentation
        self.info.documentation_link = ""
        # Code source repository
        self.info.repository = "https://github.com/WongKinYiu/yolov7"
        # Keywords used for search
        self.info.keywords = "yolo, v7, object, detection, real-time, coco"

    def create(self, param=None):
        # Create process object
        return InferYoloV7(self.info.name, param)
