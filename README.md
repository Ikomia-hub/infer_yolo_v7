# Infer yolo_v7 

This plugin provides inference of yolo_v7 object detection models in .onnx format using cpu. 

The official [yolo_v7 repository](https://github.com/WongKinYiu/yolov7) provides an onnx export script, however the exported model won't be compatible with this model as the inference output will be different. 

To export your .pht model to .onnx format:
- clone this repository 

``` 
git clone -b onnx https://github.com/Ikomia-hub/infer_yolo_v7.git 
```

``` 
cd infer_yolo_v7 
```

- Export 

``` 
import torch
from models.experimental import attempt_load
from utils.general import check_img_size
import onnx

# Load model 
weights = "Path/to/your/pht/model"
device = "cpu"
model = attempt_load(weights, device)
model.eval()

# Input
img_size = [640,640]
img_size *= 2 if len(img_size) == 1 else 1  # expand
gs = int(max(model.stride))  # grid size (max stride)
img_size = [check_img_size(x, gs) for x in img_size]  # verify img_size are gs-multiples
batch_size = 1
img = torch.zeros(batch_size, 3, *img_size).to(device)  # torch.Size([1, 3, 640, 640])
dynamic_axes={
    'images': {
        0: 'batch',
        2: 'height',
        3: 'width'},  # shape(1,3,640,640)
    'output': {
        0: 'batch',
        1: 'anchors'}  # shape(1,25200,85)
}

# Export
torch.onnx.export(model,                   # model being run
                img,                       # model input (or a tuple for multiple inputs)
                "yolov7-dynamics.onnx",    # where to save the model (can be a file or file-like object)
                export_params=True,        # store the trained parameter weights inside the model file
                opset_version=12,          # the ONNX version to export the model to
                do_constant_folding=True,  # whether to execute constant folding for optimization
                verbose=False,             # prints a description of the model being exported to stdout
                input_names = ['images'],  # the model's input names
                output_names = ['output'], # the model's output names
                dynamic_axes = dynamic_axes,  # variable length axes
                )


# Checks
onnx_model = onnx.load("yolov7-dynamics.onnx")  # load onnx model
onnx.checker.check_model(onnx_model)  # check onnx model
```