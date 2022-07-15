# !/usr/bin/pyvenv3.7
import onnx
import cv2
import torch

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.export.caffe2_export import export_onnx_model
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

im = cv2.imread("input.jpg")
import numpy as np
from PIL import Image
import detectron2


cfg = get_cfg()
cfg.MODEL.DEVICE = 'cpu'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
model = build_model(cfg)




aug = T.ResizeShortestEdge([cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST],
                           cfg.INPUT.MAX_SIZE_TEST)
height, width = im.shape[:2]
image = aug.get_transform(im).apply_image(im)
image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
inputs2 = {"image":image}
print(image.shape)



output = torch.onnx.export(model, image.reshape(-1), "/model/detectron2.onnx")
