import cv2
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs
import numpy as np
import time
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, False)
#     except RuntimeError as e:
#         print(e)

class YOLO(object):
    def __init__(self):
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if len(physical_devices) > 0:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
        self.yolo = YoloV3(classes=80)
        self.yolo.load_weights('./model_data/yolov3.tf')
        self.class_names = [c.strip() for c in open('./model_data/coco_classes.txt').readlines()]

    def detect_image(self, image):
        #takes a cv2 image!
        img_in = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_in = tf.expand_dims(img_in, 0)
        img_in = transform_images(img_in, 416)
        boxes, objectness, classes, nums = self.yolo.predict_on_batch(img_in)
        boxes, objectness, classes, nums = boxes[0], objectness[0], classes[0], nums[0]
        return_boxs = []
        for i in range(nums):
            if self.class_names[int(classes[i])] != 'person':
                continue
            wh = np.flip(image.shape[0:2])
            x1y1 = tuple((np.array(boxes[i][0:2]) * wh).astype(np.int32))
            x2y2 = tuple((np.array(boxes[i][2:4]) * wh).astype(np.int32))
            x = x1y1[0]
            y = x1y1[1]
            w = x2y2[0] - x1y1[0]
            h = x2y2[1] - x1y1[1]
            if x < 0 :
                w = w + x
                x = 0
            if y < 0 :
                h = h + y
                y = 0
            return_boxs.append([x,y,w,h])

        return return_boxs