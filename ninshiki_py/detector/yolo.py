# Copyright (c) 2021 Ichiro ITS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import cv2
import numpy as np
import os
from rclpy.node import MsgType
from ninshiki_interfaces.msg import DetectedObject


class Yolo:
    def __init__(self, gpu: bool = False, myriad: bool = False):
        self.file_name = os.path.expanduser('~') + "/yolo_model/obj.names"
        self.classes = None

        config = os.path.expanduser('~') + "/yolo_model/config.cfg"
        weights = os.path.expanduser('~') + "/yolo_model/yolo_weights.weights"
        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.outs = None

        # self.width = 0
        # self.height = 0

        self.gpu = gpu
        self.myriad = myriad

    def pass_image_to_network(self, image: np.ndarray):
        with open(self.file_name, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')

        self.net.getLayerNames()
        layerOutput = self.net.getUnconnectedOutLayersNames()

        if self.gpu:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        elif self.myriad:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_MYRIAD)
        else:
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        input_width = 416
        input_height = 416
        blob = cv2.dnn.blobFromImage(image, 1/255, (input_width, input_height),
                                     [0, 0, 0], 1, crop=False)

        self.net.setInput(blob)
        self.outs = self.net.forward(layerOutput)

    def detection(self, image: np.ndarray, detection_result: MsgType):
        # Get object name, score, and location
        confident_threshold = 0.4
        nms_threshold = 0.3

        class_id = np.argmax(self.outs[0][0][5:])
        frame_h, frame_w, frame_c = image.shape
        class_ids = []
        confidences = []
        boxes = []

        for out in self.outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                c_x = int(detection[0] * frame_w)
                c_y = int(detection[1] * frame_h)
                w = int(detection[2] * frame_w)
                h = int(detection[3] * frame_h)
                x = int(c_x - w / 2)
                y = int(c_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

        if len(boxes):
            indices = cv2.dnn.NMSBoxes(boxes, confidences, confident_threshold, nms_threshold)
            # Get object location
            for i in indices:
                if i < (len(boxes) - 1):
                    box = np.array(boxes)[i]
                    if len(box) == 4:
                        x = np.array(box)[0]
                        y = np.array(box)[1]
                        w = np.array(box)[2]
                        h = np.array(box)[3]

                        # Add detected object into list
                        detection_object = DetectedObject()

                        detection_object.label = self.classes[class_ids[i]]
                        detection_object.score = confidences[i]
                        detection_object.left = x / frame_w
                        detection_object.top = y / frame_h
                        detection_object.right = (x+w) / frame_w
                        detection_object.bottom = (y+h) / frame_h

                        detection_result.detected_objects.append(detection_object)

    def load_data():
        # Load data from settei
        pass
