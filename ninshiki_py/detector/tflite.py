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
from tflite_runtime.interpreter import Interpreter
from ninshiki_interfaces.msg import DetectedObject


class TfLite:
    def __init__(self):
        self.file_label = os.path.expanduser('~') + "/tflite/labelmap.txt"
        self.labels = None
        self.model = os.path.expanduser('~') + "/tflite/detect.tflite"

        # variable declaration
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_width = 0
        self.model_height = 0
        self.floating_model = None
        self.input_mean = 0
        self.input_std = 0

    def initiate_tflite(self):
        # Load the label map
        with open(self.file_label, 'r') as f:
            self.labels = [line.strip() for line in f.readlines()]

        self.interpreter = Interpreter(model_path=self.model)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.model_height = self.input_details[0]['shape'][1]
        self.model_width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.input_mean = 127.5
        self.input_std = 127.5
    
    def detection(self, image: np.ndarray, detection_result: MsgType):
        min_conf_threshold = 0.5

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_height, img_width, _ = image.shape 
        image_resized = cv2.resize(image_rgb, (self.model_width, self.model_height))
        input_data = np.expand_dims(image_resized, axis=0)

        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'],input_data)
        self.interpreter.invoke()

        # Get detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0] # Class index of detected objects
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0] # Confidence of detected objects

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                y_min = int(max(1, (boxes[i][0] * img_height)))
                x_min = int(max(1, (boxes[i][1] * img_width)))
                y_max = int(min(img_height, (boxes[i][2] * img_height)))
                x_max = int(min(img_width, (boxes[i][3] * img_width)))
                
                detection_object = DetectedObject()

                detection_object.label = self.labels[int(classes[i])]
                detection_object.score = float(scores[i])
                detection_object.left = x_min / img_width
                detection_object.top = y_min / img_height
                detection_object.right = x_max / img_width
                detection_object.bottom = y_max / img_height

                detection_result.detected_objects.append(detection_object)
