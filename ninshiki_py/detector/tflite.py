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
import json
import numpy as np
import os
from rclpy.node import MsgType
from tflite_runtime.interpreter import Interpreter
from tflite_support import metadata
from ninshiki_interfaces.msg import DetectedObject


class TfLite:
    def __init__(self):
        self.labels = None
        self.model = os.path.expanduser('~') + "/tflite_model/efficientdet_lite0.tflite"

        # variable declaration
        self.interpreter = None
        self.input_details = None
        self.output_details = None
        self.model_width = 0
        self.model_height = 0
        self.floating_model = None
        self.input_mean = 0
        self.input_std = 0

        self.initiate_tflite()

    def initiate_tflite(self):
        self.interpreter = Interpreter(model_path=self.model)
        self.interpreter.allocate_tensors()

        # Get model details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.model_height = self.input_details[0]['shape'][1]
        self.model_width = self.input_details[0]['shape'][2]

        self.floating_model = (self.input_details[0]['dtype'] == np.float32)

        self.load_metadata()

    def load_metadata(self):
        # Load metadata from model.
        displayer = metadata.MetadataDisplayer.with_model_file(self.model)

        # Save model metadata for preprocessing later.
        model_metadata = json.loads(displayer.get_metadata_json())
        process_units = model_metadata['subgraph_metadata'][0][
            'input_tensor_metadata'][0]['process_units']
        mean = 127.5
        std = 127.5
        for option in process_units:
            if option['options_type'] == 'NormalizationOptions':
                mean = option['options']['mean'][0]
                std = option['options']['std'][0]
        self.input_mean = mean
        self.input_std = std

        # Load label list from metadata.
        file_name = displayer.get_packed_associated_file_list()[0]
        label_map_file = displayer.get_associated_file_buffer(file_name).decode()
        label_list = list(filter(len, label_map_file.splitlines()))
        self.labels = label_list

    def detection(self, image: np.ndarray, detection_result: MsgType, conf_threshold: float):
        min_conf_threshold = conf_threshold
        img_height, img_width, _ = image.shape

        # Preprocess
        image_resized = cv2.resize(image, (self.model_width, self.model_height))
        input_data = np.expand_dims(image_resized, axis=0)
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.floating_model:
            input_data = (np.float32(input_data) - self.input_mean) / self.input_std

        # Perform the actual detection by running the model with the image as input
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get detection results
        boxes = self.interpreter.get_tensor(self.output_details[0]['index'])[0]
        classes = self.interpreter.get_tensor(self.output_details[1]['index'])[0]
        scores = self.interpreter.get_tensor(self.output_details[2]['index'])[0]

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions,
                # need to force them to be within image using max() and min()
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
