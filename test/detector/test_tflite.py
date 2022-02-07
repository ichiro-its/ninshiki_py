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
import os
import unittest
from rclpy.node import MsgType
import wget

from ninshiki_interfaces.msg import DetectedObjects
from ninshiki_py.detector.tflite import TfLite


class Object:
    def __init__(self, label: str, score: float, left: float,
                 top: float, right: float, bottom: float):
        self.label = label
        self.score = score
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom


class TestDetection(unittest.TestCase):
    def download(self, url: str, directory: str):
        file_name = ""
        file_name_changed = ""
        dir_path = os.path.expanduser('~') + directory
        # check if directory not exist
        if not os.path.isdir(dir_path):
            os.makedirs(dir_path)
            wget.download(url, out=dir_path)

        # check all files in that directory
        for r, d, f in os.walk(dir_path):
            for file in f:
                file_name = file
                if file.split(".")[-1] == "tflite":
                    file_name_changed = "efficientdet_lite0.tflite"

                    old_file = os.path.join(dir_path, file_name)
                    new_file = os.path.join(dir_path, file_name_changed)
                    os.rename(old_file, new_file)

    def make_detected_objects(self) -> list:
        objects = []
        # need update when change the model
        objects.append(Object('cup', 0.66, 0.09, 0.41, 0.32, 0.78))
        objects.append(Object('dining table', 0.58, -0.01, 0.26, 0.97, 0.99))
        objects.append(Object('book', 0.43, 0.14, 0.26, 0.62, 0.45))
        objects.append(Object('knife', 0.45, 0.53, 0.52, 0.92, 0.87))

        return objects

    def delete_folder(self, directory: str):
        dir_path = os.path.expanduser('~') + directory
        for f in os.listdir(dir_path):
            os.remove(os.path.join(dir_path, f))
        os.rmdir(dir_path)

    def iou(self, detected_object: list, detection_result: MsgType) -> float:
        # Determine the the intersection rectangle
        x_min = max(detected_object.left, detection_result.left)
        y_min = max(detected_object.top, detection_result.top)
        x_max = min(detected_object.right, detection_result.right)
        y_max = min(detected_object.bottom, detection_result.bottom)

        # Compute the area of intersection rectangle
        inter_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)

        # Compute the area of the each input rectangle
        detected_object_width = detected_object.right - detected_object.left
        detected_object_height = detected_object.bottom - detected_object.top
        detected_object_area = detected_object_width * detected_object_height

        detection_result_width = detection_result.right - detection_result.left
        detection_result_height = detection_result.bottom - detection_result.top
        detection_result_area = detection_result_width * detection_result_height

        # Compute the intersection over union
        iou = inter_area / float(detected_object_area + detection_result_area - inter_area)

        return iou

    def test_detection(self):
        # download image, config, class name, and weights
        self.download("https://raw.githubusercontent.com/tensorflow/examples/master/"
                      "lite/examples/object_detection/raspberry_pi/test_data/table.jpg",
                      "/example_img_tflite/")
        self.download("https://tfhub.dev/tensorflow/lite-model/efficientdet/lite0/"
                      "detection/metadata/1?lite-format=tflite", "/tflite_model/")

        image_path = os.path.expanduser('~') + "/example_img_tflite/table.jpg"
        image = cv2.imread(image_path)
        detection_result = DetectedObjects()

        detected_objects = self.make_detected_objects()

        detection = TfLite()
        detection.detection(image, detection_result, 0.4)

        for i in range(len(detection_result.detected_objects)):
            # check detected object name is the same
            self.assertEqual(detection_result.detected_objects[i].label, detected_objects[i].label)

            # Calculate the Intersection over Union ratio
            score = self.iou(detected_objects[i], detection_result.detected_objects[i])
            self.assertGreaterEqual(score, 0.8)

        self.delete_folder("/example_img_tflite/")
        self.delete_folder("/tflite_model/")
