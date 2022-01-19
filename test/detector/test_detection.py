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
from rclpy.node import MsgType
import shutil
import wget

from ninshiki_interfaces.msg import DetectedObjects
from ninshiki_yolo.detector.detection import Detection


class Object:
    def __init__(self, label: str, score: float, left: float,
                 top: float, right: float, bottom: float):
        self.label = label
        self.score = score
        self.left = left
        self.right = right
        self.top = top
        self.bottom = bottom


def download(url: str, directory: str):
    file_name_changed = ""
    dir_path = os.path.expanduser('~') + directory
    # check if directory not exist
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    file_name = url.split("/")[-1]
    file_extension = file_name.split(".")[-1]
    
    file_path = dir_path + file_name
    # check if file not exist
    if not os.path.isfile(file_path):
        wget.download(url, out=dir_path)

    # change file name
    if file_extension != "jpg":
        if file_extension == "cfg":
            file_name_changed = "config.cfg"
        elif file_extension == "weights":
            file_name_changed = "yolo_weights.weights"
        elif file_extension == "names":
            file_name_changed = "obj.names"
    
        old_file = os.path.join(dir_path, file_name)
        new_file = os.path.join(dir_path, file_name_changed)
        os.rename(old_file, new_file)
    

def make_detected_objects() -> list:
    objects = []
    # need update when change the model
    objects.append(Object('dog', 0.83, 0.16, 0.38, 0.5, 0.9))
    objects.append(Object('car', 0.73, 0.61, 0.14, 0.89, 0.3))

    return objects


def delete_folder(directory: str):
    dir_path = os.path.expanduser('~') + directory
    shutil.rmtree(dir_path)


def check_detected_objects(detected_objects: list, detection_result: MsgType):
    for i in range(len(detection_result.detected_objects)):
        assert detection_result.detected_objects[i].label == detected_objects[i].label
        # print(type(detected_objects[i].score))
        # print(round(detected_objects[i].score, 2))
        assert detection_result.detected_objects[i].score == round(detected_objects[i].score, 2)
        assert detection_result.detected_objects[i].left == round(detected_objects[i].left, 2)
        assert detection_result.detected_objects[i].right == round(detected_objects[i].right, 2)
        assert detection_result.detected_objects[i].top == round(detected_objects[i].top, 2)
        assert detection_result.detected_objects[i].bottom == round(detected_objects[i].bottom, 2)


def test_detection():
    # download image, config, class name, and weights
    download("https://raw.githubusercontent.com/pjreddie/darknet/master/data/dog.jpg",
             "/example_img/")
    download("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
             "/yolo_model/")
    download("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names",
             "/yolo_model/")
    download("https://pjreddie.com/media/files/yolov3-tiny.weights",
             "/yolo_model/")

    image_path = os.path.expanduser('~') + "/example_img/dog.jpg"
    image = cv2.imread(image_path)
    detection_result = DetectedObjects()

    detected_objects = make_detected_objects()

    detection = Detection()
    detection.set_width(image.shape[1])
    detection.set_height(image.shape[0])
    detection.pass_image_to_network(image)
    detection.detection(image, detection_result)
    assert detection.width == 768
    assert detection.height == 576
    assert detection.gpu is False
    assert detection.myriad is False
    check_detected_objects(detected_objects, detection_result)

    delete_folder("/example_img/")
    delete_folder("/yolo_model/")
