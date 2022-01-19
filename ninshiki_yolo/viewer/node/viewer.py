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
from ninshiki_interfaces.msg import DetectedObjects
from ninshiki_yolo.utils.draw_detection_result import draw_detection_result


class Viewer:
    def __init__(self):
        self.detection_result = DetectedObjects()
        self.width = 0
        self.height = 0

    def show_detection_result(self, image: np.ndarray):
        if (image is not None):
            if (image.size != 0):
                # if postprocess:

                detection_frame = draw_detection_result(self.width, self.height,
                                                        image, self.detection_result)
                cv2.imshow("viewer", detection_frame)
                cv2.waitKey(1)
                # self.get_logger().debug("once, received image and display it")
        else:
            # self.get_logger().warn("once, received empty image")
            pass

    def set_width(self, width):
        self.width = width

    def set_height(self, height):
        self.height = height

    def set_detection_result(self, detection_result):
        self.detection_result = detection_result
