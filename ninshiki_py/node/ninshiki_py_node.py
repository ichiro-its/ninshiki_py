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
import rclpy
from rclpy.node import MsgType

from ninshiki_interfaces.msg import DetectedObjects
from shisen_interfaces.msg import Image
from ninshiki_py.detector.tflite import TfLite
from ninshiki_py.detector.yolo import Yolo


class NinshikiPyNode:
    def __init__(self, node: rclpy.node.Node, topic_name: str):
        self.node = node
        self.topic_name = topic_name

        self.detection_result = DetectedObjects()
        self.received_frame = None

        self.detection = None

        self.image_subscription = self.node.create_subscription(
            Image, self.topic_name, self.listener_callback, 10)
        self.node.get_logger().info(
            "subscribe image on "
            + self.image_subscription.topic_name)

        self.detected_object_publisher = self.node.create_publisher(
            DetectedObjects, self.node.get_name() + "/detection", 10)
        self.node.get_logger().info(
            "publish detected images on "
            + self.detected_object_publisher.topic_name)

        # create timer
        timer_period = 0.008  # seconds
        self.node.timer = self.node.create_timer(timer_period, self.publish)

    def listener_callback(self, message: MsgType):
        if (message.data != []):
            self.received_frame = np.array(message.data)
            self.received_frame = np.frombuffer(self.received_frame, dtype=np.uint8)

            # Raw Image
            if (message.quality < 0):
                self.received_frame = self.received_frame.reshape(message.rows, message.cols, 3)
            # Compressed Image
            else:
                self.received_frame = cv2.imdecode(self.received_frame, cv2.IMREAD_UNCHANGED)

    def publish(self):
        if (self.received_frame is not None):
            if (self.received_frame.size != 0):
                if isinstance(self.detection, Yolo):
                    self.detection.pass_image_to_network(self.received_frame)
                    self.detection.detection(self.received_frame, self.detection_result, 0.4, 0.3)
                    # print("detector: ", self.detection_result)
                elif isinstance(self.detection, TfLite):
                    self.detection.detection(self.received_frame, self.detection_result, 0.4)

                self.detected_object_publisher.publish(self.detection_result)
        else:
            # self.get_logger().warn("once, received empty image")
            pass

        # Clear message list
        self.detection_result.detected_objects.clear()

    def set_detection(self, detection: Yolo or TfLite):
        self.detection = detection
