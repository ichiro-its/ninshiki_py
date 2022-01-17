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
from .viewer import Viewer


class ViewerNode:
    def __init__(self, node: rclpy.node.Node, img_topic: str, 
                 detection_topic: str, viewer: Viewer):
        # self.enable_view_detection_result = postprocess
        self.node = node
        self.received_frame = None

        self.viewer = viewer

        self.detected_object_subscription = self.node.create_subscription(
            DetectedObjects, detection_topic, self.listener_callback_msg, 10)
        self.node.get_logger().info("subscribe detection result on "
                               + self.detected_object_subscription.topic_name)

        self.image_subscription = self.node.create_subscription(
            Image, img_topic, self.listener_callback_img, 10)
        self.node.get_logger().info("subscribe image on " + self.image_subscription.topic_name)

    def listener_callback_msg(self, message: MsgType):
        self.viewer.set_detection_result(message)

    def listener_callback_img(self, message: MsgType):
        if (message.data != []):
            self.viewer.set_width(message.cols)
            self.viewer.set_height(message.rows)

            self.received_frame = np.array(message.data)
            self.received_frame = np.frombuffer(self.received_frame, dtype=np.uint8)

            # Raw Image
            if (message.quality < 0):
                self.received_frame = self.received_frame.reshape(message.rows, message.cols, 3)
            # Compressed Image
            else:
                self.received_frame = cv2.imdecode(self.received_frame, cv2.IMREAD_UNCHANGED)
            
            self.viewer.show_detection_result(self.received_frame)
