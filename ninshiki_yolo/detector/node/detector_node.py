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
from ninshiki_interfaces.msg import DetectedObject, DetectedObjects
from shisen_interfaces.msg import Image
from ninshiki_yolo.utils.draw_detection_result import draw_detection_result
from .detection import Detection

class DetectorNode:
    def __init__(self, node: rclpy.node.Node, topic_name: str,
                 config: str, names: str, weights: str, 
                 gpu: bool, myriad: bool):
        self.detection_result = DetectedObjects()
        self.received_frame = None
        # self.enable_view_detection_result = postprocess

        self.detection = Detection(config, names, weights, self.detection_result, 
                                   gpu, myriad)
        
        self.image_subscription = node.create_subscription(
            Image, topic_name, self.listener_callback, 10)
        node.get_logger().info(
            "subscribe image on "
            + self.image_subscription.topic_name)

        self.detected_object_publisher = node.create_publisher(
            DetectedObjects, node.get_name() + "/detection", 10)
        node.get_logger().info(
            "publish detected images on "
            + self.detected_object_publisher.topic_name)
    
    def listener_callback(self, message: MsgType):
        if (message.data != []):
            self.detection.width = message.cols
            self.detection.height = message.rows

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
                self.detection.pass_image_to_network(self.received_frame)
                self.detection.detection(self.received_frame, self.detection_result)
                # print("detector: ", self.detection_result)
                self.detected_object_publisher.publish(self.detection_result)

                # if self.enable_view_detection_result:
                #     postprocess_frame = draw_detection_result(self.detection.width, self.detection.height,
                #                                             self.received_frame, self.detection_result)
                #     cv2.imshow(self.image_subscription.topic_name, postprocess_frame)
                #     cv2.waitKey(1)
                #     # self.get_logger().debug("once, received image and display it")
                # else:
                #     # self.get_logger().debug("once, received image but not display it")
                #     pass
        else:
            # self.get_logger().warn("once, received empty image")
            pass

        # Clear message list
        self.detection_result.detected_objects.clear()
