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

import argparse
import cv2
import numpy as np
import rclpy
from rclpy.node import MsgType
from rclpy.node import Node
from ninshiki_interfaces.msg import DetectedObjects
from shisen_interfaces.msg import Image
from ninshiki_yolo.utils.draw_detection_result import draw_detection_result


class Viewer(Node):
    def __init__(self, node_name: str, detection_topic: str, img_topic: str):
        super().__init__(node_name)
        self.detection_result = DetectedObjects()
        self.width = 0
        self.height = 0

        self.detected_object_subscription = self.create_subscription(
            DetectedObjects, detection_topic, self.listener_callback_msg, 10)
        self.get_logger().info("subscribe detection result on "
                               + self.detected_object_subscription.topic_name)

        self.image_subscription = self.create_subscription(
            Image, img_topic, self.listener_callback_img, 10)
        self.get_logger().info("subscribe image on " + self.image_subscription.topic_name)

    def listener_callback_msg(self, message: MsgType):
        self.detection_result = message

    def listener_callback_img(self, message: MsgType):
        self.width = message.cols
        self.height = message.rows

        received_frame = np.array(message.data)
        received_frame = np.frombuffer(received_frame, dtype=np.uint8)

        # Raw Image
        if (message.quality < 0):
            received_frame = received_frame.reshape(message.rows, message.cols, 3)
        # Compressed Image
        else:
            received_frame = cv2.imdecode(received_frame, cv2.IMREAD_UNCHANGED)

        if (received_frame.size != 0):
            detection_frame = draw_detection_result(self.width, self.height,
                                                    received_frame, self.detection_result)
            cv2.imshow(self.image_subscription.topic_name, detection_frame)
            cv2.waitKey(1)
            self.get_logger().debug("once, received image and display it")
        else:
            self.get_logger().warn("once, received empty image")


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('detection_topic', help='specify topic name that contain'
                        'detection result to subscribe')
    parser.add_argument('img_topic', help='specify topic name that contain'
                        'image to subscribe')
    arg = parser.parse_args()

    rclpy.init(args=args)
    viewer = Viewer("viewer", arg.detection_topic, arg.img_topic)

    rclpy.spin(viewer)

    viewer.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
