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
from ninshiki_interfaces.msg import YoloDetectedObject, YoloDetectedObjects


class Viewer(Node):
    def __init__(self, node_name: str, topic_name: str):
        super().__init__(node_name)

        self.image_subscription = self.create_subscription(
            YoloDetectedObjects, topic_name, self.listener_callback, 10)
        self.get_logger().info("subscribe image on " + self.image_subscription.topic_name)

    def listener_callback(self, message: MsgType):
        # print("message = ", message)
        # print(type(message))
        # print("=========================")
        for detected_object in message.detected_objects:
            print(detected_object)
            print("")



def main(args=None):
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--topic', help='specify topic name to subscribe')
        arg = parser.parse_args()

        rclpy.init(args=args)
        viewer = Viewer("viewer", arg.topic)

        rclpy.spin(viewer)

        viewer.destroy_node()
        rclpy.shutdown()
    except (IndexError):
        viewer.get_logger("Usage: ros2 run ninshiki_yolo viewer --topic TOPIC")


if __name__ == '__main__':
    main()