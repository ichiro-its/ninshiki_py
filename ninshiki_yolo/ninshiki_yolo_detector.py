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
import rclpy
from rclpy.node import Node

from ninshiki_yolo.detector.detection import Detection
from ninshiki_yolo.node.ninshiki_yolo_node import NinshikiYoloNode


def main(args=None):
    rclpy.init(args=args)

    parser = argparse.ArgumentParser()
    parser.add_argument('topic', help='specify topic name to subscribe')
    parser.add_argument('--GPU', help='if we chose the computation using GPU',
                        type=int, choices=[0, 1], default=0)
    parser.add_argument('--MYRIAD', help='if we chose the computation using Compute Stick',
                        type=int, choices=[0, 1], default=0)
    arg = parser.parse_args()

    node = Node("ninshiki_yolo")

    detection = Detection(gpu=arg.GPU, myriad=arg.MYRIAD)
    detector_node = NinshikiYoloNode(node, arg.topic, detection)

    detector_node.set_detector()

    rclpy.spin(detector_node.node)

    detector_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
