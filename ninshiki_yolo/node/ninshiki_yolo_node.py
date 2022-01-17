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

import rclpy
from rclpy.node import MsgType
from rclpy.node import Node

from ninshiki_yolo.detector.node.detection import Detection
from ninshiki_yolo.detector.node.detector_node import DetectorNode
from ninshiki_yolo.viewer.node.viewer import Viewer
from ninshiki_yolo.viewer.node.viewer_node import ViewerNode

class NinshikiYoloNode:
    def __init__(self, node: rclpy.node.Node, topic_name: str, postprocess: bool):
        self.node = node
        self.postprocess = postprocess
        self.topic_name = topic_name

        self.detector_node = None
        self.viewer_node = None

        self.node.timer = None
    
    def set_detector(self, detection: Detection):
        self.detector_node = DetectorNode(self.node, self.topic_name, detection)
        timer_period = 0.008  # seconds
        self.node.timer = self.node.create_timer(timer_period, self.publish)
    
    def set_viewer(self, viewer: Viewer):
        if self.postprocess:
            self.viewer_node = ViewerNode(self.node, self.topic_name,
                                        self.detector_node.detected_object_publisher.topic_name,
                                        viewer, self.postprocess)

    def publish(self):
        self.detector_node.publish()
