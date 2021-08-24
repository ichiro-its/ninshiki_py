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


class Viewer(Node):
    def __init__(self, node_name: str, detection_topic: str, img_topic: str):
        super().__init__(node_name)
        self.detection_result = DetectedObjects()

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
        received_frame = np.array(message.data)
        received_frame = np.frombuffer(received_frame, dtype=np.uint8)

        # Raw Image
        if (message.quality < 0):
            received_frame = received_frame.reshape(message.rows, message.cols, 3)
        # Compressed Image
        else:
            received_frame = cv2.imdecode(received_frame, cv2.IMREAD_UNCHANGED)

        if (received_frame.size != 0):
            detection_frame = self.draw_detection_result(received_frame, self.detection_result)
            cv2.imshow(self.image_subscription.topic_name, detection_frame)
            cv2.waitKey(1)
            self.get_logger().debug("once, received image and display it")
        else:
            self.get_logger().warn("once, received empty image")

    def draw_detection_result(self, frame: np.array, detection_result: MsgType) -> np.ndarray:
        for detected_object in detection_result.detected_objects:
            label = '%s: %.1f%%' % (detected_object.label, detected_object.score)
            x0 = detected_object.left
            y0 = detected_object.top
            xt = detected_object.right
            yt = detected_object.bottom
            color = (255, 127, 0)
            text_color = (255, 255, 255)

            y0, yt = max(y0 - 15, 0), min(yt + 15, frame.shape[0])
            x0, xt = max(x0 - 15, 0), min(xt + 15, frame.shape[1])

            # Draw detection result
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(frame, (x0, y0 + baseline), (max(xt, x0 + w), yt), color, 2)
            cv2.rectangle(frame, (x0, y0 - h), (x0 + w, y0 + baseline), color, -1)
            cv2.putText(frame, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        text_color, 1, cv2.LINE_AA)

        return frame


def main(args=None):
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--detection_topic', help='specify topic name that contain'
                            'detection result to subscribe')
        parser.add_argument('--img_topic', help='specify topic name that contain'
                            'image to subscribe')
        arg = parser.parse_args()

        rclpy.init(args=args)
        viewer = Viewer("viewer", arg.detection_topic, arg.img_topic)

        rclpy.spin(viewer)

        viewer.destroy_node()
        rclpy.shutdown()
    except (IndexError):
        viewer.get_logger("Usage: ros2 run ninshiki_yolo viewer"
                          "--detection_topic TOPIC --img_topic TOPIC")


if __name__ == '__main__':
    main()
