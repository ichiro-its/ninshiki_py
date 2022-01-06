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
from ninshiki_interfaces.msg import DetectedObject, DetectedObjects
from shisen_interfaces.msg import Image
from .draw_detection_result import draw_detection_result


class Detector(Node):
    def __init__(self, node_name: str, topic_name: str, config: str, names: str,
                 weights: str, postprocess: int):
        super().__init__(node_name)
        self.file_name = names
        self.classes = None

        self.net = cv2.dnn.readNetFromDarknet(config, weights)
        self.outs = None

        self.width = 0
        self.height = 0
        self.enable_view_detection_result = postprocess
        self.detection_result = DetectedObjects()

        self.image_subscription = self.create_subscription(
            Image, topic_name, self.listener_callback, 10)
        self.get_logger().info(
            "subscribe image on "
            + self.image_subscription.topic_name)

        self.detected_object_publisher = self.create_publisher(
            DetectedObjects, node_name + "/detections", 10)
        self.get_logger().info(
            "publish detected images on "
            + self.detected_object_publisher.topic_name)

    def listener_callback(self, message: MsgType):
        self.width = message.cols
        self.height = message.rows
        detection_result = DetectedObjects()

        received_frame = np.array(message.data)
        received_frame = np.frombuffer(received_frame, dtype=np.uint8)

        # Raw Image
        if (message.quality < 0):
            received_frame = received_frame.reshape(message.rows, message.cols, 3)
        # Compressed Image
        else:
            received_frame = cv2.imdecode(received_frame, cv2.IMREAD_UNCHANGED)

        if (received_frame.size != 0):
            self.detection(received_frame, detection_result)
            self.detected_object_publisher.publish(detection_result)

            if self.enable_view_detection_result:
                postprocess_frame = draw_detection_result(self.width, self.height,
                                                          received_frame, detection_result)
                cv2.imshow(self.image_subscription.topic_name, postprocess_frame)
                cv2.waitKey(1)
                self.get_logger().debug("once, received image and display it")
            else:
                self.get_logger().debug("once, received image but not display it")
        else:
            self.get_logger().warn("once, received empty image")

        # Clear message list
        self.detection_result.detected_objects.clear()
    
    def define_model(self, image: np.ndarray):
        with open(self.file_name, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        
        self.net.getLayerNames()
        layerOutput = self.net.getUnconnectedOutLayersNames()

        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        inpWidth = 416
        inpHeight = 416
        blob = cv2.dnn.blobFromImage(image, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        self.net.setInput(blob)
        self.outs = self.net.forward(layerOutput)

    def detection(self, image: np.ndarray, detection_result: MsgType):
        self.define_model(image)

        # Get object name, score, and location
        confThreshold: float = 0.4
        nmsThreshold: float = 0.3

        classId = np.argmax(self.outs[0][0][5:])
        frame_h, frame_w, frame_c = image.shape
        classIds = []
        confidences = []
        boxes = []

        for out in self.outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                c_x = int(detection[0] * frame_w)
                c_y = int(detection[1] * frame_h)
                w = int(detection[2] * frame_w)
                h = int(detection[3] * frame_h)
                x = int(c_x - w / 2)
                y = int(c_y - h / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
        # Get object location
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            self.add_detected_object(self.classes[classIds[i]], confidences[i],
                                     x / self.width, y / self.height,
                                     (x+w) / self.width, (y+h) / self.height,
                                     detection_result)

    def add_detected_object(self, label: str, score: float,
                            x0: float, y0: float, x1: float, y1: float,
                            detection_result: MsgType):
        detection_object = DetectedObject()

        detection_object.label = label
        detection_object.score = score
        detection_object.left = x0
        detection_object.top = y0
        detection_object.right = x1
        detection_object.bottom = y1

        detection_result.detected_objects.append(detection_object)


def main(args=None):
    help_message = """Usage: ros run ninshiki_yolo detector --topic TOPIC
       --config CONFIG --names NAMES --weights WEIGHT --postprocess POSTPROCESS

Value for optional argument:
- TOPIC              string
- CONFIG             string
- NAMES              string
- WEIGHT             string
- POSTPROCESS        1 or 0"""

    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--topic', help='specify topic name to subscribe')
        parser.add_argument('--config', help='specify model configuration')
        parser.add_argument('--names', help='specify class file name')
        parser.add_argument('--weights', help='specify model weights')
        parser.add_argument('--postprocess', help='show detection result')
        arg = parser.parse_args()

        rclpy.init(args=args)
        detector = Detector("detector", arg.topic, arg.config, arg.names, arg.weights,
                            int(arg.postprocess))

        rclpy.spin(detector)

        detector.destroy_node()
        rclpy.shutdown()
    except (TypeError):
        print("WARNING: Missing Argument !!!")
        print(help_message)
    # except:
    #     print("WARNING: Value Error !!!")
    #     print(help_message)


if __name__ == '__main__':
    main()
