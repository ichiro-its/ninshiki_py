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
from rclpy.node import Node
from shisen_interfaces.msg import Image


class Detector (Node):
    def __init__(self, node_name, topic_name, config, names, weights):
        self.config = config
        self.names = names
        self.weights = weights
        super().__init__(node_name)

        self.image_subscription = self.create_subscription(
            Image,
            topic_name,
            self.listener_callback,
            10)

        self.get_logger().info("subscribe image on " + self.image_subscription.topic_name)

    def listener_callback(self, message):
        received_frame = np.array(message.data)
        received_frame = np.frombuffer(received_frame, dtype=np.uint8)

        # Raw Image
        if (message.quality < 0):
            received_frame = received_frame.reshape(message.rows, message.cols, 3)
            print("Raw Image")
        # Compressed Image
        else:
            received_frame = cv2.imdecode(received_frame, cv2.IMREAD_UNCHANGED)
            print("Compressed Image")

        if (received_frame.size != 0):
            output_img = self.detection(received_frame)
            cv2.imshow(self.image_subscription.topic_name, output_img)
            cv2.waitKey(1)
            self.get_logger().debug("once, received image and display it")
        else:
            self.get_logger().warn("once, received empty image")

    def detection(self, image):
        class_file = self.names
        classes = None
        with open(class_file, 'rt') as f:
            classes = f.read().rstrip('\n').split('\n')
        model_configuration = self.config
        model_weights = self.weights

        net = cv2.dnn.readNetFromDarknet(model_configuration, model_weights)
        net.getLayerNames()
        layerOutput = net.getUnconnectedOutLayersNames()

        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        inpWidth = 416
        inpHeight = 416
        blob = cv2.dnn.blobFromImage(image, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

        net.setInput(blob)
        outs = net.forward(layerOutput)
        image = self.postprocess(outs, image, classes)
        return image

    def draw_detection_result(self, img, label, x0, y0, xt, yt,
                              color=(255, 127, 0), text_color=(255, 255, 255)):

        y0, yt = max(y0 - 15, 0), min(yt + 15, img.shape[0])
        x0, xt = max(x0 - 15, 0), min(xt + 15, img.shape[1])

        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
        cv2.rectangle(img, (x0, y0 + baseline), (max(xt, x0 + w), yt), color, 2)
        cv2.rectangle(img, (x0, y0 - h), (x0 + w, y0 + baseline), color, -1)
        cv2.putText(img, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    text_color, 1, cv2.LINE_AA)

        return img

    def postprocess(self, outs, frame, classes, confThreshold=0.4, nmsThreshold=0.3):
        classId = np.argmax(outs[0][0][5:])

        frame_h, frame_w, frame_c = frame.shape
        classIds = []
        confidences = []
        boxes = []
        for out in outs:
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
        for i in indices:
            i = i[0]
            box = boxes[i]
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]

            label = '%s: %.1f%%' % (classes[classIds[i]], (confidences[i]*100))

            frame = self.draw_detection_result(frame, label, x, y, x+w, y+h, color=(255, 127, 0),
                                               text_color=(255, 255, 255))

        return frame


def main(args=None):
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--topic', help='specify topic name to subscribe')
        parser.add_argument('--config', help='specify model configuration')
        parser.add_argument('--names', help='specify class file name')
        parser.add_argument('--weights', help='specify model weights')
        arg = parser.parse_args()

        rclpy.init(args=args)
        detector = Detector("detector", arg.topic, arg.config, arg.names, arg.weights)

        rclpy.spin(detector)

        detector.destroy_node()
        rclpy.shutdown()
    except (IndexError):
        detector.get_logger("Usage: ros2 run ninshiki_yolo detector --names")


if __name__ == '__main__':
    main()
