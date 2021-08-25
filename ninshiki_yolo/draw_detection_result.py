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
from rclpy.node import MsgType


def draw_detection_result(frame_width: int, frame_height: int, frame: np.array,
                          detection_result: MsgType) -> np.ndarray:
    for detected_object in detection_result.detected_objects:
        label = '%s: %.1f%%' % (detected_object.label, detected_object.score)
        x0 = int(detected_object.left * frame_width)
        y0 = int(detected_object.top * frame_height)
        xt = int(detected_object.right * frame_width)
        yt = int(detected_object.bottom * frame_height)
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
