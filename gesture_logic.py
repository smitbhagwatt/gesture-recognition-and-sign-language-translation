import math

class GestureLogic:
    def __init__(self):
        self.gesture_buffer = []
        self.buffer_size = 5

    def get_fingers(self, lm_list):
        fingers = []

        if lm_list[4][1] > lm_list[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        tips = [8, 12, 16, 20]
        for tip in tips:
            if lm_list[tip][2] < lm_list[tip - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers

    def stabilize_gesture(self, fingers):
        self.gesture_buffer.append(tuple(fingers))

        if len(self.gesture_buffer) > self.buffer_size:
            self.gesture_buffer.pop(0)

        if self.gesture_buffer.count(self.gesture_buffer[0]) == self.buffer_size:
            return self.gesture_buffer[0]

        return None

    def detect_pinch(self, lm_list):
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]

        distance = math.hypot(x2 - x1, y2 - y1)
        return distance < 25