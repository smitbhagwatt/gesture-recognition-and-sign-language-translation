import pyautogui
import time

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


class SystemControl:

    def __init__(self):
        self.dragging = False
        self.pinch_start_time = None

        self.prev_x, self.prev_y = 0, 0
        self.smoothening = 2

        self.selection_end_time = 0

    def move_mouse(self, x, y, frame_w, frame_h):
        screen_w, screen_h = pyautogui.size()

        margin = 80
        x = max(margin, min(frame_w - margin, x))
        y = max(margin, min(frame_h - margin, y))

        screen_x = (x - margin) * screen_w / (frame_w - 2 * margin)
        screen_y = (y - margin) * screen_h / (frame_h - 2 * margin)

        curr_x = self.prev_x + (screen_x - self.prev_x) / self.smoothening
        curr_y = self.prev_y + (screen_y - self.prev_y) / self.smoothening

        pyautogui.moveTo(curr_x, curr_y)

        self.prev_x, self.prev_y = curr_x, curr_y

    def handle_pinch(self, pinch):
        current_time = time.time()

        if pinch:
            if self.pinch_start_time is None:
                self.pinch_start_time = current_time

            elif (current_time - self.pinch_start_time > 0.4) and not self.dragging:
                pyautogui.mouseDown()
                self.dragging = True

        else:
            if self.pinch_start_time is not None:

                # 🔥 FIX: delay release to allow right-click
                if self.dragging:
                    if current_time - self.pinch_start_time > 0.7:
                        pyautogui.mouseUp()
                        self.dragging = False
                        self.selection_end_time = current_time
                else:
                    pyautogui.click()

                self.pinch_start_time = None

    def context_right_click(self):
        # 🔥 FIX: allow right click anytime (even during drag window)
        pyautogui.rightClick()