import cv2
import mediapipe as mp
import numpy as np
import pickle
import av
import threading
from streamlit_webrtc import VideoProcessorBase


class SignLanguageProcessor(VideoProcessorBase):
    """Processes webcam frames for sign language recognition via WebRTC."""

    def __init__(self):
        # MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_styles = mp.solutions.drawing_styles

        # Load KNN model
        with open("model.pkl", "rb") as f:
            self.model = pickle.load(f)

        # Thread-safe shared state
        self._lock = threading.Lock()
        self._prediction = ""
        self._landmarks_detected = False
        self._fingers = [0, 0, 0, 0, 0]

    # ── Public properties (read from Streamlit main thread) ──────────

    @property
    def prediction(self):
        with self._lock:
            return self._prediction

    @property
    def landmarks_detected(self):
        with self._lock:
            return self._landmarks_detected

    @property
    def fingers(self):
        with self._lock:
            return list(self._fingers)

    # ── Feature extraction (wrist-normalized) ────────────────────────

    @staticmethod
    def _extract_features(lm_list):
        features = []
        base_x, base_y = lm_list[0]
        for x, y in lm_list:
            features.append(x - base_x)
            features.append(y - base_y)
        return features

    @staticmethod
    def _get_fingers(lm_list):
        fingers = []
        # Thumb
        if lm_list[4][0] > lm_list[3][0]:
            fingers.append(1)
        else:
            fingers.append(0)
        # Other four fingers
        tips = [8, 12, 16, 20]
        for tip in tips:
            if lm_list[tip][1] < lm_list[tip - 2][1]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    # ── Frame processing (runs in WebRTC thread) ─────────────────────

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        prediction = ""
        detected = False
        fingers = [0, 0, 0, 0, 0]

        if results.multi_hand_landmarks:
            for hand_lms in results.multi_hand_landmarks:
                detected = True

                # Draw styled landmarks
                self.mp_draw.draw_landmarks(
                    img,
                    hand_lms,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_styles.get_default_hand_landmarks_style(),
                    self.mp_styles.get_default_hand_connections_style(),
                )

                # Build landmark list
                lm_list = []
                for lm in hand_lms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                if len(lm_list) == 21:
                    fingers = self._get_fingers(lm_list)
                    features = self._extract_features(lm_list)
                    try:
                        prediction = self.model.predict([features])[0]
                    except Exception:
                        prediction = ""

                # ── Draw overlay ─────────────────────────────────
                # Semi-transparent banner at top
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, 72), (15, 15, 20), -1)
                cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

                # Prediction text
                if prediction:
                    cv2.putText(
                        img,
                        f"Sign: {prediction}",
                        (16, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.3,
                        (102, 126, 234),
                        3,
                        cv2.LINE_AA,
                    )
                else:
                    cv2.putText(
                        img,
                        "Detecting...",
                        (16, 48),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (113, 128, 150),
                        2,
                        cv2.LINE_AA,
                    )

                # LIVE badge
                cv2.circle(img, (w - 30, 30), 8, (0, 0, 255), -1)
                cv2.putText(
                    img,
                    "LIVE",
                    (w - 80, 38),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )

        else:
            # No hand — show hint
            cv2.putText(
                img,
                "Show your hand to the camera",
                (w // 2 - 200, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (113, 128, 150),
                2,
                cv2.LINE_AA,
            )

        # Update shared state
        with self._lock:
            self._prediction = prediction
            self._landmarks_detected = detected
            self._fingers = fingers

        return av.VideoFrame.from_ndarray(img, format="bgr24")
