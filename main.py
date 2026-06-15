import cv2
import mediapipe as mp
import time
import pickle
import numpy as np
import threading
import pyttsx3
from gesture_logic import GestureLogic
from system_control import SystemControl

# Load modeldiction = model.predict([latest_features])[0]
with open("model.pkl", "rb") as f:
    model = pickle.load(f)
# Audio
engine = pyttsx3.init()

# ML thread globals
prediction = ""
latest_features = None
run_ml = True

def ml_worker():
    global prediction, latest_features
    while run_ml:
        if latest_features is not None:
            try:
                prediction = model.predict([latest_features])[0]
            except:
                pass
        time.sleep(0.12)  # lag fix

threading.Thread(target=ml_worker, daemon=True).start()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

pTime = 0
gesture = GestureLogic()
control = SystemControl()

# Sentence
sentence = []
last_added_time = 0

# Timers
speak_start = 0
clear_start = 0
speak_done = False
clear_done = False

def extract_features(lm_list):
    features = []
    base_x, base_y = lm_list[0][1], lm_list[0][2]

    for _, x, y in lm_list:
        features.append(x - base_x)
        features.append(y - base_y)

    return features

def speak_async(text):
    def _run():
        engine.say(text)
        engine.runAndWait()
    threading.Thread(target=_run, daemon=True).start()

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = hands.process(imgRGB)

    if results.multi_hand_landmarks:
        for handLms, handType in zip(results.multi_hand_landmarks, results.multi_handedness):

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

            lm_list = []
            h, w, c = img.shape

            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

            fingers = gesture.get_fingers(lm_list)
            stable_gesture = gesture.stabilize_gesture(fingers)
            pinch = gesture.detect_pinch(lm_list)

            current_time = time.time()

            cv2.putText(img, f'Fingers: {fingers}', (20, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if pinch:
                cv2.putText(img, 'PINCH', (20, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Control (unchanged)
            control.handle_pinch(pinch)

            if fingers[1] == 1 and fingers[2] == 0:
                x, y = lm_list[8][1], lm_list[8][2]
                control.move_mouse(x, y, w, h)

            if stable_gesture and stable_gesture[1] == 1 and stable_gesture[2] == 1:
                control.context_right_click()

            # ML input
            if not pinch:
                latest_features = extract_features(lm_list)

            # Show sign
            cv2.putText(img, f'Sign: {prediction}', (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Sentence build (idle only)
            if prediction != "":
                if fingers[1] == 0 and not pinch and not (fingers[1] == 1 and fingers[2] == 1):
                    if current_time - last_added_time > 2:
                        sentence.append(prediction)
                        last_added_time = current_time

            # Speak (unchanged logic)
            if fingers == [0,1,1,1,1]:
                if speak_start == 0:
                    speak_start = current_time
                    speak_done = False

                if current_time - speak_start > 1 and not speak_done:
                    if len(sentence) > 0:
                        speak_async(' '.join(sentence))
                        speak_done = True
            else:
                speak_start = 0

            # 🧹 CLEAR (FIXED — TRUE 2 SEC HOLD)
            if fingers[1] == 1 and fingers[2] == 1 and fingers[3] == 1:

                if clear_start == 0:
                    clear_start = current_time
                    clear_done = False

                hold_time = current_time - clear_start

                if hold_time >= 2 and not clear_done:
                    sentence = []
                    clear_done = True

                    cv2.putText(img, "CLEARED", (20, 260),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                clear_start = 0
                clear_done = False

    # Show sentence
    cv2.putText(img, ' '.join(sentence), (20, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    # FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Gesture + Sign System", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

run_ml = False
cap.release()
cv2.destroyAllWindows()