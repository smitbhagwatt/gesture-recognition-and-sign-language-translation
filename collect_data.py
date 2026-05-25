import cv2
import mediapipe as mp
import csv
import os

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

label = input("Enter label (e.g., HELLO): ")

file_exists = os.path.isfile("dataset.csv")

with open("dataset.csv", mode="a", newline="") as f:
    writer = csv.writer(f)

    if not file_exists:
        header = []
        for i in range(21):
            header += [f"x{i}", f"y{i}"]
        header.append("label")
        writer.writerow(header)

    print("Press 's' to save samples")

    while True:
        success, img = cap.read()
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(imgRGB)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                lm_list = []
                h, w, _ = img.shape

                for lm in handLms.landmark:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lm_list.append((cx, cy))

                if len(lm_list) == 21:
                    if cv2.waitKey(1) & 0xFF == ord('s'):
                        row = []
                        for x, y in lm_list:
                            row += [x, y]
                        row.append(label)
                        writer.writerow(row)
                        print("Saved")

        cv2.imshow("Collect Data", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()