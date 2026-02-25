import os
import pickle
import cv2
import mediapipe as mp
import numpy as np

MODEL_PATH = os.path.join(os.path.expanduser("~"), "Documents", "model.pickle")

if not os.path.exists(MODEL_PATH):
    print("model.pickle tidak ditemukan di Documents!")
    exit()

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_drawing = mp.solutions.drawing_utils

print("Tekan Q untuk keluar...")

while True:
    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()
    if not ret:
        break

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            for landmark in hand_landmarks.landmark:
                x_.append(landmark.x)
                y_.append(landmark.y)

            for landmark in hand_landmarks.landmark:
                data_aux.append(landmark.x - min(x_))
                data_aux.append(landmark.y - min(y_))

        prediction = model.predict([np.asarray(data_aux)])
        predicted_label = prediction[0]

        if predicted_label == "0":
            text = "KELAS 0"
            color = (0, 255, 0)
        else:
            text = "KELAS 1"
            color = (0, 0, 255)

        cv2.putText(
            frame,
            text,
            (50, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            color,
            3
        )

    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()