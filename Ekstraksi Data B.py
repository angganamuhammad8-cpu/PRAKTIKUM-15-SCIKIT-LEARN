import os
import pickle
import mediapipe as mp
import cv2

DATA_DIR = os.path.join(os.path.expanduser("~"), "Documents", "dataset_kamera")

if not os.path.exists(DATA_DIR):
    print("Folder dataset_kamera tidak ditemukan di Documents!")
    exit()

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.6
)

data = []
labels = []

print("Mulai ekstraksi fitur...\n")

for label_name in sorted(os.listdir(DATA_DIR)):

    class_path = os.path.join(DATA_DIR, label_name)

    if not os.path.isdir(class_path):
        continue

    print(f"Memproses kelas: {label_name}")

    image_count = 0

    for img_name in os.listdir(class_path):

        if not img_name.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        img_path = os.path.join(class_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        data_aux = []
        x_ = []
        y_ = []

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:

            for hand_landmarks in results.multi_hand_landmarks:

                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)

                min_x = min(x_)
                min_y = min(y_)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

            data.append(data_aux)
            labels.append(int(label_name))
            image_count += 1

    print(f"   -> {image_count} gambar berhasil diproses\n")

hands.close()

print("Total data berhasil diekstrak:", len(data))

save_path = os.path.join(os.path.expanduser("~"), "Documents", "data.pickle")

with open(save_path, "wb") as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data berhasil disimpan di:")
print(save_path)