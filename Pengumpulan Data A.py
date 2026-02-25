import os
import cv2

DATA_DIR = os.path.join(os.path.expanduser("~"), "Documents", "dataset_kamera")

os.makedirs(DATA_DIR, exist_ok=True)

number_of_classes = 2
dataset_size = 100

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera tidak bisa dibuka")
    exit()

for j in range(number_of_classes):

    class_path = os.path.join(DATA_DIR, str(j))
    os.makedirs(class_path, exist_ok=True)

    print(f'Collecting data for class {j}')
    print("Tekan tombol Q untuk mulai mengambil gambar...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kamera tidak terbaca")
            break

        cv2.putText(frame, 'Ready? Press "Q" !',
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret:
            break

        cv2.imshow('Frame', frame)
        cv2.waitKey(1)

        file_path = os.path.join(class_path, f'{counter}.jpg')
        cv2.imwrite(file_path, frame)

        print(f"Saved {file_path}")

        counter += 1

print("Pengambilan data selesai!")
cap.release()
cv2.destroyAllWindows()