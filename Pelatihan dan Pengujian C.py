import os
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

DATA_PATH = os.path.join(os.path.expanduser("~"), "Documents", "data.pickle")
MODEL_PATH = os.path.join(os.path.expanduser("~"), "Documents", "model.pickle")

if not os.path.exists(DATA_PATH):
    print("File data.pickle tidak ditemukan!")
    print("Pastikan file ada di Documents")
    exit()

with open(DATA_PATH, "rb") as f:
    data_dict = pickle.load(f)

data = np.array(data_dict['data'])
labels = np.array(data_dict['labels'])

print("Jumlah data:", len(data))

if len(data) < 5:
    print("Data terlalu sedikit untuk training!")
    exit()

x_train, x_test, y_train, y_test = train_test_split(
    data,
    labels,
    test_size=0.2,
    shuffle=True,
    stratify=labels
)

model = RandomForestClassifier(n_estimators=100)
model.fit(x_train, y_train)

y_predict = model.predict(x_test)
score = accuracy_score(y_test, y_predict)

print(f'Akurasi model: {score * 100:.2f}%')

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("Model berhasil disimpan di:")
print(MODEL_PATH)