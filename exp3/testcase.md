from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np

faces = fetch_olivetti_faces()
X, y = faces.images, faces.target
X = X.reshape(-1, 64, 64, 1).astype('float32')
y_cat = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y_cat, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dense(40, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), verbose=2)

y_pred_probs = model.predict(X_test)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

num_samples = 10
print(f"{'Input Face Image':<17} {'Expected Identity':<18} {'Predicted Identity':<19} {'Correct (Y/N)'}")
for i in range(num_samples):
    expected = y_true_classes[i]
    predicted = y_pred_classes[i]
    correct = 'Y' if expected == predicted else 'N'
    print(f"Image {i+1:<13} Person {chr(65 + expected):<16} Person {chr(65 + predicted):<17} {correct}")


<img width="728" height="552" alt="Screenshot 2025-09-03 091650" src="https://github.com/user-attachments/assets/25673c51-83a1-49b2-986c-cf654e245fb1" />
