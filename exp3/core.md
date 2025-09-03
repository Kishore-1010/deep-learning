from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.utils import to_categorical
import numpy as np
# Load data
faces = fetch_olivetti_faces()
X, y = faces.images, faces.target
X = X.reshape(-1, 64, 64, 1)
X = X.astype('float32')
y = to_categorical(y)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# CNN Model
model = Sequential([
 Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
 MaxPooling2D((2,2)),
 Flatten(),
 Dense(256, activation='relu'),
 Dense(40, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

<img width="1019" height="430" alt="Screenshot 2025-09-03 091632" src="https://github.com/user-attachments/assets/3e81111d-a36d-41fb-b666-97fd1d233e00" />
