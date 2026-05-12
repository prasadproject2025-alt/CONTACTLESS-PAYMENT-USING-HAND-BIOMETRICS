import cv2
import numpy as np
import mediapipe as mp
import os
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands

# Dataset Path
dataset_path = "C:/Users/SUBBU/Desktop/projects/biometric/dataset"

# Prepare Data Storage
labels = []
features = []

SEQUENCE_LENGTH = 10  # Number of frames per sequence

# Load Dataset and Extract Hand Landmarks
with mp_hands.Hands(static_image_mode=True, max_num_hands=1) as hands:
    for person in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person)
        person_features = []
        
        for img_file in sorted(os.listdir(person_path))[:SEQUENCE_LENGTH]:  # Limit sequence length
            image_path = os.path.join(person_path, img_file)
            image = cv2.imread(image_path)
            if image is None:
                continue

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    feature_vector = []
                    for landmark in hand_landmarks.landmark:
                        feature_vector.extend([landmark.x, landmark.y])
                    person_features.append(feature_vector)

        # Ensure all sequences have the same length
        if len(person_features) == SEQUENCE_LENGTH:
            features.append(person_features)
            labels.append(person)

# Convert to NumPy arrays
features = np.array(features)
labels = np.array(labels)

# Encode labels to integers
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)

# Save label encoder for later use
joblib.dump(label_encoder, "label_encoder.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Define CNN+LSTM Model
def create_cnn_lstm_model():
    model = Sequential([
        Conv1D(64, kernel_size=3, activation="relu", input_shape=(SEQUENCE_LENGTH, 42)),  # 21 landmarks * (x, y)
        MaxPooling1D(pool_size=2),
        Dropout(0.3),
        LSTM(128, return_sequences=True),
        LSTM(64, return_sequences=False),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(len(set(labels)), activation="softmax")  # Multi-class classification
    ])
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Create and Train Model
model = create_cnn_lstm_model()
model.fit(X_train, y_train, epochs=20, batch_size=8, validation_data=(X_test, y_test))

# Save Model
model.save("hand_biometric_cnn_lstm.h5")

print("✅ CNN+LSTM Model Trained and Saved Successfully!")
