import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

def preprocess_image(image_path):
    # Load image
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Histogram Equalization
    equalized = cv2.equalizeHist(gray)
    
    # Noise Reduction
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
    
    # Edge Detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Adaptive Thresholding
    adaptive_thresh = cv2.adaptiveThreshold(equalized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
    
    return adaptive_thresh

def extract_features(image_path):
    processed_img = preprocess_image(image_path)
    resized_img = cv2.resize(processed_img, (64, 64))
    return resized_img

def load_dataset(dataset_path):
    images = []
    labels = []
    
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        if os.path.isdir(label_path):
            for image_name in os.listdir(label_path):
                image_path = os.path.join(label_path, image_name)
                features = extract_features(image_path)
                images.append(features)
                labels.append(label)
    
    return np.array(images).reshape(-1, 64, 64, 1), np.array(labels)

def train_model(dataset_path):
    X, y = load_dataset(dataset_path)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Normalize data
    X_train, X_test = X_train / 255.0, X_test / 255.0
    
    # Build CNN Model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(len(set(y_encoded)), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    return model, label_encoder

def extract_sift_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return keypoints, descriptors

def match_nose_prints(img1_path, img2_path):
    keypoints1, descriptors1 = extract_sift_features(img1_path)
    keypoints2, descriptors2 = extract_sift_features(img2_path)
    
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    return len(good_matches)

# Example Usage
dataset_path = 'dog_nose_dataset'  # Replace with actual dataset path
model, encoder = train_model(dataset_path)
model.save("dog_nose_recognition_model.h5")

# Matching Example
img1 = 'dog_nose_1.jpg'  # Replace with actual path
img2 = 'dog_nose_2.jpg'  # Replace with actual path
match_score = match_nose_prints(img1, img2)
print(f'Match Score: {match_score}')
