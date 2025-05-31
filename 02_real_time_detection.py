import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, VGG16, MobileNetV2
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.distance import cosine
import os

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features(image_path, model):
    img_array = preprocess_image(image_path)
    features = model.predict(img_array)
    return features.flatten()

def load_dataset(dataset_path, model):
    images = []
    labels = []
    
    for label in os.listdir(dataset_path):
        label_path = os.path.join(dataset_path, label)
        features = extract_features(label_path, model)
        images.append(features)
        labels.append(label)
        # if os.path.isdir(label_path):
        #     for image_name in os.listdir(label_path):
        #         image_path = os.path.join(label_path, image_name)
        #         features = extract_features(image_path, model)
        #         images.append(features)
        #         labels.append(label)
    
    return np.array(images), np.array(labels)

def train_model(dataset_path):
    base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    
    X, y = load_dataset(dataset_path, base_model)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Build Classifier Model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(len(set(y_encoded)), activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))
    return model, label_encoder

def match_nose_print(live_image_path, model, dataset_features, labels):
    live_features = extract_features(live_image_path, model)
    
    best_match = None
    best_score = float("inf")
    
    for idx, stored_features in enumerate(dataset_features):
        score = cosine(live_features, stored_features)
        if score < best_score:
            best_score = score
            best_match = labels[idx]
    
    return best_match if best_match else "No match found"

# Train and Save Model
dataset_path = '/media/brainwired/D/BW_ML/01_AUG_FARM_TEST/study/Eye_controlled_mouse/data'  # Replace with actual dataset path
# model, encoder = train_model(dataset_path)
# model.save("dog_nose_recognition_model.h5")

# Load the trained model for real-time matching
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
dataset_features, dataset_labels = load_dataset(dataset_path, base_model)

def real_time_recognition():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2.imshow('Live Nose Print Capture', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Press 's' to capture an image
            image_path = 'live_nose_print.jpg'
            cv2.imwrite(image_path, frame)
            match = match_nose_print(image_path, base_model, dataset_features, dataset_labels)
            print(f"Identified Dog: {match}")
        elif key == ord('q'):  # Press 'q' to exit
            break
    
    cap.release()
    cv2.destroyAllWindows()

# Run Real-Time Recognition
real_time_recognition()


# U-Net model combined with texture and spatial relationship feature extraction to
