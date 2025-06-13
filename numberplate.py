from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import cv2
import numpy as np
import os
import xml.etree.ElementTree as ET

def parse_annotation(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find("size")
    img_w = int(size.find("width").text)
    img_h = int(size.find("height").text)
    obj = root.find("object")
    bbox = obj.find("bndbox")
    xmin = int(bbox.find("xmin").text) / img_w
    ymin = int(bbox.find("ymin").text) / img_h
    xmax = int(bbox.find("xmax").text) / img_w
    ymax = int(bbox.find("ymax").text) / img_h
    return [xmin, ymin, xmax, ymax]

def load_data_from_nested_folders(root_dir, img_size=(224, 224)):
    X, y = [], []
    for state_folder in os.listdir(root_dir):
        state_path = os.path.join(root_dir, state_folder)
        if not os.path.isdir(state_path): continue
        for file in os.listdir(state_path):
            if not file.endswith(".jpg"): continue
            img_path = os.path.join(state_path, file)
            xml_path = os.path.join(state_path, file.replace(".jpg", ".xml"))
            if not os.path.exists(xml_path):
                continue
            img = cv2.imread(img_path)
            img = cv2.resize(img, img_size)
            img = img / 255.0
            X.append(img)
            bbox = parse_annotation(xml_path)
            y.append(bbox)
    return np.array(X), np.array(y)

def build_model(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(64, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Conv2D(128, (3,3), activation='relu'),
        layers.MaxPooling2D(2,2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(4, activation='sigmoid')
    ])
    return model

X, y = load_data_from_nested_folders("State-wise_OLX")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = build_model()
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(X_train, y_train, epochs=20, validation_data=(X_val, y_val), batch_size=16)

def show_prediction(model, img):
    pred = model.predict(img[np.newaxis])[0]
    h, w = img.shape[:2]
    x1 = int(pred[0] * w)
    y1 = int(pred[1] * h)
    x2 = int(pred[2] * w)
    y2 = int(pred[3] * h)
    img_disp = img.copy()
    cv2.rectangle(img_disp, (x1,y1), (x2,y2), (0,255,0), 2)
    plt.imshow(img_disp)
    plt.axis('off')
    plt.show()

show_prediction(model, X_val[0])
