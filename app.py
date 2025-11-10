import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import tempfile
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import urllib.request
import os

# -------------------------------
# 🔧 CONFIG
# -------------------------------
REPO_BASE = "https://raw.githubusercontent.com/Abiraame03/Biometrics-multimodal-system/main/gesture%20auth%20app%20models"

MODEL_PATH_H5 = f"{REPO_BASE}/gesture_model.h5"
MODEL_PATH_TFLITE = f"{REPO_BASE}/gesture_model.tflite"
LABEL_ENCODER_PATH = f"{REPO_BASE}/gesture_label_encoder.pkl"
FACE_EMB_PATH = f"{REPO_BASE}/face_embeddings.pkl"
IRIS_EMB_PATH = f"{REPO_BASE}/iris_embeddings.pkl"

# -------------------------------
# 🧩 LOAD MODELS
# -------------------------------
@st.cache_resource
def load_gesture_model():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp:
        urllib.request.urlretrieve(MODEL_PATH_H5, tmp.name)
        model = load_model(tmp.name)
    return model

@st.cache_resource
def load_label_encoder():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        urllib.request.urlretrieve(LABEL_ENCODER_PATH, tmp.name)
        with open(tmp.name, "rb") as f:
            le = pickle.load(f)
    return le

gesture_model = load_gesture_model()
gesture_labels = load_label_encoder()

# -------------------------------
# 🧍 FACE + IRIS PLACEHOLDERS
# -------------------------------
def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        st.success("✅ Face detected")
    else:
        st.warning("⚠️ No face detected")

def detect_iris(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.info("👁️ Iris detection simulated (placeholder for actual model)")

# -------------------------------
# ✋ GESTURE RECOGNITION
# -------------------------------
def predict_gesture(frame):
    img = cv2.resize(frame, (128, 128)) / 255.0
    img = np.expand_dims(img, axis=0)
    preds = gesture_model.predict(img)
    pred_label = gesture_labels.inverse_transform([np.argmax(preds)])[0]
    conf = np.max(preds)
    return pred_label, conf

# -------------------------------
# 🖥️ STREAMLIT UI
# -------------------------------
st.title("🧠 Multimodal Biometric Auth System (Face + Iris + Gesture)")
st.markdown("This app demonstrates multimodal authentication for **any user** using webcam input.")

mode = st.sidebar.selectbox("Choose Mode", ["Face", "Iris", "Gesture"])

if mode == "Face":
    st.header("📸 Face Detection")
    img_file = st.camera_input("Capture your face")
    if img_file:
        img = np.array(Image.open(img_file))
        detect_face(img)

elif mode == "Iris":
    st.header("👁️ Iris Detection")
    img_file = st.camera_input("Capture your eye region")
    if img_file:
        img = np.array(Image.open(img_file))
        detect_iris(img)

elif mode == "Gesture":
    st.header("✋ Real-time Gesture Recognition")
    st.write("Show a hand gesture to the camera...")

    cam = st.camera_input("Capture Gesture Frame")
    if cam:
        frame = np.array(Image.open(cam))
        label, conf = predict_gesture(frame)
        st.success(f"Gesture: **{label}** (Confidence: {conf:.2f})")

st.caption("Built using Streamlit • TensorFlow • OpenCV • Keras")
