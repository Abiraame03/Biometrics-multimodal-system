import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import tempfile
from tensorflow.keras.models import load_model
from PIL import Image
import urllib.request

# -------------------------------
# 🔧 CONFIG
# -------------------------------
REPO_BASE = "https://raw.githubusercontent.com/Abiraame03/Biometrics-multimodal-system/main/gesture%20auth%20app%20models"

MODEL_PATH_H5 = f"{REPO_BASE}/gesture_model.h5"
LABEL_ENCODER_PATH = f"{REPO_BASE}/gesture_label_encoder.pkl"

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

# ✅ Get input shape dynamically
try:
    input_shape = gesture_model.input_shape[1:3]  # (H, W)
except Exception:
    input_shape = (128, 128)

st.sidebar.info(f"Gesture model input size: {input_shape}")

# -------------------------------
# ✋ GESTURE RECOGNITION
# -------------------------------
def predict_gesture(frame):
    img = cv2.resize(frame, input_shape)
    if gesture_model.input_shape[-1] == 1:  # grayscale model
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.expand_dims(img, -1)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    preds = gesture_model.predict(img)
    label = gesture_labels.inverse_transform([np.argmax(preds)])[0]
    conf = np.max(preds)
    return label, conf

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
    st.info("👁️ Iris detection simulated (placeholder)")

# -------------------------------
# 🖥️ STREAMLIT UI
# -------------------------------
st.title("🧠 Multimodal Biometric Authentication (Face + Iris + Gesture)")
st.markdown("Authenticate using **any user’s input** — face, iris, or gesture.")

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
    st.header("✋ Gesture Recognition")
    st.write("Show your gesture to the camera.")

    cam = st.camera_input("Capture Gesture")
    if cam:
        frame = np.array(Image.open(cam))
        label, conf = predict_gesture(frame)
        st.success(f"Gesture Detected: **{label}** (Confidence: {conf:.2f})")

st.caption("Built with Streamlit • TensorFlow • OpenCV • Keras")
