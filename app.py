import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import pickle
import tempfile
import urllib.request
import os
from tensorflow.keras.models import load_model
from PIL import Image
import speech_recognition as sr
from gtts import gTTS

# =========================================================
# 🔹 CONFIG
# =========================================================
REPO_BASE = "https://raw.githubusercontent.com/Abiraame03/Biometrics-multimodal-system/main/gesture%20auth%20app%20models"

MODEL_PATH_H5 = f"{REPO_BASE}/gesture_model.h5"
LABEL_ENCODER_PATH = f"{REPO_BASE}/gesture_label_encoder.pkl"
VOICE_MAP_PATH = f"{REPO_BASE}/voice_map.pkl"

# =========================================================
# 🔹 LOAD MODELS
# =========================================================
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

@st.cache_resource
def load_voice_map():
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as tmp:
        urllib.request.urlretrieve(VOICE_MAP_PATH, tmp.name)
        with open(tmp.name, "rb") as f:
            voice_map = pickle.load(f)
    return voice_map

gesture_model = load_gesture_model()
gesture_labels = load_label_encoder()
voice_map = load_voice_map()

# Get gesture input shape dynamically
try:
    input_shape = gesture_model.input_shape[1:3]
except Exception:
    input_shape = (128, 128)

# =========================================================
# 🔹 FUNCTIONS
# =========================================================
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

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    if len(faces) > 0:
        st.success("✅ Face detected successfully")
    else:
        st.warning("⚠️ No face detected")

def detect_iris(image):
    st.info("👁️ Iris detection simulated (demo mode)")

def recognize_voice():
    st.info("🎙 Speak now for voice authentication...")
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        try:
            text = recognizer.recognize_google(audio)
            st.write(f"🗣 You said: **{text}**")
            # Match voice text to gesture map if possible
            matched = None
            for gesture, phrase in voice_map.items():
                if phrase.lower() in text.lower():
                    matched = gesture
                    break
            if matched:
                st.success(f"✅ Voice matched with gesture: {matched}")
                tts = gTTS(f"Access granted for {matched}")
                tts.save("voice_feedback.mp3")
                os.system("start voice_feedback.mp3" if os.name == "nt" else "afplay voice_feedback.mp3")
            else:
                st.error("❌ No match found for the spoken phrase.")
        except Exception as e:
            st.error(f"Speech recognition failed: {e}")

# =========================================================
# 🖥 STREAMLIT UI
# =========================================================
st.title("🧠 Multimodal Biometric Authentication System")
st.markdown("Authenticate using **Face**, **Iris**, **Gesture**, or **Voice**.")

mode = st.sidebar.selectbox(
    "Choose Authentication Mode",
    ["Face", "Iris", "Gesture", "Voice"]
)

if mode == "Face":
    st.header("📸 Face Recognition")
    img_file = st.camera_input("Capture your face")
    if img_file:
        img = np.array(Image.open(img_file))
        detect_face(img)

elif mode == "Iris":
    st.header("👁️ Iris Recognition")
    img_file = st.camera_input("Capture your eye region")
    if img_file:
        img = np.array(Image.open(img_file))
        detect_iris(img)

elif mode == "Gesture":
    st.header("✋ Gesture Recognition")
    cam = st.camera_input("Show your hand gesture")
    if cam:
        frame = np.array(Image.open(cam))
        label, conf = predict_gesture(frame)
        st.success(f"Gesture Detected: **{label}** (Confidence: {conf:.2f})")

elif mode == "Voice":
    st.header("🎤 Voice Authentication")
    if st.button("Start Voice Recognition"):
        recognize_voice()

st.caption("Developed with ❤️ using Streamlit, TensorFlow, OpenCV, and SpeechRecognition.")
