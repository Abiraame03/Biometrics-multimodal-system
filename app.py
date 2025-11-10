# =========================================================
# 📸 Multimodal Biometrics App — Streamlit version
# Face + Iris + Gesture + Voice feedback
# =========================================================
import streamlit as st
import cv2, mediapipe as mp, numpy as np, tempfile, os
import tensorflow as tf
from gtts import gTTS
from io import BytesIO

# -------------------------------
# Config
# -------------------------------
st.set_page_config(page_title="Multimodal Auth", page_icon="🤖", layout="wide")

gesture_voice = {
    "FIVE": "Hello! How are you?",
    "PEACE": "Thank you!",
    "THUMBS_UP": "Yes, I understand.",
    "THUMBS_DOWN": "No, please repeat.",
    "FIST": "Goodbye!",
    "CALL_ME": "Call someone for help!",
    "ROCK": "Let’s go!",
    "OK": "Everything is fine."
}

mp_face = mp.solutions.face_detection.FaceDetection(0.5)
mp_hands = mp.solutions.hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# -------------------------------
# Load TFLite model
# -------------------------------
@st.cache_resource
def load_tflite_model(path="gesture_model.tflite"):
    interpreter = tf.lite.Interpreter(model_path=path)
    interpreter.allocate_tensors()
    return interpreter

def predict_gesture(interpreter, img):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    idx = np.argmax(preds)
    conf = float(np.max(preds))
    classes = list(gesture_voice.keys())
    label = classes[idx] if idx < len(classes) else "Unknown"
    return label, conf

# -------------------------------
# Voice output
# -------------------------------
def play_voice(text):
    tts = gTTS(text)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        tts.save(fp.name)
        audio_file = open(fp.name, "rb")
        audio_bytes = audio_file.read()
        st.audio(audio_bytes, format="audio/mp3")
        audio_file.close()
        os.remove(fp.name)

# -------------------------------
# Streamlit camera UI
# -------------------------------
st.title("🤖 Multimodal Authentication and Voice Feedback")
st.write("Face + Iris + Gesture + Voice")

gesture_model_path = "gesture_model.tflite"
if not os.path.exists(gesture_model_path):
    st.warning("Upload your gesture_model.tflite file below 👇")
    uploaded = st.file_uploader("Upload model", type=["tflite"])
    if uploaded:
        with open(gesture_model_path, "wb") as f:
            f.write(uploaded.getbuffer())
        st.success("Model uploaded successfully! ✅")

interpreter = None
if os.path.exists(gesture_model_path):
    interpreter = load_tflite_model(gesture_model_path)

frame_window = st.image([])
cam = st.camera_input("Show your gesture 👇")

if cam is not None and interpreter:
    # Read image
    file_bytes = np.asarray(bytearray(cam.read()), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face detection
    face_result = mp_face.process(frame_rgb)
    if face_result.detections:
        for det in face_result.detections:
            bboxC = det.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Hand (gesture)
    hands_result = mp_hands.process(frame_rgb)
    if hands_result.multi_hand_landmarks:
        for handLms in hands_result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp.solutions.hands.HAND_CONNECTIONS)

        label, conf = predict_gesture(interpreter, frame)
        st.write(f"### ✋ Gesture: {label} ({conf:.2f})")
        if label in gesture_voice:
            st.info(f"🗣 Voice: {gesture_voice[label]}")
            play_voice(gesture_voice[label])

    frame_window.image(frame, channels="BGR")

st.caption("© 2025 Multimodal Auth System")
