import cv2
import numpy as np
import streamlit as st
from tensorflow.lite.python.interpreter import Interpreter
import mediapipe as mp
from gtts import gTTS
import tempfile
import os
from playsound import playsound

# =====================================================
# 1️⃣ Streamlit App Title
# =====================================================
st.set_page_config(page_title="Multimodal Authentication", layout="centered")
st.title("🔒 Multimodal Authentication & Voice Feedback System")

# =====================================================
# 2️⃣ Load Gesture TFLite Model
# =====================================================
gesture_model_path = "gesture_model.tflite"  # your model path
interpreter = Interpreter(model_path=gesture_model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# =====================================================
# 3️⃣ Gesture Label Mapping (with Voice Command)
# =====================================================
gesture_to_voice = {
    "FIVE": "Hello! How are you?",
    "PEACE": "Thank you!",
    "THUMBS_UP": "Yes, I understand.",
    "THUMBS_DOWN": "No, please repeat.",
    "FIST": "Goodbye!",
    "CALL_ME": "Call someone for help!",
    "ROCK": "Let's go!",
    "OK": "Everything is fine."
}

gesture_labels = list(gesture_to_voice.keys())

# =====================================================
# 4️⃣ Initialize Mediapipe
# =====================================================
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# =====================================================
# 5️⃣ Utility Functions
# =====================================================
def predict_gesture(frame):
    img = cv2.resize(frame, (224, 224))  # must match training input size
    img = np.expand_dims(img / 255.0, axis=0).astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])
    label = gesture_labels[np.argmax(preds)]
    conf = np.max(preds)
    return label, conf

def speak(text):
    tts = gTTS(text=text, lang="en")
    with tempfile.NamedTemporaryFile(delete=True) as fp:
        temp_path = f"{fp.name}.mp3"
        tts.save(temp_path)
        playsound(temp_path)
        os.remove(temp_path)

# =====================================================
# 6️⃣ Camera Input
# =====================================================
run = st.checkbox("Start Camera")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    st.info("Camera started. Showing real-time detection...")
    with mp_hands.Hands(max_num_hands=1) as hands, mp_face.FaceDetection(min_detection_confidence=0.6) as face_detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.warning("Camera feed not available.")
                break

            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face Detection
            face_results = face_detector.process(rgb_frame)
            if face_results.detections:
                for det in face_results.detections:
                    mp_drawing.draw_detection(frame, det)

            # Gesture Detection
            hand_results = hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                for handLms in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
                    x_min = int(min([lm.x for lm in handLms.landmark]) * frame.shape[1])
                    y_min = int(min([lm.y for lm in handLms.landmark]) * frame.shape[0])
                    x_max = int(max([lm.x for lm in handLms.landmark]) * frame.shape[1])
                    y_max = int(max([lm.y for lm in handLms.landmark]) * frame.shape[0])

                    roi = frame[y_min:y_max, x_min:x_max]
                    if roi.size > 0:
                        label, conf = predict_gesture(roi)
                        cv2.putText(frame, f"{label} ({conf:.2f})", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                        # Voice feedback
                        speak(gesture_to_voice[label])

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
else:
    st.warning("Turn on the camera to start detection.")

st.success("✅ App ready! Use gestures to trigger voice feedback.")
