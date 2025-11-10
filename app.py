import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import tempfile
import pyttsx3
from PIL import Image

# =====================================
# INITIAL SETUP
# =====================================
st.set_page_config(page_title="Multimodal Biometrics", layout="wide")
st.title("🧠 Multimodal Biometric Recognition System")
st.markdown("### Face, Iris, and Gesture Detection with Voice Output")

mp_face = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

engine = pyttsx3.init()

# =====================================
# GESTURE TO VOICE MAPPING
# =====================================
gesture_voice = {
    "FIVE": "Hello! How are you?",
    "PEACE": "Thank you!",
    "THUMBS_UP": "Yes, I understand.",
    "THUMBS_DOWN": "No, please repeat.",
    "FIST": "Goodbye!",
    "CALL_ME": "Call someone for help!",
    "ROCK": "Let's go!",
    "OK": "Everything is fine."
}

# =====================================
# HELPER FUNCTIONS
# =====================================

def detect_face_and_iris(frame):
    with mp_face.FaceDetection(model_selection=1, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                mp_draw.draw_detection(frame, detection)
        return frame, bool(results.detections)


def detect_gesture(frame):
    gesture = None
    with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                # Simple geometric checks
                if landmarks[8][1] < landmarks[6][1] and landmarks[12][1] < landmarks[10][1]:
                    gesture = "PEACE"
                elif landmarks[4][1] < landmarks[3][1] and landmarks[8][1] > landmarks[6][1]:
                    gesture = "THUMBS_UP"
                elif landmarks[4][1] > landmarks[3][1] and landmarks[8][1] < landmarks[6][1]:
                    gesture = "THUMBS_DOWN"
                elif all(landmarks[i][1] > landmarks[0][1] for i in [8, 12, 16, 20]):
                    gesture = "FIST"
                elif all(landmarks[i][1] < landmarks[0][1] for i in [8, 12, 16, 20]):
                    gesture = "FIVE"

        return frame, gesture


def speak(text):
    engine.say(text)
    engine.runAndWait()

# =====================================
# MAIN LOGIC
# =====================================

mode = st.sidebar.selectbox("Select Mode", ["Live Webcam", "Upload Image"])

if mode == "Live Webcam":
    stframe = st.empty()
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to access camera.")
            break

        frame = cv2.flip(frame, 1)

        frame, face_detected = detect_face_and_iris(frame)
        frame, gesture = detect_gesture(frame)

        # Combine recognition info
        text = ""
        if face_detected:
            text += "Face Detected | "
        if gesture:
            text += f"Gesture: {gesture}"
            if gesture in gesture_voice:
                speak(gesture_voice[gesture])

        cv2.putText(frame, text, (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()

else:
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        img = Image.open(uploaded_file)
        frame = np.array(img)
        frame, face_detected = detect_face_and_iris(frame)
        frame, gesture = detect_gesture(frame)

        st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if gesture:
            st.success(f"Detected Gesture: {gesture}")
            speak(gesture_voice.get(gesture, "Gesture not recognized"))
