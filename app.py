import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from gtts import gTTS
import tempfile
import os

st.set_page_config(page_title="Biometric Multimodal Emotion & Gesture App", layout="wide")

st.title("🎯 Multimodal Emotion + Gesture Recognition System")
st.markdown("This app detects **face**, **iris**, and **gesture** from live webcam feed using a TensorFlow Lite model and gives **voice feedback** of the detected emotion.")

# Load TFLite model
model_path = "gesture_model.tflite"
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# Map gestures → emotions
gesture_to_emotion = {
    "thumbs_up": "Happy",
    "thumbs_down": "Sad",
    "peace": "Calm",
    "fist": "Angry",
    "open_palm": "Surprised"
}

# Voice output function
def speak_emotion(emotion):
    tts = gTTS(text=f"The detected emotion is {emotion}", lang='en')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name, format="audio/mp3")

# Run webcam
run = st.toggle("Enable Camera")

if run:
    cap = cv2.VideoCapture(0)
    with mp_face.FaceMesh(max_num_faces=1) as face_mesh, mp_hands.Hands(max_num_hands=1) as hands:
        stframe = st.empty()
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access camera.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Face + iris detection
            face_result = face_mesh.process(rgb)
            if face_result.multi_face_landmarks:
                for landmarks in face_result.multi_face_landmarks:
                    mp_draw.draw_landmarks(frame, landmarks, mp_face.FACEMESH_CONTOURS)

            # Gesture detection
            hand_result = hands.process(rgb)
            emotion_detected = None
            if hand_result.multi_hand_landmarks:
                for hand_landmarks in hand_result.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    lm = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
                    lm = np.expand_dims(lm, axis=0).astype(np.float32)

                    # Model inference
                    interpreter.set_tensor(input_details[0]['index'], lm)
                    interpreter.invoke()
                    pred = interpreter.get_tensor(output_details[0]['index'])
                    gesture_idx = int(np.argmax(pred))

                    # Replace with your gesture label list
                    gesture_labels = list(gesture_to_emotion.keys())
                    if gesture_idx < len(gesture_labels):
                        gesture = gesture_labels[gesture_idx]
                        emotion_detected = gesture_to_emotion.get(gesture, "Neutral")

            cv2.putText(frame, f"Emotion: {emotion_detected or 'Detecting...'}", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            stframe.image(frame, channels="BGR")

            if emotion_detected:
                speak_emotion(emotion_detected)
                break

        cap.release()
        st.success("✅ Detection completed.")
else:
    st.info("👆 Turn on the camera toggle to start detection.")
