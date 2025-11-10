import streamlit as st
import cv2, mediapipe as mp, numpy as np, pickle, requests, tensorflow as tf
from gtts import gTTS
import os, time

# ============================================================
# 🔹 Load models from GitHub repository
# ============================================================
@st.cache_resource
def load_models():
    repo_url = "https://github.com/Abiraame03/Biometrics-multimodal-system/raw/main/gesture%20auth%20app%20models/"
    files = {
        "gesture_model": "gesture_model.tflite",
        "encoder": "gesture_label_encoder.pkl",
        "voice_map": "voice_map.pkl"
    }

    local_models = {}
    os.makedirs("models", exist_ok=True)
    for key, fname in files.items():
        url = repo_url + fname
        r = requests.get(url)
        if r.status_code == 200:
            with open(f"models/{fname}", "wb") as f:
                f.write(r.content)
            local_models[key] = f"models/{fname}"
        else:
            st.error(f"❌ Couldn't fetch {fname} from repo.")
    return local_models


# ============================================================
# 🔹 Initialize
# ============================================================
models = load_models()
interpreter = tf.lite.Interpreter(model_path=models["gesture_model"])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

encoder = pickle.load(open(models["encoder"], "rb"))
voice_map = pickle.load(open(models["voice_map"], "rb"))

mp_hands = mp.solutions.hands

st.set_page_config(page_title="Universal Gesture Recognition", layout="centered")
st.title("✋ Universal Gesture Recognition System")
st.markdown("Recognize common hand gestures in real-time — usable by **any user** 👤")

run = st.checkbox("🎥 Start Camera")
stframe = st.image([])
hands = mp_hands.Hands(max_num_hands=1)

# ============================================================
# ✋ Real-time Gesture Recognition
# ============================================================
if run:
    cap = cv2.VideoCapture(0)
    prev_gesture = None
    last_spoken = time.time()

    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("⚠️ Camera not accessible.")
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand in result.multi_hand_landmarks:
                pts = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
                input_data = np.expand_dims(pts.astype(np.float32), axis=0)
                interpreter.set_tensor(input_details[0]['index'], input_data)
                interpreter.invoke()
                output_data = interpreter.get_tensor(output_details[0]['index'])
                gesture = encoder.inverse_transform([np.argmax(output_data)])[0]
                confidence = np.max(output_data)

                cv2.putText(frame, f"{gesture} ({confidence:.2f})", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,255,0), 3)

                # 🔊 Speak gesture name once per detection
                if gesture != prev_gesture or time.time() - last_spoken > 3:
                    text = voice_map.get(gesture, gesture)
                    tts = gTTS(f"Detected {text}")
                    tts.save("voice.mp3")
                    os.system("start voice.mp3" if os.name == "nt" else "afplay voice.mp3")
                    last_spoken = time.time()
                    prev_gesture = gesture

        stframe.image(frame, channels="BGR")

    cap.release()
    st.success("✅ Camera stopped.")

st.markdown("---")
st.caption("Developed the Universal Gesture Recognition System")
