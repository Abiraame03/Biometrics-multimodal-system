import streamlit as st
import cv2, mediapipe as mp, numpy as np, pickle, requests, os
from gtts import gTTS
import tempfile
from tflite_runtime.interpreter import Interpreter

# ============================================================
# CONFIG
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
st.set_page_config(page_title="Multimodal Biometric System", layout="centered")

st.title("🧠 Multimodal Biometric Authentication System")
st.markdown("### 👁️ Face + Iris + ✋ Gesture Recognition with Voice Feedback")

# ============================================================
# 🔹 Load models from GitHub repository
# ============================================================
@st.cache_resource
def load_models():
    repo_url = "https://github.com/Abiraame03/Biometrics-multimodal-system/raw/main/gesture%20auth%20app%20models/"
    files = {
        "gesture_model": "gesture_model.tflite",
        "encoder": "gesture_label_encoder.pkl",
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

models = load_models()

# ============================================================
# 🔹 Load Gesture Model
# ============================================================
interpreter = Interpreter(model_path=models["gesture_model"])
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

encoder = pickle.load(open(models["encoder"], "rb"))

# ============================================================
# 🔹 Mediapipe Setup
# ============================================================
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_iris = mp.solutions.face_mesh

# ============================================================
# 🔹 Gesture → Voice mapping
# ============================================================
gesture_voice_map = {
    "FIVE": "Hello! How are you?",
    "PEACE": "Thank you!",
    "THUMBS_UP": "Yes, I understand.",
    "THUMBS_DOWN": "No, please repeat.",
    "FIST": "Goodbye!",
    "CALL_ME": "Call someone for help!",
    "ROCK": "Let's go!",
    "OK": "Everything is fine."
}

# ============================================================
# 🔹 Helper functions
# ============================================================
def speak_text(text):
    """Convert gesture meaning to speech"""
    tts = gTTS(text)
    temp_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_path.name)
    st.audio(temp_path.name, format="audio/mp3")

def predict_gesture(frame):
    """Run the gesture recognition model"""
    img = cv2.resize(frame, (128,128))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    if len(input_details[0]['shape']) == 2:
        img = img.reshape((1, -1))

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    pred_idx = np.argmax(output_data)
    gesture = encoder.inverse_transform([pred_idx])[0]
    confidence = float(np.max(output_data))
    return gesture, confidence

# ============================================================
# 🔹 Streamlit UI
# ============================================================
st.markdown("---")
st.subheader("🎥 Live Camera Feed")

run = st.checkbox("Start Camera")

if run:
    cap = cv2.VideoCapture(0)
    stframe = st.empty()

    face_detect = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    iris_mesh = mp_iris.FaceMesh(max_num_faces=1)
    hands = mp_hands.Hands(max_num_hands=1)

    st.info("🟢 Camera running... Press Stop to end session")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not accessible.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- Face detection ---
        faces = face_detect.process(rgb)
        if faces.detections:
            for det in faces.detections:
                box = det.location_data.relative_bounding_box
                h, w, _ = frame.shape
                x, y, ww, hh = int(box.xmin * w), int(box.ymin * h), int(box.width * w), int(box.height * h)
                cv2.rectangle(frame, (x, y), (x+ww, y+hh), (0, 255, 0), 2)
                cv2.putText(frame, "Face Detected", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        # --- Iris detection ---
        iris_results = iris_mesh.process(rgb)
        if iris_results.multi_face_landmarks:
            cv2.putText(frame, "Iris Detected", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

        # --- Gesture detection ---
        hand_results = hands.process(rgb)
        if hand_results.multi_hand_landmarks:
            gesture, conf = predict_gesture(frame)
            cv2.putText(frame, f"{gesture} ({conf:.2f})", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 3)
            if conf > 0.7:
                message = gesture_voice_map.get(gesture, "")
                if message:
                    speak_text(message)

        stframe.image(frame, channels="BGR")

    cap.release()

st.markdown("---")
st.caption("Developed for multimodal user-independent authentication with real-time voice feedback.")
