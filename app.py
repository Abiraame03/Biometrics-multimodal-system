import streamlit as st
import cv2
import numpy as np
import os
import pickle
import pandas as pd
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from gtts import gTTS
import mediapipe as mp
import tensorflow as tf
from deepface import DeepFace

# --- Configuration & Paths (Matching your GitHub structure) ---
MODELS_DIR = "gesture_auth_app models"
FACE_DB_PATH = os.path.join(MODELS_DIR, "face_embeddings.pkl")
IRIS_DB_PATH = os.path.join(MODELS_DIR, "iris_embeddings.pkl")
VOICE_MAP_PATH = os.path.join(MODELS_DIR, "voice_map.pkl") # Used for loading, but overwritten by table data
GESTURE_MODEL_PATH = os.path.join(MODELS_DIR, "gesture_model.h5")
GESTURE_LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "gesture_label_encoder.pkl")

# --- USER-DEFINED GESTURE COMMAND MAP ---
# This dictionary provides the default voice output for each gesture
NEW_GESTURE_COMMANDS = {
    "FIVE": "Hello! How are you?",
    "PEACE": "Thank you!",
    "THUMBS_UP": "Yes, I understand.",
    "THUMBS_DOWN": "No, please repeat.",
    "FIST": "Goodbye!",
    "CALL_ME": "Call someone for help!",
    "ROCK": "Let's go!",
    "OK": "Everything is fine."
}
# ----------------------------------------------------

# --- Streamlit Setup ---
st.set_page_config(page_title="Multi-Modal Biometric System", layout="wide")
st.title("🛡️ Multi-Modal Biometric & Gesture Control")

# =========================================================================
# 🔹 INITIALIZATION (Caching models for efficiency)
# =========================================================================

@st.cache_resource
def load_models_and_dbs():
    """Load all necessary models and databases from the new structure."""
    
    # 1. Load Biometric Databases
    face_user_db = {}; iris_user_db = {}; gesture_label_encoder = None
    
    if os.path.exists(FACE_DB_PATH):
        with open(FACE_DB_PATH, "rb") as f: face_user_db = pickle.load(f)
    if os.path.exists(IRIS_DB_PATH):
        with open(IRIS_DB_PATH, "rb") as f: iris_user_db = pickle.load(f)
    
    st.sidebar.success(f"Loaded {len(face_user_db)} face and {len(iris_user_db)} iris users.")

    # 2. Load Gesture Model and Encoder
    gesture_model = None
    if os.path.exists(GESTURE_MODEL_PATH):
        try:
            # Load Keras/TensorFlow model
            gesture_model = tf.keras.models.load_model(GESTURE_MODEL_PATH)
            st.sidebar.success("✅ Keras Gesture Model loaded.")
        except Exception as e:
            st.sidebar.error(f"❌ Error loading gesture model: {e}")
    
    if os.path.exists(GESTURE_LABEL_ENCODER_PATH):
        with open(GESTURE_LABEL_ENCODER_PATH, "rb") as f:
            gesture_label_encoder = pickle.load(f)
            st.sidebar.success("✅ Gesture Label Encoder loaded.")

    # 3. Load Voice/Command Map (OVERRIDING with the new table data)
    gesture_commands = {}
    if os.path.exists(VOICE_MAP_PATH):
        try:
            with open(VOICE_MAP_PATH, "rb") as f: 
                # Load existing map, if any
                gesture_commands = pickle.load(f)
                st.sidebar.info(f"Loaded existing {len(gesture_commands)} commands from voice_map.pkl.")
        except Exception as e:
            st.sidebar.warning(f"Could not load/read existing voice_map.pkl: {e}. Using new commands.")

    # Apply the new command map, ensuring the user's table takes priority
    gesture_commands.update(NEW_GESTURE_COMMANDS)
    st.sidebar.success(f"Applied {len(NEW_GESTURE_COMMANDS)} commands from the required table.")

    # 4. Initialize Mediapipe Hands (Needed for landmark extraction)
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)
    
    # 5. Placeholder for Emotion Detection (FER library replacement)
    def mock_emotion_detector(frame_rgb):
        """Mock detector since FER library is removed."""
        # Returns a structure similar to what FER would return
        return [{"emotions": {"neutral": 0.5, "happy": 0.2, "angry": 0.1, "fear": 0.2}}]

    return {
        "face_db": face_user_db,
        "iris_db": iris_user_db,
        "gesture_model": gesture_model,
        "label_encoder": gesture_label_encoder,
        "gesture_commands": gesture_commands,
        "hands": hands,
        "emotion_detector": mock_emotion_detector
    }

SYSTEM = load_models_and_dbs()
hands = SYSTEM["hands"]

# =========================================================================
# 🛠 HELPER FUNCTIONS
# =========================================================================

def speak(text, filename="output.mp3"):
    """Generates audio and plays it via Streamlit."""
    try:
        if text:
            tts = gTTS(text)
            mp3_fp = BytesIO()
            tts.write_to_fp(mp3_fp)
            mp3_fp.seek(0)
            st.audio(mp3_fp, format="audio/mp3")
    except Exception as e:
        st.error(f"Voice output failed: {e}")

def get_image_from_bytes(file_bytes):
    """Converts uploaded file bytes or camera input bytes to an OpenCV image."""
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def extract_hand_landmarks(frame):
    """Extracts normalized Mediapipe hand landmarks for gesture model input."""
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)
    if not results.multi_hand_landmarks: return None
    
    lm = results.multi_hand_landmarks[0]
    # Extract x,y coordinates
    points = np.array([(l.x, l.y) for l in lm.landmark])
    
    # Simple Normalization (Centering and Scaling)
    center = np.mean(points, axis=0)
    points -= center
    norm = np.linalg.norm(points)
    if norm > 0: points /= norm
    
    # The gesture model expects a flat feature vector (e.g., 21 points * 2 coords = 42)
    return points.flatten()

# =========================================================================
# 🎯 2. AUTHENTICATION HELPERS (REVISED)
# =========================================================================

def authenticate_face(frame, username, threshold=0.35):
    """Authenticates face using DeepFace's Facenet512 against loaded embeddings."""
    face_user_db = SYSTEM["face_db"]
    if username not in face_user_db: return False, -1.0
    
    try:
        # Get embedding of the captured image
        cap_emb = DeepFace.represent(frame, enforce_detection=True, model_name="Facenet512")[0]["embedding"]
    except:
        # Face not detected
        return False, -1.0
        
    best_similarity = -1
    # Compare against all known embeddings for the user
    for emb in face_user_db[username]:
        # Use cosine similarity (dot product of normalized vectors)
        cos_sim = np.dot(np.array(emb), np.array(cap_emb))
        if cos_sim > best_similarity:
            best_similarity = cos_sim
            
    return best_similarity >= threshold, best_similarity

def authenticate_iris(frame, username, threshold=0.6):
    """Placeholder for Iris authentication. Logic for feature extraction is assumed to be external/removed."""
    iris_user_db = SYSTEM["iris_db"]
    if username not in iris_user_db: return False, 0.0

    # NOTE: Actual Iris feature extraction (e.g., LBP/Haar-based) is required here.
    # Since the original complex dependency for this was removed, this section is mocked.
    
    # Placeholder: Assuming the user exists and the system would normally succeed.
    if username in iris_user_db:
        sim = 0.65 
        return sim >= threshold, sim
    else:
        return False, 0.0

def authenticate_hand(frame, username, threshold=2.5):
    """Placeholder for Hand authentication. Currently, just checks if a hand is detected."""
    # NOTE: For real hand authentication, you would compare the extracted landmark vector
    # against a stored hand embedding for the specific user.
    
    emb = extract_hand_landmarks(frame)
    if emb is None:
        return False, 999.0

    # Since a dedicated hand DB is not explicitly defined in the new structure, 
    # we use a very simple mock: just checking if a hand was detected by Mediapipe.
    best_dist = 1.0 # Mock high success
    return best_dist < threshold, best_dist

# =========================================================================
# 🚀 3. MASTER FLOWS: AUTH & GESTURE-EMOTION
# =========================================================================

def master_multi_modal_authenticate(frame, username):
    """Coordinates the multi-modal authentication process."""
    st.info(f"🔑 Attempting Multi-Modal Authentication for **{username}**...")

    # Run individual authentications
    auth_face, sim_face = authenticate_face(frame, username)
    auth_iris, sim_iris = authenticate_iris(frame, username)
    auth_hand, dist_hand = authenticate_hand(frame, username) # dist_hand is mocked here

    auth_results = {"Face": auth_face, "Iris": auth_iris, "Hand": auth_hand}
    successful_matches = sum(auth_results.values())

    # Display results
    st.subheader("Authentication Results")
    col1, col2, col3 = st.columns(3)
    
    with col1: st.metric("Face (Similarity)", f"{sim_face:.3f}", delta="PASS" if auth_face else "FAIL")
    with col2: st.metric("Iris (Mock Sim)", f"{sim_iris:.3f}", delta="PASS" if auth_iris else "FAIL")
    with col3: st.metric("Hand (Detection)", "Detected" if dist_hand < 2.5 else "Not Detected", delta="PASS" if auth_hand else "FAIL", delta_color="inverse")
    
    st.markdown("---")
    
    # Final decision: require at least 2 successful matches
    if successful_matches >= 2:
        status = f"✅ **AUTHENTICATION PASSED!** ({successful_matches}/3 modalities matched)"
        st.success(status)
        speak("Authentication successful!")
        return True, status
    else:
        status = f"❌ **AUTHENTICATION FAILED.** ({successful_matches}/3 modalities matched)"
        st.error(status)
        speak("Authentication failed.")
        return False, status

def master_gesture_emotion_voice(frame):
    """Coordinates gesture prediction and emotion-based voice command generation."""
    model = SYSTEM["gesture_model"]
    encoder = SYSTEM["label_encoder"]
    emotion_detector = SYSTEM["emotion_detector"]
    gesture_commands = SYSTEM["gesture_commands"]

    if model is None or encoder is None:
        st.error("❌ Cannot run. Gesture model/encoder not loaded.")
        return

    st.info("🧠 Analyzing Gesture and Emotion...")
    
    # 1. Detect Gesture using Keras Model
    detected_gesture = None
    emb = extract_hand_landmarks(frame) # Get the 42-element normalized feature vector
    
    if emb is not None:
        # Prepare input for Keras model: Add batch dimension (1, 42)
        X = np.expand_dims(emb, axis=0) 
        
        # Predict class probabilities
        prediction = model.predict(X, verbose=0)[0]
        pred_class_index = np.argmax(prediction)
        
        # Decode the predicted index back to a gesture name
        if pred_class_index < len(encoder.classes_):
            detected_gesture = encoder.classes_[pred_class_index]

    if detected_gesture is None:
        st.warning("❌ No recognizable hand gesture detected.")
        return

    # 2. Detect Emotion (Using Mock Detector)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = emotion_detector(frame_rgb)
    dominant_emotion = "neutral"
    
    if results:
        emotions = results[0]["emotions"]
        dominant_emotion = max(emotions, key=emotions.get)
        
        st.subheader("Emotion Detection Results (Mock)")
        emotions_df = pd.DataFrame(list(emotions.items()), columns=['Emotion', 'Confidence']).sort_values(by='Confidence', ascending=False)
        st.dataframe(emotions_df, use_container_width=True, hide_index=True)

    st.metric("🎯 Detected Gesture", detected_gesture)
    st.metric("🎭 Dominant Emotion", dominant_emotion)

    # 3. Command Mapping & Voice Output
    base_command = gesture_commands.get(detected_gesture, "Gesture recognized, but no default command assigned.")

    final_text = base_command
    
    # Conditional logic based on gesture and emotion (Custom Overrides)
    if detected_gesture == "FIVE" and dominant_emotion == "fear":
        final_text = "EMERGENCY! I need HELP immediately!"
    elif detected_gesture == "FIST" and dominant_emotion == "angry":
        final_text = "STOP! I am highly displeased with this."
    elif dominant_emotion == "happy":
        # Enhance the base command if the user is happy
        final_text = "That's wonderful! " + base_command

    st.markdown(f"🗣 **Final Voice Command:** `'{final_text}'`")
    speak(final_text)

# =========================================================================
# 🖼 MAIN UI
# =========================================================================

# Sidebar for common input
st.sidebar.markdown("## 👤 User Configuration")
name_input = st.sidebar.text_input("User Name for Authentication:", value="")

# Main Tabs
tab1, tab2 = st.tabs(["1. Authentication", "2. Command"])

with tab1:
    st.subheader("1. Run Multi-Modal Authentication")
    st.info("Capture: Show your **FACE** and an **OPEN HAND** clearly for authentication.")
    auth_cam_input = st.camera_input("Take Authentication Photo", key="auth_cam_final")
    
    if auth_cam_input is not None:
        name = name_input.strip()
        if not name:
            st.warning("⚠ Please enter a user name first in the sidebar.")
        else:
            frame = get_image_from_bytes(auth_cam_input.read())
            # Resize for better visibility (optional)
            resized_frame = cv2.resize(frame, (320, 240))
            st.image(resized_frame, channels="BGR", caption="Captured Image", use_column_width=False)
            
            master_multi_modal_authenticate(frame, name)

with tab2:
    st.subheader("2. Run Gesture-Emotion Voice Command")
    st.info("Capture: Show a **GESTURE** and express an **EMOTION**.")
    command_cam_input = st.camera_input("Take Command Photo", key="command_cam_final")
    
    if command_cam_input is not None:
        frame = get_image_from_bytes(command_cam_input.read())
        # Resize for better visibility (optional)
        resized_frame = cv2.resize(frame, (320, 240))
        st.image(resized_frame, channels="BGR", caption="Captured Image", use_column_width=False)
        
        master_gesture_emotion_voice(frame)

st.sidebar.markdown("---")
st.sidebar.markdown(f"**DB Status**")
st.sidebar.markdown(f"Face Users: {len(SYSTEM['face_db'])}")
st.sidebar.markdown(f"Iris Users: {len(SYSTEM['iris_db'])}")
st.sidebar.markdown(f"Gesture Model Loaded: {'Yes' if SYSTEM['gesture_model'] else 'No'}")
st.sidebar.markdown("Gesture Commands:")
st.sidebar.json(SYSTEM['gesture_commands'])
