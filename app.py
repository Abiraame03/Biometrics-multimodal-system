import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import requests
import io

# -------------------------------
# 🌐 App Config
# -------------------------------
st.set_page_config(page_title="Multimodal Biometric System", layout="centered")
st.title("🧠 Multimodal Biometric System (Gesture Recognition)")

st.markdown("""
This app uses your **camera** to recognize gestures in real-time using a TensorFlow Lite model.
The model is fetched directly from your GitHub repository.
""")

# -------------------------------
# ⚙️ Load TFLite Model from GitHub
# -------------------------------
MODEL_URL = "https://github.com/Abiraame03/Biometrics-multimodal-system/raw/main/models/gesture_model.tflite"

@st.cache_resource
def load_tflite_model():
    response = requests.get(MODEL_URL)
    response.raise_for_status()
    model_bytes = io.BytesIO(response.content)
    interpreter = tf.lite.Interpreter(model_content=model_bytes.read())
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------------
# 🧩 Class Labels
# -------------------------------
CLASS_NAMES = ["Thumbs Up 👍", "Thumbs Down 👎", "Peace ✌️", "Stop ✋", "OK 👌", "Fist ✊"]

# -------------------------------
# 📸 Camera Input
# -------------------------------
st.header("📷 Capture Your Gesture")
img_file = st.camera_input("Show your hand gesture below")

if img_file is not None:
    image = Image.open(img_file)
    st.image(image, caption="Captured Gesture", use_column_width=True)

    # Preprocess the image
    img = image.convert("RGB")
    img = img.resize((128, 128))  # Change to your model’s input size
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    # Get prediction
    predicted_class = CLASS_NAMES[np.argmax(preds)]
    confidence = float(np.max(preds)) * 100

    st.success(f"### 🧾 Prediction: {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    st.progress(confidence / 100)

else:
    st.info("Please show your gesture to the camera.")

# -------------------------------
# 🔮 For Future Expansion
# -------------------------------
st.markdown("""
---
### 🌍 Future Scope
- Add **Face** and **Iris** authentication (same pipeline).
- Integrate **Emotion Detection** via MediaPipe.
- Combine all into a single **multimodal authentication system**.
- Deploy on mobile or IoT devices using **TensorFlow Lite**.
""")
