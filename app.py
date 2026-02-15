import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import json
from tensorflow.keras.applications.efficientnet import preprocess_input

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="Cattle Breed Classifier",
    page_icon="🐄",
    layout="centered"
)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("model/best_model.keras")
    return model

model = load_model()

with open("class_indices.json") as f:
    class_indices = json.load(f)

classes = list(class_indices.keys())

# ==========================
# UI DESIGN
# ==========================
st.markdown(
    """
    <h1 style='text-align: center; color: #2E86C1;'>🐄 Cattle Breed Classifier</h1>
    <p style='text-align: center;'>Upload an image and let AI identify the breed.</p>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload Cattle Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((256, 256))
    img_array = np.array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = classes[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown("---")
    st.success(f"### 🏆 Predicted Breed: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
