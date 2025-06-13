import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Constants
TARGET_SIZE = (224, 224)
LABELS = ["Good", "Defective"]  # Update these labels if needed

# Load Keras .h5 model once
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("keras_model.h5")
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# Preprocess the image
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    try:
        image = image.convert("RGB")
        image = image.resize(target_size)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"⚠️ Image preprocessing failed: {e}")
        return None

# Streamlit App
st.set_page_config(page_title="Defect Detection", layout="centered")
st.title("🔍 Defect Detection using Keras + Streamlit")

# Load model
model = load_model()

# Image upload
uploaded_file = st.file_uploader("📁 Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    input_data = preprocess_image(image, TARGET_SIZE)

    if input_data is not None and model is not None:
        with st.spinner("🧠 Running inference..."):
            pred_tensor = model.predict(input_data)
            pred_label = LABELS[np.argmax(pred_tensor)]
            confidence = np.max(pred_tensor)

            st.success(f"🎯 Prediction: **{pred_label}**")
            st.info(f"📊 Confidence: `{confidence:.2%}`")
    else:
        st.error("🚫 Inference skipped due to earlier errors.")
else:
    st.warning("👆 Please upload an image file to start.")




