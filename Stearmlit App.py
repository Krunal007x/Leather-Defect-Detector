import streamlit as st
from keras.models import load_model
import numpy as np
from PIL import Image
import io

# Constants
TARGET_SIZE = (224, 224)
LABELS = ["Good", "Defective"]

# Load model only once
@st.cache_resource
def load_model():
    try:
        return TFSMLayer("model", call_endpoint="serving_default")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

# Preprocess the input image
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    try:
        assert image.mode in ["RGB", "RGBA", "L"], "Unsupported image mode."
        image = image.convert("RGB")
        image = image.resize(target_size)
        img_array = np.array(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        st.error(f"⚠️ Image preprocessing failed: {e}")
        return None

# App layout
st.set_page_config(page_title="Defect Detection", layout="centered")
st.title("🔍 Defect Detection using Keras 3 + Streamlit")

# Load model
model = load_model()

uploaded_file = st.file_uploader("📁 Upload an image (JPG or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="📷 Uploaded Image", use_column_width=True)

        input_data = preprocess_image(image, TARGET_SIZE)

        if input_data is not None and model is not None:
            with st.spinner("🧠 Running inference..."):
                preds = model(input_data)
                pred_tensor = list(preds.values())[0]
                preds_np = pred_tensor.numpy()

                pred_label = LABELS[np.argmax(preds_np)]
                confidence = np.max(preds_np)

                st.success(f"🎯 Prediction: **{pred_label}**")
                st.info(f"📊 Confidence: `{confidence:.2%}`")
        else:
            st.error("🚫 Inference skipped due to earlier errors.")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
else:
    st.warning("👆 Please upload an image file to start.")
