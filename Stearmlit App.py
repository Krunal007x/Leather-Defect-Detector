import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the TensorFlow SavedModel
@st.cache_resource
def load_model():
    try:
        model = tf.saved_model.load("model")
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

# Preprocess the uploaded image
def preprocess_image(image: Image.Image, target_size=(224, 224)):
    image = image.convert("RGB")
    image = image.resize(target_size)
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array.astype(np.float32)

labels = ["Good", "Defective"]

st.title("Defect Detection without Keras")

model = load_model()

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file and model:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    input_data = preprocess_image(image)

    # Run inference using TensorFlow SavedModel
    infer = model.signatures["serving_default"]
    preds = infer(tf.convert_to_tensor(input_data))
    pred_tensor = list(preds.values())[0].numpy()

    pred_label = labels[np.argmax(pred_tensor)]
    confidence = np.max(pred_tensor)

    st.write(f"Prediction: **{pred_label}**")
    st.write(f"Confidence: {confidence:.2f}")
