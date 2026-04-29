import os
# Unnecessary logs ko rokne ke liye
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page Configuration
st.set_page_config(page_title="Insect Detector", page_icon="🐞")
st.title("🐞 Ants vs Bees Detector")
st.write("Upload an image to identify the insect.")

# Model Loading with Cache to prevent re-loading
@st.cache_resource
def load_my_model():
    try:
        # File name must be exactly 'insect_model.keras'
        return tf.keras.models.load_model('insect_model.keras')
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# Image Uploader
uploaded_file = st.file_uploader("Choose a photo...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing (Size 160x160 based on your training)[cite: 1]
    img_resized = img.resize((160, 160))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction logic[cite: 1]
    with st.spinner('Analyzing...'):
        prediction = model.predict(img_array)
        # Class 0: ants, Class 1: bees[cite: 1]
        class_names = ['Ants', 'Bees']
        result = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

    st.success(f"Prediction: **{result}** ({confidence:.2f}% confidence)")
else:
    st.info("Please upload an image to start.")
