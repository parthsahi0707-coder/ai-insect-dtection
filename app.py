import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page setting
st.set_page_config(page_title="Insect Detector", page_icon="🐞")
st.title("🐞 Ants vs Bees Detector")
st.write("Upload an image to identify if it's an Ant or a Bee.")

# Model load
@st.cache_resource
def load_model():
    # File name must match exactly
    return tf.keras.models.load_model('insect_model.keras')

model = load_model()

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Uploaded Image', use_container_width=True)
    
    # Preprocessing (Size 160x160 as per your notebook)
    img_resized = img.resize((160, 160))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Prediction[cite: 1]
    prediction = model.predict(img_array)
    # Classes: 0 = ants, 1 = bees[cite: 1]
    class_names = ['Ants', 'Bees']
    result = class_names[np.argmax(prediction)]
    
    st.success(f"I am confident this is: **{result}**")
