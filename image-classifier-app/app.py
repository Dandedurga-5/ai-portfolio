import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

model = tf.keras.models.load_model("image_classifier.h5")

st.set_page_config(page_title="Image Classifier")

st.title("🐶🐱 AI Image Classifier")

file = st.file_uploader("Upload Image", type=["jpg","png","jpeg"])

if file:
    img = Image.open(file)
    st.image(img)

    img = img.resize((160,160))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0][0]

    if pred > 0.5:
        st.success(f"Dog 🐶 ({pred:.2f})")
    else:
        st.success(f"Cat 🐱 ({1-pred:.2f})")
