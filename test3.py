import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import os

st.set_page_config(page_title='Melvin demo')
st.header('Extracting text from Image')

# File uploader
image = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'])

# Ensure a writable model directory
MODEL_DIR = "/tmp/easyocr_models"
os.makedirs(MODEL_DIR, exist_ok=True)

@st.cache_resource
def load_model():
    st.write("TRACE: Initializing EasyOCR model...")
    reader = easyocr.Reader(
        ['en'],
        gpu=False,
        model_storage_directory=MODEL_DIR,
        user_network_directory=MODEL_DIR
    )
    st.write("TRACE: Model initialized successfully.")
    return reader

try:
    reader = load_model()
except Exception as e:
    st.error(f"TRACE: Failed to initialize OCR model: {e}")
    st.stop()

if image is not None:
    st.write("TRACE: Image uploaded.")
    try:
        input_image = Image.open(image).convert('RGB')
        st.write("TRACE: Image converted to RGB.")
        st.image(input_image, caption="Uploaded Image", use_column_width=True)

        with st.spinner("Melvin at work..."):
            st.write("TRACE: Starting OCR readtext...")
            result = reader.readtext(np.array(input_image))
            st.write("TRACE: OCR readtext completed.")

            result_text = [text[1] for text in result]
            st.write("TRACE: Extracted text list constructed.")

            st.write("### Extracted Text")
            st.write(result_text)
            st.success("Here you go!")
    except Exception as e:
        st.error(f"TRACE: OCR failed with error: {e}")
else:
    st.write("TRACE: No image uploaded yet.")

st.caption("Made by Melvin")
