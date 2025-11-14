import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import os
import gc
import torch

st.set_page_config(page_title='Melvin demo')
st.header('Extracting text from Image')

# File uploader
image = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'])

# Use /tmp for model storage (writable in Streamlit Cloud)
MODEL_DIR = "/tmp/easyocr_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Confidence threshold for marking results
CONF_THRESHOLD = 0.6

# Initialize OCR model once per session
if "reader" not in st.session_state:
    st.write("TRACE: Initializing EasyOCR model...")
    try:
        st.session_state.reader = easyocr.Reader(
            ['en'],
            gpu=False,
            model_storage_directory=MODEL_DIR,
            user_network_directory=MODEL_DIR
        )
        st.write("TRACE: Model initialized successfully.")
    except Exception as e:
        st.error(f"TRACE: Failed to initialize OCR model: {e}")
        st.stop()

reader = st.session_state.reader

if image is not None:
    st.write("TRACE: Image uploaded.")
    try:
        input_image = Image.open(image).convert('RGB')
        st.write("TRACE: Image converted to RGB.")

        # Downscale large images to avoid memory blowouts
        max_size = 1024
        input_image.thumbnail((max_size, max_size))
        st.write(f"TRACE: Image resized to {input_image.size}")

        st.image(input_image, caption="Uploaded Image", use_column_width=True)

        img_array = np.array(input_image)
        st.write(f"TRACE: Image array shape: {img_array.shape}, dtype: {img_array.dtype}")

        with st.spinner("Melvin at work..."):
            st.write("TRACE: Starting OCR readtext...")
            result = reader.readtext(img_array)
            st.write("TRACE: OCR readtext completed.")

            # Display all results with confidence markers
            st.write("### Extracted Text (all results)")
            for _, text, conf in result:
                if conf >= CONF_THRESHOLD:
                    st.write(f"✅ {text} (confidence: {conf:.2f})")
                else:
                    st.write(f"* {text} (low confidence: {conf:.2f})")

            st.success("Here you go!")

        # Cleanup memory after OCR
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        st.error(f"TRACE: OCR failed with error: {e}")
        # Reset reader if it crashed, so next run starts clean
        if "reader" in st.session_state:
            del st.session_state.reader
else:
    st.write("TRACE: No image uploaded yet.")

st.caption("Made by Melvin")
