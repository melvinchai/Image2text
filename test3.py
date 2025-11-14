import streamlit as st
from PIL import Image
import numpy as np
import easyocr
import os

st.set_page_config(page_title='Melvin demo')
st.header('Extracting text from Image')

# File uploader
image = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'])

# Ensure a writable model directory (Streamlit Cloud supports /tmp)
MODEL_DIR = "/tmp/easyocr_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Cache the OCR model so it loads only once
@st.cache_resource
def load_model():
    return easyocr.Reader(
        ['en'],
        gpu=False,  # force CPU-only
        model_storage_directory=MODEL_DIR,
        user_network_directory=MODEL_DIR
    )

try:
    reader = load_model()
except Exception as e:
    st.error(f"Failed to initialize OCR model: {e}")
    st.stop()

if image is not None:
    # Normalize image mode
    input_image = Image.open(image).convert('RGB')
    st.image(input_image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Melvin at work..."):
        try:
            # Run OCR
            result = reader.readtext(np.array(input_image))
            result_text = [text[1] for text in result]

            # Display extracted text
            st.write("### Extracted Text")
            st.write(result_text)
            st.success("Here you go!")
        except Exception as e:
            st.error(f"OCR error: {e}")
else:
    st.write("Upload an image to begin")

st.caption("Made by Melvin")
