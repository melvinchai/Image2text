import streamlit as st
from PIL import Image
import numpy as np
import easyocr

st.set_page_config(page_title='Melvin demo')
st.header('Extracting text from Image')

# Upload image
image = st.file_uploader("Upload your image", type=['png', 'jpg', 'jpeg'])

# Load OCR model once
@st.cache_resource
def load_model():
    return easyocr.Reader(['en'])

reader = load_model()

if image is not None:
    # Ensure image is in RGB mode
    input_image = Image.open(image).convert('RGB')
    st.image(input_image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Melvin at work..."):
        # Run OCR
        result = reader.readtext(np.array(input_image))
        result_text = [text[1] for text in result]

        # Display extracted text
        st.write("### Extracted Text")
        st.write(result_text)

    st.success("Here you go!")
else:
    st.write("Upload an image to begin")

st.caption("Made by Melvin")
