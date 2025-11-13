import streamlit as st # web
from PIL import Image # image processing
import numpy as np # image processing
import easyocr as ocr # ocr

st.set_page_config(page_title='Melvin demo')
st.header('Extracting text from Image')

image = st.file_uploader(label = "Upload your image", type=['png','jpg','jpeg','pdf'])


def load_model():
    reader = ocr.Reader(['en'],model_storage_directory='.')
    return reader
reader = load_model()

if image is not None:
    input_image = Image.open(image)
    st.image(input_image)
    
    with st.spinner ("Melvin at work"):
    
        result = reader.readtext(np.array(input_image))
        result_text = []                             

        for text in result:
            result_text.append(text[1])
        st.write(result_text)
    st.success("Here you go")
    # st.balloons()
else:
    st.write("Upload an image")
st.caption("Made by Melvin")
