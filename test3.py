import streamlit as st
from PIL import Image
import numpy as np
import easyocr as ocr

# Optional: handle PDFs
from pdf2image import convert_from_bytes

st.set_page_config(page_title="Melvin demo")
st.header("Extracting text from Image")

# Upload widget
image = st.file_uploader(
    label="Upload your image",
    type=["png", "jpg", "jpeg", "pdf"]
)

@st.cache_resource
def load_model():
    # Use a writable directory for model cache
    return ocr.Reader(["en"], model_storage_directory="/tmp/.easyocr")

reader = load_model()

if image is not None:
    # Handle PDF uploads
    if image.type == "application/pdf":
        pages = convert_from_bytes(image.read())
        input_image = pages[0].convert("RGB")
    else:
        input_image = Image.open(image).convert("RGB")

    st.image(input_image, caption="Uploaded image", use_column_width=True)

    # Downscale large images to avoid memory spikes
    max_dim = 2000
    w, h = input_image.size
    scale = min(1.0, max_dim / max(w, h))
    if scale < 1.0:
        input_image = input_image.resize((int(w * scale), int(h * scale)))

    with st.spinner("Melvin at work"):
        try:
            result = reader.readtext(np.array(input_image))
            result_text = [
                {"text": text, "confidence": float(conf)}
                for (_, text, conf) in result
            ]
            st.success("OCR complete")
            st.write(result_text)
        except Exception as e:
            st.error(f"OCR failed: {e}")
else:
    st.write("Upload an image")

st.caption("Made by Melvin")
