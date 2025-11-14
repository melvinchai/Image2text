import streamlit as st
import easyocr
from PIL import Image, ImageOps

# Initialize EasyOCR reader once (English receipts; add 'ch_sim' if needed)
reader = easyocr.Reader(['en'])

def load_and_fix_orientation(uploaded_file):
    """Load an uploaded image and normalize EXIF orientation so it's upright."""
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    return img

def run_easyocr(img):
    """Run EasyOCR on a PIL image and return results."""
    results = reader.readtext(img)
    return results

# --- Streamlit UI ---
st.title("Receipt OCR Scanner (EasyOCR)")

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show the image
    st.image(uploaded_file, caption="Uploaded receipt", use_column_width=True)

    with st.spinner("Scanning receipt with EasyOCR..."):
        img = load_and_fix_orientation(uploaded_file)
        results = run_easyocr(img)

    st.subheader("OCR Results")
    for idx, (bbox, text, confidence) in enumerate(results):
        st.write(f"{idx}: {text} (confidence: {confidence:.2f})")
