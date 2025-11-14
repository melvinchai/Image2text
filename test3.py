import streamlit as st
import easyocr
from PIL import Image, ImageOps
import numpy as np
import json
import os

# --- Configure EasyOCR to use bundled models ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "easyocr")

@st.cache_resource
def get_reader():
    # Force CPU mode for Streamlit Cloud
    reader = easyocr.Reader(
        ['en', 'ch_sim'],   # add languages you need
        gpu=False,
        model_storage_directory=MODEL_DIR,
        user_network_directory=MODEL_DIR,
        download_enabled=False
    )
    # Warm-up: trigger model load with a tiny dummy image
    dummy = np.zeros((16, 16, 3), dtype=np.uint8)
    try:
        _ = reader.readtext(dummy)
    except Exception:
        pass
    return reader

with st.spinner("Initializing EasyOCR models… first run may take a few seconds"):
    reader = get_reader()

# --- Utility functions ---
def load_and_fix_orientation(uploaded_file):
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    return img

def run_easyocr(img):
    img_np = np.array(img)
    results = reader.readtext(img_np)
    return results

def build_structured_json(results, filename, threshold=0.7):
    structured = {
        "filename": filename,
        "vendor_name": None,
        "date": None,
        "currency": "RM",
        "total_amount": None,
        "payment_method": None,
        "invoice_number": None,
        "line_items": []
    }
    for idx, (bbox, text, confidence) in enumerate(results):
        flagged_text = text + (" *" if confidence < threshold else "")
        structured["line_items"].append({
            "description": flagged_text,
            "code": None,
            "quantity": None,
            "unit_price": None,
            "line_total": None,
            "confidence": confidence
        })
    return structured

# --- Streamlit UI ---
st.title("Receipt OCR Scanner (EasyOCR → JSON)")

# Confidence threshold slider
threshold = st.slider(
    "Confidence threshold (flag low-confidence with *)",
    0.0, 1.0, 0.7, 0.05
)

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_and_fix_orientation(uploaded_file)
    st.image(img, caption="Upright receipt", use_column_width=True)

    with st.spinner("Scanning receipt with EasyOCR…"):
        results = run_easyocr(img)

    st.subheader("Raw OCR results")
    for idx, (bbox, text, confidence) in enumerate(results):
        marker = "*" if confidence < threshold else ""
        st.write(f"{idx}: {text}{marker} (confidence: {confidence:.2f})")

    st.subheader("Structured JSON")
    structured = build_structured_json(results, uploaded_file.name, threshold=threshold)
    st.json(structured)

    # Allow download
    json_str = json.dumps(structured, indent=2)
    st.download_button(
        "Download JSON",
        json_str,
        file_name="receipt.json",
        mime="application/json"
    )
