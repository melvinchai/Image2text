import streamlit as st
import easyocr
from PIL import Image, ImageOps
import numpy as np
import json
import os
import traceback

# --- Configure EasyOCR to use bundled models ---
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "easyocr")
st.write("🔎 Trace: Starting app, MODEL_DIR =", MODEL_DIR)

@st.cache_resource
def get_reader():
    st.write("🔎 Trace: Entering get_reader()")
    try:
        reader = easyocr.Reader(
            ['en', 'ch_sim'],   # ✅ correct language codes
            gpu=False,
            model_storage_directory=MODEL_DIR,
            user_network_directory=MODEL_DIR,
            download_enabled=False
        )
        st.write("✅ Trace: EasyOCR Reader initialized successfully")
        # Warm-up
        dummy = np.zeros((16, 16, 3), dtype=np.uint8)
        _ = reader.readtext(dummy)
        st.write("✅ Trace: Warm-up readtext() succeeded")
        return reader
    except Exception as e:
        st.error("❌ Trace: Failed to initialize EasyOCR Reader: " + str(e))
        st.write(traceback.format_exc())
        return None

with st.spinner("Initializing EasyOCR models…"):
    reader = get_reader()

if reader is None:
    st.stop()

# --- Utility functions ---
def load_and_fix_orientation(uploaded_file):
    st.write("🔎 Trace: Entering load_and_fix_orientation()")
    try:
        img = Image.open(uploaded_file)
        img = ImageOps.exif_transpose(img)
        st.write("✅ Trace: Image loaded and orientation fixed")
        return img
    except Exception as e:
        st.error("❌ Trace: Failed to load image: " + str(e))
        st.write(traceback.format_exc())
        return None

def run_easyocr(img):
    st.write("🔎 Trace: Entering run_easyocr()")
    try:
        img_np = np.array(img)
        st.write(f"Trace: Image converted to numpy array, shape={img_np.shape}, dtype={img_np.dtype}")
        results = reader.readtext(img_np)
        st.write(f"✅ Trace: OCR returned {len(results)} results")
        if results:
            st.write("Trace: First result sample:", results[0])
        return results
    except Exception as e:
        st.error("❌ Trace: OCR failed: " + str(e))
        st.write(traceback.format_exc())
        return []

def build_structured_json(results, filename, threshold=0.7):
    st.write("🔎 Trace: Entering build_structured_json() with results type:", type(results))
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
    try:
        for idx, item in enumerate(results):
            st.write(f"Trace: Processing result {idx}: {item}")
            bbox, text, confidence = item
            flagged_text = text + (" *" if confidence < threshold else "")
            structured["line_items"].append({
                "description": flagged_text,
                "code": None,
                "quantity": None,
                "unit_price": None,
                "line_total": None,
                "confidence": confidence
            })
        st.write("✅ Trace: Structured JSON built successfully")
    except Exception as e:
        st.error("❌ Trace: Failed to build JSON: " + str(e))
        st.write(traceback.format_exc())
    return structured

# --- Streamlit UI ---
st.title("Receipt OCR Scanner (EasyOCR → JSON)")

threshold = st.slider(
    "Confidence threshold (flag low-confidence with *)",
    0.0, 1.0, 0.7, 0.05
)

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("🔎 Trace: File uploaded:", uploaded_file.name)
    img = load_and_fix_orientation(uploaded_file)
    if img is None:
        st.stop()

    st.image(img, caption="Upright receipt", use_column_width=True)

    try:
        with st.spinner("Scanning receipt with EasyOCR…"):
            results = run_easyocr(img)

        st.subheader("Raw OCR results")
        for idx, (bbox, text, confidence) in enumerate(results):
            marker = "*" if confidence < threshold else ""
            st.write(f"{idx}: {text}{marker} (confidence: {confidence:.2f})")

        st.subheader("Structured JSON")
        structured = build_structured_json(results, uploaded_file.name, threshold=threshold)
        st.json(structured)

        json_str = json.dumps(structured, indent=2)
        st.download_button(
            "Download JSON",
            json_str,
            file_name="receipt.json",
            mime="application/json"
        )
    except Exception as e:
        st.error("❌ Fatal error caught: " + str(e))
        st.write(traceback.format_exc())
        st.stop()
else:
    st.write("🔎 Trace: No file uploaded yet")
