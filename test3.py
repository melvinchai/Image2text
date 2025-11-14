import streamlit as st
import easyocr
from PIL import Image, ImageOps
import numpy as np
import json

# Initialize EasyOCR reader once
reader = easyocr.Reader(['en'])

def load_and_fix_orientation(uploaded_file):
    """Load an uploaded image and normalize EXIF orientation so it's upright."""
    img = Image.open(uploaded_file)
    img = ImageOps.exif_transpose(img)
    return img

def run_easyocr(img):
    """Run EasyOCR on a PIL image by converting to NumPy array."""
    img_np = np.array(img)  # Convert PIL → NumPy
    results = reader.readtext(img_np)
    return results

def build_structured_json(results, filename, threshold=0.7):
    """Build structured JSON skeleton from OCR results, flagging low-confidence items."""
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

uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Fix orientation before showing
    img = load_and_fix_orientation(uploaded_file)
    st.image(img, caption="Upright receipt", use_column_width=True)

    with st.spinner("Scanning receipt with EasyOCR..."):
        results = run_easyocr(img)

    st.subheader("Raw OCR Results")
    threshold = 0.7
    for idx, (bbox, text, confidence) in enumerate(results):
        marker = "*" if confidence < threshold else ""
        st.write(f"{idx}: {text}{marker} (confidence: {confidence:.2f})")

    st.subheader("Structured JSON (for Claude reconciliation)")
    structured = build_structured_json(results, uploaded_file.name, threshold=threshold)
    st.json(structured)

    json_str = json.dumps(structured, indent=2)
    st.download_button("Download JSON", json_str, file_name="receipt.json", mime="application/json")
