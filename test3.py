import streamlit as st
import os, json, traceback
from PIL import Image, ImageOps
import numpy as np

# --- Startup traces ---
st.title("Receipt OCR Scanner (EasyOCR → JSON)")
st.write("🔎 Trace: App bootstrapped, entering main flow")

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "easyocr")
st.write("🔎 Trace: MODEL_DIR =", MODEL_DIR)
st.write("🔎 Trace: MODEL_DIR exists =", os.path.isdir(MODEL_DIR))

# Import libraries after title so errors show inline
try:
    import easyocr
    import torch
    st.write("🔎 Trace: Streamlit version =", st.__version__)
    st.write("🔎 Trace: EasyOCR version =", easyocr.__version__)
    st.write("🔎 Trace: Torch version =", torch.__version__)
except Exception as e:
    st.error("❌ Trace: Import error: " + str(e))
    st.write(traceback.format_exc())
    st.stop()

# --- Initialize EasyOCR reader ---
reader = None
try:
    st.write("🔎 Trace: Initializing EasyOCR Reader (gpu=False, local models)")
    reader = easyocr.Reader(
        ['en', 'ch_sim'],
        gpu=False,
        model_storage_directory=MODEL_DIR,
        user_network_directory=MODEL_DIR,
        download_enabled=False
    )
    st.write("✅ Trace: EasyOCR Reader initialized")
    dummy = np.zeros((16, 16, 3), dtype=np.uint8)
    _ = reader.readtext(dummy)
    st.write("✅ Trace: Warm-up readtext() succeeded")
except Exception as e:
    st.error("❌ Trace: Failed to initialize EasyOCR Reader: " + str(e))
    st.write(traceback.format_exc())
    st.stop()

# --- Utility functions ---
def load_and_fix_orientation(uploaded_file):
    st.write("🔎 Trace: Entering load_and_fix_orientation()")
    try:
        img = Image.open(uploaded_file)
        st.write(f"✅ Trace: Image opened, mode={img.mode}, size={img.size}")
    except Exception as e:
        st.error("❌ Trace: Image.open() failed: " + str(e))
        st.write(traceback.format_exc())
        return None

    try:
        img = ImageOps.exif_transpose(img)
        st.write("✅ Trace: EXIF orientation corrected")
    except Exception as e:
        st.error("❌ Trace: ImageOps.exif_transpose() failed: " + str(e))
        st.write(traceback.format_exc())
        # fallback: return original
    return img

def run_easyocr(img):
    st.write("🔎 Trace: Entering run_easyocr()")
    try:
        # Force RGB
        if img.mode != "RGB":
            st.write(f"⚠️ Trace: Converting image mode {img.mode} → RGB")
            img = img.convert("RGB")

        # Resize if too large
        MAX_SIZE = 2000
        w, h = img.size
        if max(w, h) > MAX_SIZE:
            scale = MAX_SIZE / max(w, h)
            new_size = (int(w * scale), int(h * scale))
            st.write(f"⚠️ Trace: Resizing image from {img.size} → {new_size}")
            img = img.resize(new_size, Image.LANCZOS)

        img_np = np.array(img)
        st.write(f"Trace: numpy array shape={img_np.shape}, dtype={img_np.dtype}")

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
    st.write("🔎 Trace: Entering build_structured_json()")
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
        st.write("✅ Trace: Structured JSON built")
    except Exception as e:
        st.error("❌ Trace: Failed to build JSON: " + str(e))
        st.write(traceback.format_exc())
    return structured

# --- Streamlit UI ---
threshold = st.slider(
    "Confidence threshold (flag low-confidence with *)",
    0.0, 1.0, 0.7, 0.05
)
uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.write("🔎 Trace: No file uploaded yet")
else:
    try:
        st.write("🔎 Trace: File uploaded:", uploaded_file.name)
        img = load_and_fix_orientation(uploaded_file)
        if img is None:
            st.stop()

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

        json_str = json.dumps(structured, indent=2)
        st.download_button(
            "Download JSON",
            json_str,
            file_name="receipt.json",
            mime="application/json"
        )
        st.write("✅ Trace: Download button rendered")
    except Exception as e:
        st.error("❌ Fatal error caught in main block: " + str(e))
        st.write(traceback.format_exc())
        st.stop()
