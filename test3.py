import streamlit as st
import os, json, traceback
from PIL import Image, ImageOps
import numpy as np

# Print versions immediately (wrapped to avoid hard failures if imports fail later)
def safe_version(name, getter):
    try:
        v = getter()
        st.write(f"🔎 Trace: {name} version = {v}")
    except Exception as e:
        st.write(f"⚠️ Trace: Failed to get {name} version: {e}")

st.title("Receipt OCR Scanner (EasyOCR → JSON)")
st.write("🔎 Trace: App bootstrapped, entering main flow")

# Resolve model directory early and show path
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models", "easyocr")
st.write("🔎 Trace: MODEL_DIR =", MODEL_DIR)
st.write("🔎 Trace: MODEL_DIR exists =", os.path.isdir(MODEL_DIR))

# Try importing libraries after title so errors display inline
easyocr = None
torch = None
streamlit_mod = None
try:
    import easyocr as easyocr
    import torch as torch
    import streamlit as streamlit_mod  # alias just for version trace
    safe_version("Streamlit", lambda: streamlit_mod.__version__)
    safe_version("EasyOCR", lambda: easyocr.__version__)
    safe_version("Torch", lambda: torch.__version__)
except Exception as e:
    st.error("❌ Trace: Import error for core libraries: " + str(e))
    st.write(traceback.format_exc())
    st.stop()

# UI controls
threshold = st.slider(
    "Confidence threshold (flag low-confidence with *)",
    0.0, 1.0, 0.7, 0.05
)
uploaded_file = st.file_uploader("Upload a receipt image", type=["jpg", "jpeg", "png"])

# Initialize EasyOCR reader inside UI flow (no caching) and fully guarded
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
    # Warm-up with a tiny black image
    dummy = np.zeros((16, 16, 3), dtype=np.uint8)
    _ = reader.readtext(dummy)
    st.write("✅ Trace: Warm-up readtext() succeeded")
except Exception as e:
    st.error("❌ Trace: Failed to initialize EasyOCR Reader: " + str(e))
    st.write(traceback.format_exc())
    st.stop()

def load_and_fix_orientation(uploaded_file):
    st.write("🔎 Trace: Entering load_and_fix_orientation()")
    try:
        img = Image.open(uploaded_file)
        st.write(f"Trace: Loaded image mode={img.mode}, size={img.size}")
        img = ImageOps.exif_transpose(img)
        st.write("✅ Trace: EXIF orientation corrected")
        return img
    except Exception as e:
        st.error("❌ Trace: Failed to load/transpose image: " + str(e))
        st.write(traceback.format_exc())
        return None

def run_easyocr(img):
    st.write("🔎 Trace: Entering run_easyocr()")
    try:
        img_np = np.array(img)
        st.write(f"Trace: numpy array shape={img_np.shape}, dtype={img_np.dtype}")
        results = reader.readtext(img_np)
        st.write(f"✅ Trace: OCR found {len(results)} items")
        if results:
            st.write("Trace: First item sample:", results[0])
        return results
    except Exception as e:
        st.error("❌ Trace: reader.readtext failed: " + str(e))
        st.write(traceback.format_exc())
        return []

def build_structured_json(results, filename, threshold=0.7):
    st.write("🔎 Trace: Entering build_structured_json(), type(results) = " + str(type(results)))
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
        st.error("❌ Trace: Failed while building JSON: " + str(e))
        st.write(traceback.format_exc())
    return structured

if uploaded_file is None:
    st.write("🔎 Trace: No file uploaded yet")
else:
    try:
        st.write("🔎 Trace: File uploaded:", uploaded_file.name)
        img = load_and_fix_orientation(uploaded_file)
        if img is None:
            st.write("⚠️ Trace: Image is None after load; stopping")
            st.stop()

        st.image(img, caption="Upright receipt", use_column_width=True)

        with st.spinner("Scanning receipt with EasyOCR…"):
            results = run_easyocr(img)

        st.subheader("Raw OCR results")
        try:
            for idx, (bbox, text, confidence) in enumerate(results):
                marker = "*" if confidence < threshold else ""
                st.write(f"{idx}: {text}{marker} (confidence: {confidence:.2f})")
        except Exception as e:
            st.error("❌ Trace: Failed to iterate OCR results: " + str(e))
            st.write(traceback.format_exc())

        st.subheader("Structured JSON")
        structured = build_structured_json(results, uploaded_file.name, threshold=threshold)
        try:
            st.json(structured)
        except Exception as e:
            st.error("❌ Trace: st.json failed: " + str(e))
            st.write(traceback.format_exc())

        try:
            json_str = json.dumps(structured, indent=2)
            st.download_button(
                "Download JSON",
                json_str,
                file_name="receipt.json",
                mime="application/json"
            )
            st.write("✅ Trace: Download button rendered")
        except Exception as e:
            st.error("❌ Trace: Failed to render download button: " + str(e))
            st.write(traceback.format_exc())

    except Exception as e:
        st.error("❌ Fatal error caught in main block: " + str(e))
        st.write(traceback.format_exc())
        st.stop()
