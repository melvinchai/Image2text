import easyocr
from PIL import Image, ImageOps
import sys

def load_and_fix_orientation(image_path: str) -> Image.Image:
    """
    Load an image and normalize EXIF orientation so it's upright.
    """
    img = Image.open(image_path)
    # Auto-rotate based on EXIF metadata
    img = ImageOps.exif_transpose(img)
    return img

def run_easyocr(image_path: str, langs=None):
    """
    Run EasyOCR on the given image and return results.
    """
    if langs is None:
        langs = ['en']  # English receipts; add 'ch_sim' if Chinese text expected

    # Initialize EasyOCR reader
    reader = easyocr.Reader(langs)

    # Fix orientation before OCR
    img = load_and_fix_orientation(image_path)

    # Run OCR
    results = reader.readtext(img)

    return results

def main():
    if len(sys.argv) < 2:
        print("Usage: python test3.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    results = run_easyocr(image_path)

    print("\n🧾 OCR Results:")
    for idx, (bbox, text, confidence) in enumerate(results):
        print(f"{idx}: {text} (confidence: {confidence:.2f})")

if __name__ == "__main__":
    main()
