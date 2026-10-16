"""
Preview local PDF samples — no model or GPU needed.

Converts each page of every PDF in ./samples/ to an image,
saves the page images to ./previews/, and prints the OCR text for each file.

Usage:
    python preview_samples.py            # all PDFs in ./samples/
    python preview_samples.py --n 3      # first 3 PDFs only

Requirements:
    pip install easyocr pillow pdf2image
    brew install poppler   # required by pdf2image on macOS
"""

import argparse
import glob
import os

import easyocr
import numpy as np
from pdf2image import convert_from_path


SAMPLES_DIR  = "samples"
PREVIEWS_DIR = "previews"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=None, help="Max number of PDFs to preview")
    args = parser.parse_args()

    pdf_paths = sorted(glob.glob(os.path.join(SAMPLES_DIR, "*.pdf")))
    if not pdf_paths:
        print(f"No PDF files found in ./{SAMPLES_DIR}/")
        return

    if args.n:
        pdf_paths = pdf_paths[:args.n]

    os.makedirs(PREVIEWS_DIR, exist_ok=True)
    reader = easyocr.Reader(["en"], gpu=False)

    for i, pdf_path in enumerate(pdf_paths):
        name = os.path.splitext(os.path.basename(pdf_path))[0]
        print(f"\n{'=' * 60}")
        print(f"Document {i + 1}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")
        print("=" * 60)

        pages = convert_from_path(pdf_path)
        all_text = []

        for p, page in enumerate(pages):
            img_path = os.path.join(PREVIEWS_DIR, f"{name}_page{p + 1}.png")
            page.save(img_path)
            print(f"  Page {p + 1} image saved: {img_path}  ({page.size[0]}x{page.size[1]} px)")

            image_np = np.array(page)
            results  = reader.readtext(image_np, detail=0, paragraph=True)
            page_text = "\n".join(results)
            all_text.append(page_text)

        full_text = "\n\n".join(all_text)
        preview   = full_text[:500] + ("..." if len(full_text) > 500 else "")
        print(f"\n  OCR text ({len(full_text)} chars):\n    {preview}")

    print(f"\nDone. Page images written to ./{PREVIEWS_DIR}/")


if __name__ == "__main__":
    main()
