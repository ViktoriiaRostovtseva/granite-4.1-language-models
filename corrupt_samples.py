"""
Corrupt moderated PDFs for robustness testing.

Applies two layers of corruption to each PDF in ./samples/moderated/:
  1. Visual noise  — blur, rotation, brightness jitter, salt-and-pepper noise on page images
  2. Text layer    — random char substitutions, deletions, insertions mimicking OCR errors

Corrupted PDFs are saved to ./samples/corrupted/.

Usage:
    python corrupt_samples.py                  # default severity (medium)
    python corrupt_samples.py --severity low
    python corrupt_samples.py --severity high

Requirements:
    pip install pillow pdf2image reportlab numpy
    brew install poppler
"""

import argparse
import glob
import os
import random
import re

import numpy as np
from PIL import Image, ImageFilter
from pdf2image import convert_from_path
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

SOURCE_DIR = "samples/original"
OUTPUT_DIR = "samples/very_corrupted"

# ─── Severity presets ─────────────────────────────────────────────────────────

PRESETS = {
    "low": {
        "blur_radius":       0.5,
        "rotation_deg":      1.0,
        "brightness_delta":  0.05,
        "noise_prob":        0.01,
        "char_sub_prob":     0.02,
        "char_del_prob":     0.01,
        "char_ins_prob":     0.01,
    },
    "medium": {
        "blur_radius":       1.5,
        "rotation_deg":      2.0,
        "brightness_delta":  0.15,
        "noise_prob":        0.04,
        "char_sub_prob":     0.05,
        "char_del_prob":     0.03,
        "char_ins_prob":     0.02,
    },
    "high": {
        "blur_radius":       3.0,
        "rotation_deg":      5.0,
        "brightness_delta":  0.30,
        "noise_prob":        0.10,
        "char_sub_prob":     0.12,
        "char_del_prob":     0.07,
        "char_ins_prob":     0.05,
    },
}

# ─── Visual corruption ────────────────────────────────────────────────────────

def corrupt_image(img: Image.Image, cfg: dict) -> Image.Image:
    """Apply blur, rotation, brightness jitter and salt-and-pepper noise."""
    # Gaussian blur
    img = img.filter(ImageFilter.GaussianBlur(radius=cfg["blur_radius"]))

    # Random rotation (small angle)
    angle = random.uniform(-cfg["rotation_deg"], cfg["rotation_deg"])
    img = img.rotate(angle, expand=False, fillcolor=(255, 255, 255))

    # Brightness jitter
    arr = np.array(img, dtype=np.float32)
    delta = random.uniform(-cfg["brightness_delta"], cfg["brightness_delta"])
    arr = np.clip(arr + delta * 255, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)

    # Salt-and-pepper noise
    arr = np.array(img)
    mask_salt   = np.random.random(arr.shape[:2]) < cfg["noise_prob"] / 2
    mask_pepper = np.random.random(arr.shape[:2]) < cfg["noise_prob"] / 2
    arr[mask_salt]   = 255
    arr[mask_pepper] = 0
    return Image.fromarray(arr)


# ─── Text corruption ──────────────────────────────────────────────────────────

# Common OCR-like substitutions
_OCR_SUBS = {
    "o": "0", "0": "o", "l": "1", "1": "l", "I": "l",
    "s": "5", "S": "5", "g": "9", "B": "8", "e": "c",
    "a": "a", "n": "m", "u": "v", "h": "b", "d": "cl",
}
_NOISE_CHARS = list("abcdefghijklmnopqrstuvwxyz0123456789!@#$%")


def corrupt_text(text: str, cfg: dict) -> str:
    """Apply character-level substitutions, deletions and insertions."""
    result = []
    for ch in text:
        r = random.random()
        if r < cfg["char_del_prob"]:
            continue  # delete
        elif r < cfg["char_del_prob"] + cfg["char_ins_prob"]:
            result.append(random.choice(_NOISE_CHARS))  # insert noise char
            result.append(ch)
        elif r < cfg["char_del_prob"] + cfg["char_ins_prob"] + cfg["char_sub_prob"]:
            result.append(_OCR_SUBS.get(ch, random.choice(_NOISE_CHARS)))  # substitute
        else:
            result.append(ch)
    return "".join(result)


# ─── PDF rebuild ──────────────────────────────────────────────────────────────

def images_to_pdf(images: list, output_path: str):
    """Save a list of PIL images as a PDF using reportlab."""
    c = canvas.Canvas(output_path)
    for img in images:
        w, h = img.size
        c.setPageSize((w, h))
        c.drawImage(ImageReader(img), 0, 0, width=w, height=h)
        c.showPage()
    c.save()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--severity", default="medium", choices=["low", "medium", "high"],
        help="Corruption severity preset"
    )
    args = parser.parse_args()
    cfg = PRESETS[args.severity]

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdf_paths = sorted(glob.glob(os.path.join(SOURCE_DIR, "*.pdf")))
    if not pdf_paths:
        print(f"No PDFs found in {SOURCE_DIR}/")
        return

    print(f"Corrupting {len(pdf_paths)} PDF(s) from {SOURCE_DIR}/ → {OUTPUT_DIR}/  [severity={args.severity}]")

    for pdf_path in pdf_paths:
        name     = os.path.splitext(os.path.basename(pdf_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"{name}_very_corrupted.pdf")
        print(f"  {os.path.basename(pdf_path)} → {os.path.basename(out_path)}")

        pages = convert_from_path(pdf_path, dpi=150)
        corrupted_pages = [corrupt_image(page, cfg) for page in pages]
        images_to_pdf(corrupted_pages, out_path)

    print(f"\nDone. {len(pdf_paths)} corrupted PDFs saved to ./{OUTPUT_DIR}/")
    print("To run the pipeline on them:")
    print("  python main.py --source corrupted")


if __name__ == "__main__":
    main()
