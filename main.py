"""
Two-stage pipeline:
  Stage 1 — OCR:        Extract raw text from PDF files using EasyOCR.
  Stage 2 — Extraction: Feed raw text to Granite 4.1 and get structured JSON parameters back.

Place your PDF files in the ./samples/ directory.
Results are saved to a timestamped results_*.json file.

Requirements:
    pip install torch transformers easyocr pillow pdf2image accelerate
    brew install poppler   # required by pdf2image on macOS
"""

import argparse
import datetime
import glob
import json
import os
import re

import easyocr
import numpy as np
import torch
from pdf2image import convert_from_path
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_PATH  = "ibm-granite/granite-4.1-3b"
DEVICE      = "cpu"
SAMPLES_DIR = "samples"   # base directory; subfolders: original/, moderated/



# ─── Stage 1: OCR ─────────────────────────────────────────────────────────────

def load_pdf_paths(directory: str) -> list:
    """Return sorted list of PDF paths in the given directory."""
    paths = sorted(glob.glob(os.path.join(directory, "*.pdf")))
    print(f"Found {len(paths)} PDF(s) in {directory}/")
    return paths


def run_ocr(pdf_path: str, reader: easyocr.Reader) -> str:
    """Convert each page of a PDF to an image and run EasyOCR on it."""
    print(f"  OCR: {os.path.basename(pdf_path)} ...")
    pages = convert_from_path(pdf_path)
    all_text = []
    for page in pages:
        image_np = np.array(page)
        results = reader.readtext(image_np, detail=0, paragraph=True)
        all_text.append("\n".join(results))
    return "\n\n".join(all_text)


# ─── Stage 2: Structured Extraction with Granite 4.1 ─────────────────────────

def load_model():
    """Load Granite 4.1 tokenizer and model in float16 (half memory vs float32)."""
    print(f"Loading model: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        dtype=torch.float32,  # float16 is not stable on macOS CPU
    )
    model = model.to("cpu")
    model.eval()
    return tokenizer, model


def extract_parameters(raw_text: str, tokenizer, model) -> dict:
    """
    Send the OCR text to Granite 4.1 and ask it to extract structured fields
    as JSON, following the JSON-output pattern from the README.
    """
    # --- Commented out: 6 specific fixed fields ---
    # system_prompt = (
    #     "You are a helpful assistant that extracts structured metadata from "
    #     "scanned document text and returns it as JSON.\n"
    #     "Extract the following 6 fields if they exist in the text. If a field is not found or empty, set its value to null.\n"
    #     "1. title: Title of the document\n"
    #     "2. pages: Number of pages (e.g. '38', '14')\n"
    #     "3. authors: Semicolon or colon-separated list of authors\n"
    #     "4. place: Publishing or institutional place/organization\n"
    #     "5. keywords: Comma or semicolon-separated keywords/key-words\n"
    #     "6. abstract: Summary or abstract text (Abst)\n\n"
    #     "Here is the JSON schema you must follow:\n"
    #     "<schema>\n"
    #     "{\n"
    #     '  "title": {"type": ["string", "null"]},\n'
    #     '  "pages": {"type": ["string", "null"]},\n'
    #     '  "authors": {"type": ["string", "null"]},\n'
    #     '  "place": {"type": ["string", "null"]},\n'
    #     '  "keywords": {"type": ["string", "null"]},\n'
    #     '  "abstract": {"type": ["string", "null"]}\n'
    #     "}\n"
    #     "</schema>"
    # )
    #
    # user_message = (
    #     "Extract the 6 metadata fields (title, pages, authors, place, keywords, abstract) from this document text.\n\n"
    #     f"Document text:\n{raw_text}"
    # )

    # General structured extraction prompt for arbitrary document types
    system_prompt = (
        "You are a helpful assistant that extracts structured key information and metadata from "
        "scanned document text and returns it as a clean, well-structured JSON object."
    )

    user_message = (
        "Analyze the following document text and extract all relevant structured key-value pairs, "
        "sections, or metadata as a valid JSON object.\n\n"
        f"Document text:\n{raw_text}"
    )

    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(**input_tokens, max_new_tokens=1024)
    generated = tokenizer.decode(
        output_ids[0][input_tokens["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print(f"\n--- Granite Output ---\n{generated}\n")

    try:
        start = generated.index("{")
        end   = generated.rindex("}") + 1
        structured = json.loads(generated[start:end])
    except (ValueError, json.JSONDecodeError):
        # Try to recover a partial JSON — extract whatever fields were completed
        try:
            partial = generated[generated.index("{"):]
            # Close any open array and object to make it parseable
            partial = partial.rstrip().rstrip(",") + "]}"
            structured = json.loads(partial)
            print("Warning: JSON was truncated — recovered partial output.")
        except (ValueError, json.JSONDecodeError):
            print("Warning: could not parse JSON from model output — returning raw text.")
            structured = {"raw_output": generated}

    return structured


# ─── Stage 3: Metrics ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    Lowercase, strip punctuation and extra whitespace.
    Also strips document annotations like '(Contains ...)' often attached to abstracts.
    """
    import re
    # Strip trailing editorial notes like (Contains 20 online resources:), (Contains 6 footnotes...), etc.
    text = re.sub(r"\(contains[^)]*\)", "", text, flags=re.IGNORECASE)
    # Strip non-alphanumeric characters except spaces
    cleaned = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    # Collapse multiple spaces into single space
    return " ".join(cleaned.split()).strip()


def _lcs_length(a: str, b: str) -> int:
    """Length of longest common subsequence of words."""
    a_words = a.split()
    b_words = b.split()
    # DP table — O(n*m) but inputs are short strings
    dp = [[0] * (len(b_words) + 1) for _ in range(len(a_words) + 1)]
    for i, wa in enumerate(a_words, 1):
        for j, wb in enumerate(b_words, 1):
            dp[i][j] = dp[i-1][j-1] + 1 if wa == wb else max(dp[i-1][j], dp[i][j-1])
    return dp[len(a_words)][len(b_words)]


def rouge_l(prediction: str, reference: str) -> float:
    """ROUGE-L F1 score between two strings (word-level LCS)."""
    pred = _normalize(prediction)
    ref  = _normalize(reference)
    if not pred or not ref:
        return 0.0
    lcs   = _lcs_length(pred, ref)
    prec  = lcs / len(pred.split())
    rec   = lcs / len(ref.split())
    if prec + rec == 0:
        return 0.0
    return round(2 * prec * rec / (prec + rec), 3)


TARGET_FIELDS = ["title", "pages", "authors", "place", "keywords", "abstract"]

# Mapping from canonical field names to possible label variations in OCR text
_LABEL_ALIASES = {
    "title":    ["title"],
    "pages":    ["pages", "page", "p"],
    "authors":  ["authors", "author"],
    "place":    ["place", "institution", "organization", "journal"],
    "keywords": ["keywords", "key-words", "key words", "keyword"],
    "abstract": ["abst", "abstract", "summary"],
}


def parse_ocr_fields(ocr_text: str) -> dict:
    """
    Parse OCR text into canonical fields: title, pages, authors, place, keywords, abstract.
    """
    import re
    gt = {k: None for k in TARGET_FIELDS}

    current_key = None
    buffer = []

    for line in ocr_text.splitlines():
        trimmed = line.strip()
        if not trimmed:
            continue

        matched_field = None
        matched_val = None
        m = re.match(r"^([A-Za-z][A-Za-z\- ]{0,30}):\s*(.*)$", trimmed)
        if m:
            header = _normalize(m.group(1))
            val = m.group(2).strip()
            for field, aliases in _LABEL_ALIASES.items():
                if header in aliases:
                    matched_field = field
                    matched_val = val
                    break

        if matched_field:
            if current_key and buffer:
                gt[current_key] = " ".join(buffer).strip()
                buffer = []
            current_key = matched_field
            if matched_val:
                buffer.append(matched_val)
            if current_key == "pages":
                gt[current_key] = " ".join(buffer).strip()
                buffer = []
                current_key = None
        elif current_key:
            # Multi-line continuation rules:
            # In moderated documents, fields can wrap across lines.
            if current_key == "title":
                # Title wraps if the prior line doesn't end with sentence-ending punctuation
                # or if the whole title is relatively short (< 200 chars) and doesn't look like an abstract paragraph.
                prev_line = buffer[-1] if buffer else ""
                looks_like_paragraph = len(trimmed.split()) > 15 or re.search(r"\b(whose|which|questions|led|purposes|whereas|however|although|furthermore|findings)\b", trimmed, re.I)
                if len(" ".join(buffer)) < 220 and not looks_like_paragraph:
                    buffer.append(trimmed)
                else:
                    gt[current_key] = " ".join(buffer).strip()
                    buffer = []
                    current_key = None
            elif current_key == "authors":
                # Authors can wrap across multiple lines (e.g. separated by commas/semicolons)
                first_words = trimmed.lower().split()[:3]
                looks_like_prose = any(w in {"this", "the", "study", "paper", "report", "article", "we", "in", "when", "although"} for w in first_words)
                if not looks_like_prose:
                    buffer.append(trimmed)
                else:
                    gt[current_key] = " ".join(buffer).strip()
                    buffer = []
                    current_key = None
            else:
                # Place, Keywords, Abstract continue wrapping across lines
                buffer.append(trimmed)

    if current_key and buffer:
        gt[current_key] = " ".join(buffer).strip()

    return gt


def load_ground_truth(file_path: str) -> dict:
    """
    Load ground truth dictionary from ground_truth.json if available.
    Matches keys by basename or prefix (e.g. 'mod1.pdf' -> 'doc1', 'cor10.pdf' -> 'doc10').
    """
    gt_file = "ground_truth.json"
    if os.path.exists(gt_file):
        try:
            with open(gt_file, "r") as f:
                all_gt = json.load(f)
            base = os.path.basename(file_path)
            # 1. Exact match (e.g. "doc1.pdf")
            if base in all_gt:
                return all_gt[base]
            # 2. Key without extension (e.g. "doc1")
            name_no_ext = os.path.splitext(base)[0]
            if name_no_ext in all_gt:
                return all_gt[name_no_ext]
            # 3. Match normalized prefix (e.g. "mod1" -> "doc1", "cor10" -> "doc10")
            match = re.search(r"(\d+)", name_no_ext)
            if match:
                doc_key = f"doc{match.group(1)}"
                if doc_key in all_gt:
                    return all_gt[doc_key]
        except Exception as e:
            print(f"Warning: could not read ground_truth.json: {e}")
    return None


def compute_metrics(structured_output: dict, ocr_text: str, file_path: str = None) -> dict:
    """
    Compute metrics across the 6 target fields:
      - strong_match_rate : fraction of target fields whose normalized extracted value
                            matches the ground truth exactly
      - avg_rouge_l       : average ROUGE-L across the 6 target fields
      - none_rate         : fraction of target fields that are None / empty
      - field_details     : detailed per-field match and ROUGE-L scores
    """
    if "raw_output" in structured_output and not any(k in structured_output for k in TARGET_FIELDS):
        return {
            "total_fields": len(TARGET_FIELDS),
            "strong_match_rate": 0.0,
            "avg_rouge_l": 0.0,
            "none_rate": 1.0,
            "field_details": {k: {"extracted": None, "ground_truth": None, "strong_match": False, "rouge_l": 0.0} for k in TARGET_FIELDS},
        }

    gt = None
    if file_path:
        gt = load_ground_truth(file_path)

    if not gt:
        gt = parse_ocr_fields(ocr_text)

    strong_matches = 0
    rouge_scores = []
    none_count = 0
    field_details = {}

    for field in TARGET_FIELDS:
        extracted = structured_output.get(field)
        if isinstance(extracted, list):
            extracted = "; ".join(str(x) for x in extracted)
        elif extracted is not None:
            extracted = str(extracted).strip()

        # Check for None / empty
        is_none = extracted is None or extracted == "" or extracted.lower() in ("null", "none")
        if is_none:
            none_count += 1
            extracted_val = ""
        else:
            extracted_val = extracted

        gt_val = gt.get(field) or ""

        # Score computation
        if is_none and not gt_val:
            # Correctly identified as not present in text
            strong = True
            r_score = 1.0
        elif is_none or not gt_val:
            strong = False
            r_score = 0.0
        else:
            norm_extracted = _normalize(extracted_val)
            norm_gt = _normalize(gt_val)
            r_score = rouge_l(extracted_val, gt_val)
            # Strong match: exact normalized match, or substring inclusion for long fields, or high ROUGE-L overlap (>= 0.95)
            strong = (
                norm_extracted == norm_gt
                or (len(norm_gt) > 50 and (norm_extracted in norm_gt or norm_gt in norm_extracted))
                or r_score >= 0.95
            )

        if strong:
            strong_matches += 1
        rouge_scores.append(r_score)

        field_details[field] = {
            "extracted": None if is_none else extracted_val,
            "ground_truth": gt_val if gt_val else None,
            "strong_match": strong,
            "rouge_l": r_score,
        }

    total = len(TARGET_FIELDS)
    return {
        "total_fields": total,
        "strong_match_rate": round(strong_matches / total, 3),
        "avg_rouge_l": round(sum(rouge_scores) / total, 3),
        "none_rate": round(none_count / total, 3),
        "field_details": field_details,
    }


# ─── Save results ─────────────────────────────────────────────────────────────

def save_results(data: dict) -> str:
    """Persist results to a timestamped JSON file and return the filename."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = f"results_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {filename}")
    return filename


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        nargs="+",
        default=["original"],
        help="Subfolder(s) inside ./samples/ to process, e.g. --source moderated or --source original moderated",
    )
    parser.add_argument(
        "--no-eval",
        "--no-ground-truth",
        action="store_true",
        dest="no_eval",
        help="Skip ground truth comparison and metric evaluation (extraction only)",
    )
    args = parser.parse_args()

    pdf_paths = []
    for subfolder in args.source:
        folder = os.path.join(SAMPLES_DIR, subfolder)
        paths  = sorted(glob.glob(os.path.join(folder, "*.pdf")))
        print(f"Found {len(paths)} PDF(s) in {folder}/")
        pdf_paths.extend(paths)

    if not pdf_paths:
        print(f"No PDF files found — check your --source folder(s).")
        return

    # Load OCR reader once, then delete it before loading the model to free RAM
    print("Loading OCR reader ...")
    reader = easyocr.Reader(["en"], gpu=False)

    ocr_results = {}
    for i, pdf_path in enumerate(pdf_paths):
        print(f"\n{'=' * 60}")
        print(f"OCR {i + 1}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")
        print("=" * 60)
        raw_text = run_ocr(pdf_path, reader)
        ocr_results[pdf_path] = raw_text
        print(f"\n--- OCR Output ---\n{raw_text[:500]}{'...' if len(raw_text) > 500 else ''}\n")

    # Free OCR reader memory before loading the model
    del reader

    tokenizer, model = load_model()

    results = []
    for i, pdf_path in enumerate(pdf_paths):
        print(f"\n{'=' * 60}")
        print(f"Extracting {i + 1}/{len(pdf_paths)}: {os.path.basename(pdf_path)}")
        print("=" * 60)

        raw_text = ocr_results[pdf_path]
        if not raw_text.strip():
            print("  No text extracted — skipping.")
            results.append({"file": os.path.basename(pdf_path), "error": "no text extracted"})
            continue

        structured_output = extract_parameters(raw_text, tokenizer, model)
        print("--- Structured Parameters ---")
        print(json.dumps(structured_output, indent=2))

        doc_entry = {
            "file":              os.path.basename(pdf_path),
            "ocr_text":          raw_text,
            "structured_output": structured_output,
        }

        if not args.no_eval:
            metrics = compute_metrics(structured_output, raw_text, file_path=pdf_path)
            print(f"--- Metrics ---")
            print(f"  Strong match rate : {metrics['strong_match_rate']}")
            print(f"  Avg ROUGE-L       : {metrics['avg_rouge_l']}")
            print(f"  None rate         : {metrics['none_rate']}")
            doc_entry["metrics"] = metrics

        results.append(doc_entry)

    # Aggregate summary across all docs (if evaluated)
    payload = {"documents": results}
    if not args.no_eval:
        valid = [r for r in results if "metrics" in r]
        if valid:
            summary = {
                "documents_evaluated": len(valid),
                "avg_strong_match_rate": round(sum(r["metrics"]["strong_match_rate"] for r in valid) / len(valid), 3),
                "avg_rouge_l":           round(sum(r["metrics"]["avg_rouge_l"]        for r in valid) / len(valid), 3),
                "avg_none_rate":         round(sum(r["metrics"]["none_rate"]           for r in valid) / len(valid), 3),
            }
            print(f"\n{'=' * 60}")
            print("SUMMARY")
            print(f"{'=' * 60}")
            print(f"  Documents evaluated  : {summary['documents_evaluated']}")
            print(f"  Avg strong match rate: {summary['avg_strong_match_rate']}")
            print(f"  Avg ROUGE-L          : {summary['avg_rouge_l']}")
            print(f"  Avg None rate        : {summary['avg_none_rate']}")
            payload["summary"] = summary

    save_results(payload)


if __name__ == "__main__":
    main()
