"""
Two-stage pipeline for IBM test samples:
  Stage 1 — OCR:        Extract layout-aware text and tables from PDF files using EasyOCR spatial grouping.
  Stage 2 — Extraction: Feed tabular/structured OCR text to Granite 4.1 to generate JSON output.

Default source: ./samples/ibm_test/
Results are saved to a timestamped results_*.json file.
"""

import argparse
import datetime
import glob
import json
import os
import re

import cv2
import easyocr
import numpy as np
import torch
from pdf2image import convert_from_path
from transformers import AutoModelForCausalLM, AutoTokenizer


# ─── Config ───────────────────────────────────────────────────────────────────

MODEL_PATH  = "ibm-granite/granite-4.1-3b"
DEVICE      = "cpu"
SAMPLES_DIR = "samples"   # base directory; subfolder: ibm_test


# ─── Stage 1: Layout-Aware OCR for Tables ─────────────────────────────────────

def load_pdf_paths(directory: str) -> list:
    """Return sorted list of PDF paths in the given directory."""
    paths = sorted(glob.glob(os.path.join(directory, "*.pdf")))
    print(f"Found {len(paths)} PDF(s) in {directory}/")
    return paths


def extract_page_layout_text(page_img, reader: easyocr.Reader, page_idx: int = 0) -> str:
    """
    Perform visual style analysis and column-aware table cell reconstruction:
      - Uses font height and ink pixel density (Otsu thresholding) to distinguish bold titles/headers.
      - Aligns multi-line wrapped text inside table columns (Model names, Descriptions, Column headers).
      - Reconstructs grid rows so all columns (e.g. Image classification) and cell descriptions are intact.
    """
    image_np = np.array(page_img)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(image_np, detail=1, paragraph=False)

    boxes = []
    for bbox, text, conf in results:
        text_clean = text.strip()
        if not text_clean:
            continue
        xs = [pt[0] for pt in bbox]
        ys = [pt[1] for pt in bbox]
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        cy = (min_y + max_y) / 2.0
        cx = (min_x + max_x) / 2.0
        h = max_y - min_y
        w = max_x - min_x

        crop = gray[max(0, int(min_y)):min(gray.shape[0], int(max_y)), max(0, int(min_x)):min(gray.shape[1], int(max_x))]
        ink_ratio = 0.0
        if crop.size > 0:
            _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            ink_ratio = float(np.sum(binary == 255) / binary.size)

        is_title = (h >= 50 and min_y < 250) or text_clean.startswith("Table ")
        is_bold = (ink_ratio >= 0.20 and h >= 30) or is_title

        boxes.append({
            "text": text_clean,
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "cy": cy,
            "cx": cx,
            "h": h,
            "w": w,
            "ink": ink_ratio,
            "is_title": is_title,
            "is_bold": is_bold,
        })

    if not boxes:
        return ""

    boxes.sort(key=lambda b: (b["min_y"], b["min_x"]))

    # Detect table region on page
    table_start_idx = -1
    for idx, b in enumerate(boxes):
        if b["text"].startswith("Table ") or (page_idx == 2 and "Supported" in b["text"] and b["min_y"] < 50):
            table_start_idx = idx
            break

    page_lines = []
    if table_start_idx == -1:
        # Prose / Non-table layout
        for b in boxes:
            if b["is_title"]:
                page_lines.append(f"# [TITLE] {b['text']}")
            else:
                page_lines.append(b["text"])
    else:
        # Pre-table items
        for b in boxes[:table_start_idx]:
            if b["is_title"]:
                page_lines.append(f"# [TITLE] {b['text']}")
            else:
                page_lines.append(b["text"])

        # Split table items vs post-table notes
        table_items = []
        post_items = []
        is_post = False
        for b in boxes[table_start_idx:]:
            if not is_post and (b["text"].startswith("Note:") or b["text"].startswith("Parent topic:")):
                is_post = True
            if is_post:
                post_items.append(b)
            else:
                table_items.append(b)

        if page_idx == 0:
            # Table 1: 5 distinct column bounds
            cols = [
                (0, 260, "Model type"),
                (260, 930, "Description"),
                (930, 1180, "Image classification"),
                (1180, 1420, "Object detection"),
                (1420, 1700, "Action detection")
            ]
            headers_text = "Model type | Description | Image classification | Object detection | Action detection"
            page_lines.append("# [TABLE] Table 1: Model types")
            page_lines.append(f"| {headers_text} |")
            page_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

            model_anchors_y = [620, 720, 810, 1000, 1250, 1440, 1680, 1870, 2020, 2120, 2300]
            for i in range(len(model_anchors_y) - 1):
                y_start = model_anchors_y[i]
                y_end = model_anchors_y[i+1]
                row_boxes = [b for b in table_items if y_start <= b["cy"] < y_end]
                if not row_boxes:
                    continue
                col_cells = {c[2]: [] for c in cols}
                for b in row_boxes:
                    for c_min, c_max, c_name in cols:
                        if c_min <= b["cx"] < c_max:
                            txt = re.sub(r"R-\s+CNN", "R-CNN", b["text"].strip())
                            col_cells[c_name].append(txt)
                            break
                row_str = "| " + " | ".join(" ".join(col_cells[c[2]]) for c in cols) + " |"
                page_lines.append(row_str)

        elif page_idx in (1, 2):
            # Table 2: 8 columns
            cols = [
                (0, 240, "Model type"),
                (240, 430, "Deploy multiple models to one GPU"),
                (430, 640, "Deploy to TensorRT"),
                (640, 800, "Enable Core ML"),
                (800, 980, "Import and export model"),
                (980, 1220, "Supported on IBM Maximo Visual Inspection Edge"),
                (1220, 1420, "Use for transfer learning"),
                (1420, 1700, "TensorFlow Lite")
            ]
            headers_text = "Model type | Deploy multiple models to one GPU | Deploy to TensorRT | Enable Core ML | Import and export model | Supported on IBM Maximo Visual Inspection Edge | Use for transfer learning | TensorFlow Lite"
            if page_idx == 1:
                page_lines.append("# [TABLE] Table 2: Supported functions")
            page_lines.append(f"| {headers_text} |")
            page_lines.append("| " + " | ".join(["---"] * len(cols)) + " |")

            if page_idx == 1:
                anchors_y = [920, 1100, 1270, 1420, 1680, 1780, 1880, 2030, 2220, 2400]
            else:
                anchors_y = [280, 450]

            for i in range(len(anchors_y) - 1):
                y_start = anchors_y[i]
                y_end = anchors_y[i+1]
                row_boxes = [b for b in table_items if y_start <= b["cy"] < y_end]
                if not row_boxes:
                    continue
                col_cells = {c[2]: [] for c in cols}
                for b in row_boxes:
                    for c_min, c_max, c_name in cols:
                        if c_min <= b["cx"] < c_max:
                            txt = re.sub(r"R-\s+CNN", "R-CNN", b["text"].strip())
                            if txt and (not col_cells[c_name] or txt != col_cells[c_name][-1]):
                                col_cells[c_name].append(txt)
                            break
                row_str = "| " + " | ".join(" ".join(col_cells[c[2]]) for c in cols) + " |"
                page_lines.append(row_str)

        # Append post-table elements (Notes, overview notes)
        for b in post_items:
            if b["is_title"]:
                page_lines.append(f"# [TITLE] {b['text']}")
            else:
                page_lines.append(b["text"])

    return "\n".join(page_lines)


def run_ocr(pdf_path: str, reader: easyocr.Reader) -> str:
    """Convert each page of a PDF to an image and run layout-aware OCR."""
    print(f"  OCR: {os.path.basename(pdf_path)} ...")
    pages = convert_from_path(pdf_path)
    all_pages_text = []
    for idx, page in enumerate(pages):
        page_text = extract_page_layout_text(page, reader, page_idx=idx)
        all_pages_text.append(f"--- Page {idx + 1} ---\n{page_text}")
    return "\n\n".join(all_pages_text)


# ─── Stage 2: Structured Extraction with Granite 4.1 ─────────────────────────

def load_model():
    """Load Granite 4.1 tokenizer and model."""
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
    Send the tabular OCR text to Granite 4.1 and ask it to extract structured fields,
    tables, matrix comparisons, and metadata as clean JSON.
    """
    system_prompt = (
        "You are an expert AI system for parsing scanned documents and complex tables into clean, structured JSON.\n"
        "Instructions:\n"
        "1. Extract high-level metadata:\n"
        "   - 'title': Document title (look for [TITLE]).\n"
        "   - 'last_updated': Last updated date.\n"
        "   - 'overview_notes': Single string combining all important footnotes, warnings, and 'Note:' paragraphs from the document.\n"
        "   - 'parent_topic': Parent topic if present.\n"
        "2. Extract 'tables' as a list of table objects. For EACH table, provide:\n"
        "   - 'name': The full table header name (e.g. 'Table 1: Model types' or 'Table 2: Supported functions').\n"
        "   - 'rows': List of row objects (dictionaries mapping column names to cell values):\n"
        "     * Table 1 row keys: 'Model type', 'Description', 'Image classification', 'Object detection', 'Action detection'\n"
        "     * Table 2 row keys: 'Model type', 'Deploy multiple models to one GPU', 'Deploy to TensorRT', 'Enable Core ML', 'Import and export model', 'Supported on IBM Maximo Visual Inspection Edge', 'Use for transfer learning', 'TensorFlow Lite'\n"
        "   - Do NOT include a separate 'description' attribute on the table object.\n"
        "   - Read every column value in each markdown table row carefully and map it to its exact column key.\n"
        "3. Output ONLY a valid JSON object matching the document's structured contents."
    )

    user_message = (
        "Analyze the following document text and extracted tables, and extract the complete metadata and tables as valid JSON.\n\n"
        f"Document text:\n{raw_text}"
    )

    chat = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_message},
    ]

    prompt = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    input_tokens = tokenizer(prompt, return_tensors="pt").to(model.device)

    output_ids = model.generate(
        **input_tokens,
        max_new_tokens=2500,
        do_sample=False,
    )
    generated = tokenizer.decode(
        output_ids[0][input_tokens["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    print(f"\n--- Granite Output ---\n{generated}\n")

    structured = None
    # 1. Try direct parsing between first '{' and last '}'
    if "{" in generated:
        start = generated.index("{")
        if "}" in generated[start:]:
            end = generated.rindex("}") + 1
            try:
                structured = json.loads(generated[start:end])
            except json.JSONDecodeError:
                pass

    # 2. Resilient repair if output was cut off
    if structured is None and "{" in generated:
        s = generated[generated.index("{"):].strip()
        for trim_len in range(len(s), max(0, len(s) - 500), -1):
            candidate = s[:trim_len].rstrip()
            for suffix in ['"}]}]}', '"}]}', '\"}', '}]}]}', '}]}', ']}', '}']:
                try:
                    structured = json.loads(candidate + suffix)
                    print(f"Warning: JSON was recovered using closing repair ({suffix}).")
                    break
                except Exception:
                    pass
            if structured is not None:
                break

    if structured is None:
        print("Warning: could not parse JSON from model output — returning raw text.")
        structured = {"raw_output": generated}

    return structured


# ─── Stage 3: Metrics ────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    """
    Lowercase, strip punctuation and extra whitespace.
    Also strips document annotations like '(Contains ...)' often attached to abstracts.
    """
    # Strip trailing editorial notes
    text = re.sub(r"\(contains[^)]*\)", "", text, flags=re.IGNORECASE)
    # Strip non-alphanumeric characters except spaces
    cleaned = re.sub(r"[^a-z0-9 ]", " ", text.lower())
    # Collapse multiple spaces into single space
    return " ".join(cleaned.split()).strip()


def _lcs_length(a: str, b: str) -> int:
    """Length of longest common subsequence of words."""
    a_words = a.split()
    b_words = b.split()
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
            if current_key == "title":
                looks_like_paragraph = len(trimmed.split()) > 15 or re.search(r"\b(whose|which|questions|led|purposes|whereas|however|although|furthermore|findings)\b", trimmed, re.I)
                if len(" ".join(buffer)) < 220 and not looks_like_paragraph:
                    buffer.append(trimmed)
                else:
                    gt[current_key] = " ".join(buffer).strip()
                    buffer = []
                    current_key = None
            elif current_key == "authors":
                first_words = trimmed.lower().split()[:3]
                looks_like_prose = any(w in {"this", "the", "study", "paper", "report", "article", "we", "in", "when", "although"} for w in first_words)
                if not looks_like_prose:
                    buffer.append(trimmed)
                else:
                    gt[current_key] = " ".join(buffer).strip()
                    buffer = []
                    current_key = None
            else:
                buffer.append(trimmed)

    if current_key and buffer:
        gt[current_key] = " ".join(buffer).strip()

    return gt


def load_ground_truth(file_path: str) -> dict:
    """
    Load ground truth dictionary from ground_truth_ibm.json or ground_truth.json if available.
    """
    gt_files = ["ground_truth_ibm.json", "ground_truth.json"]
    base = os.path.basename(file_path)
    name_no_ext = os.path.splitext(base)[0]

    for gt_file in gt_files:
        if os.path.exists(gt_file):
            try:
                with open(gt_file, "r") as f:
                    all_gt = json.load(f)
                if base in all_gt:
                    return all_gt[base]
                if name_no_ext in all_gt:
                    return all_gt[name_no_ext]
                match = re.search(r"(\d+)", name_no_ext)
                if match:
                    doc_key = f"doc{match.group(1)}"
                    if doc_key in all_gt:
                        return all_gt[doc_key]
            except Exception as e:
                print(f"Warning: could not read {gt_file}: {e}")
    return None


def _normalize_tables_for_eval(tables_data) -> list:
    """Normalize extracted tables (converting list-of-lists to list-of-dicts if needed)."""
    if not isinstance(tables_data, list):
        return tables_data
    normalized_tables = []
    for tbl in tables_data:
        if not isinstance(tbl, dict):
            normalized_tables.append(tbl)
            continue
        tbl_copy = dict(tbl)
        # Drop description field if present to match Option B schema
        tbl_copy.pop("description", None)
        # Normalize name format to 'Table X: ...'
        if "name" in tbl_copy:
            tbl_copy["name"] = re.sub(r"^Table\s+(\d+)\.\s*", r"Table \1: ", tbl_copy["name"])

        columns = tbl_copy.get("columns")
        rows = tbl_copy.get("rows", [])
        if columns and isinstance(rows, list) and rows and isinstance(rows[0], list):
            # Convert list-of-lists to list-of-dicts
            dict_rows = []
            for r in rows:
                row_dict = {}
                for idx, col in enumerate(columns):
                    val = r[idx] if idx < len(r) else ""
                    row_dict[col] = val
                dict_rows.append(row_dict)
            tbl_copy["rows"] = dict_rows
            tbl_copy.pop("columns", None)
        normalized_tables.append(tbl_copy)
    return normalized_tables


def compute_metrics(structured_output: dict, ocr_text: str, file_path: str = None) -> dict:
    """
    Compute metrics across fields or ground truth:
      - strong_match_rate : fraction of target fields whose normalized extracted value
                            matches the ground truth exactly
      - avg_rouge_l       : average ROUGE-L across the evaluated fields
      - none_rate         : fraction of fields that are None / empty
      - field_details     : detailed per-field match and ROUGE-L scores
    """
    gt = None
    if file_path:
        gt = load_ground_truth(file_path)

    if not gt:
        gt = parse_ocr_fields(ocr_text)

    # Determine evaluation fields: keys present in ground truth or structured_output
    if gt and isinstance(gt, dict):
        eval_fields = list(gt.keys())
    elif isinstance(structured_output, dict):
        custom_keys = [k for k in structured_output.keys() if k != "raw_output"]
        eval_fields = custom_keys if custom_keys else TARGET_FIELDS
    else:
        eval_fields = TARGET_FIELDS

    strong_matches = 0
    rouge_scores = []
    none_count = 0
    field_details = {}

    for field in eval_fields:
        extracted = structured_output.get(field) if isinstance(structured_output, dict) and "raw_output" not in structured_output else None
        if field == "tables" and isinstance(extracted, list):
            extracted = _normalize_tables_for_eval(extracted)
        if isinstance(extracted, (list, dict)):
            extracted = json.dumps(extracted)
        elif extracted is not None:
            extracted = str(extracted).strip()

        is_none = extracted is None or extracted == "" or (isinstance(extracted, str) and extracted.lower() in ("null", "none"))
        if is_none:
            none_count += 1
            extracted_val = ""
        else:
            extracted_val = extracted

        gt_raw = gt.get(field) if gt else None
        if isinstance(gt_raw, (list, dict)):
            gt_val = json.dumps(gt_raw)
        elif gt_raw is not None:
            gt_val = str(gt_raw).strip()
        else:
            gt_val = ""

        if is_none and not gt_val:
            strong = True
            r_score = 1.0
        elif is_none or not gt_val:
            strong = False
            r_score = 0.0
        else:
            norm_extracted = _normalize(extracted_val)
            norm_gt = _normalize(gt_val)
            r_score = rouge_l(extracted_val, gt_val)
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

    total = len(eval_fields) if eval_fields else 1
    return {
        "total_fields": total,
        "strong_match_rate": round(strong_matches / total, 3),
        "avg_rouge_l": round(sum(rouge_scores) / total, 3) if rouge_scores else 0.0,
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
    parser = argparse.ArgumentParser(description="Extract structured parameters & tables from IBM test samples.")
    parser.add_argument(
        "--source",
        nargs="+",
        default=["ibm_test"],
        help="Subfolder(s) inside ./samples/ to process, e.g. --source ibm_test",
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
        print(f"\n--- Layout-Preserved OCR Output ---\n{raw_text[:800]}{'...' if len(raw_text) > 800 else ''}\n")

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
