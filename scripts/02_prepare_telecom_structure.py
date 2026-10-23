import os
import json
import re
from pathlib import Path
from PyPDF2 import PdfReader


# -------- CONFIG --------
ROOT_DIR = Path(__file__).resolve().parent.parent
BASE_DIR = ROOT_DIR / "data" / "pdf"
OUTPUT_FILE = ROOT_DIR / "data" / "telecom_structure_corpus.jsonl"

SECTOR = "Energy"
ORGANIZATION = "NERC"
REGULATION_FAMILY = "NERC CIP"
DOCUMENT_TYPE = "Standard"

CHUNK_SIZE = 900
CHUNK_OVERLAP = 100
MIN_CHUNK_SIZE = 600
# ------------------------


def clean_text(text):
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()




def chunk_text(text, chunk_size=CHUNK_SIZE):
    def is_heading(line):
        return (
            len(line) <= 90
            and len(line.split()) <= 12
            and not line.endswith((".", "?", "!"))
            and (
                line.isupper()
                or bool(re.match(
                    r'^\d+(?:\.\d+)*[.)]?\s+[A-Za-z]',
                    line
                ))
            )
        )

    blocks = []
    paragraph = []

    for raw_line in text.splitlines():
        line = raw_line.strip()

        if not line or is_heading(line):
            if paragraph:
                blocks.append(" ".join(paragraph))
                paragraph = []

            if line and is_heading(line):
                blocks.append(line)
        else:
            paragraph.append(line)

    if paragraph:
        blocks.append(" ".join(paragraph))

    structured_text = "\n\n".join(blocks)

    chunks = []
    start = 0

    while start < len(structured_text):
        limit = min(start + chunk_size, len(structured_text))
        end = limit

        if limit < len(structured_text):
            minimum = min(start + MIN_CHUNK_SIZE, limit)

            boundary = structured_text.rfind(
                "\n\n", minimum, limit
            )

            if boundary != -1:
                end = boundary
            else:
                boundary = structured_text.rfind(
                    " ", minimum, limit
                )
                if boundary != -1:
                    end = boundary

        chunk = structured_text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= len(structured_text):
            break

        next_start = max(start + 1, end - CHUNK_OVERLAP)

        match = re.search(
            r'\s+', structured_text[next_start:end]
        )

        if match:
            next_start += match.end()

        start = next_start

    return chunks



def extract_keywords(text):
    words = re.findall(r'\b[a-zA-Z]{6,}\b', text.lower())
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:5]]


def process_pdf(pdf_path, chunk_id_start):
    reader = PdfReader(pdf_path)
    pdf_name = os.path.basename(pdf_path)
    title_guess = pdf_name.replace(".pdf", "").replace("_", " ")

    entries = []
    chunk_counter = chunk_id_start

    for page_num, page in enumerate(reader.pages, start=1):
        raw_text = page.extract_text() or ""
        cleaned = clean_text(raw_text)

        if not cleaned:
            continue

        chunks = [
            chunk
            for chunk in chunk_text(cleaned, CHUNK_SIZE)
            if not (
                len(chunk.strip()) < 20
                and not re.search(r'[A-Za-z]{3,}', chunk)
            )
        ]

        for idx, chunk in enumerate(chunks):
            entry = {
                "chunk_id": f"energy-{chunk_counter:06d}",
                "source_file": pdf_name,
                "document_title": title_guess,
                "organization": ORGANIZATION,
                "sector": SECTOR,
                "document_type": DOCUMENT_TYPE,
                "regulation_family": REGULATION_FAMILY,
                "topic": title_guess,
                "page_number": page_num,
                "chunk_index": idx,
                "text": chunk,
                "keywords": extract_keywords(chunk)
            }
            entries.append(entry)
            chunk_counter += 1

    return entries, chunk_counter


def main():
    pdf_files = sorted(BASE_DIR.glob("*.pdf"))
    all_entries = []
    chunk_id_counter = 1

    for pdf in pdf_files:
        print(f"Processing: {pdf.name}")
        entries, chunk_id_counter = process_pdf(pdf, chunk_id_counter)
        all_entries.extend(entries)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for entry in all_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\nDone! JSONL corpus saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()