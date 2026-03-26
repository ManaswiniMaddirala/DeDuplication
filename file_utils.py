"""
file_utils.py — Universal text extraction
Extracts plain text from ANY supported format so that a .docx and a .pdf
with identical content produce matching token sets for MinHash comparison.

Root cause fix (2025-02):
  python-docx's p.text concatenates runs at the Python level but Word XML
  splits words across revision-mark runs e.g. "Maddir" + "ala" -> "Maddirala"
  when joined without spaces. The standard approach adds a space between runs
  which gives "Maddir ala" -- a different token from the PDF's "Maddirala".
  Fix: join all <w:t> runs within a paragraph WITHOUT spaces, then deduplicate
  whitespace. This gives correct whole-word tokens that match the PDF.
"""

import os
import re
import zipfile


def _extract_docx(file_path: str) -> str:
    """
    Extract text from .docx by joining XML text runs per-paragraph
    without inserting spaces between them.

    Word stores text in <w:t> elements inside <w:r> (run) elements.
    When a word spans multiple runs (due to spell-check, revision marks,
    formatting changes etc.) each run contains a fragment:
        <w:t>Maddir</w:t>  <w:t>ala</w:t>
    Joining with spaces gives "Maddir ala" (wrong token).
    Joining without spaces gives "Maddirala" (matches the PDF token).
    """
    try:
        with zipfile.ZipFile(file_path, 'r') as z:
            with z.open('word/document.xml') as f:
                xml = f.read().decode('utf-8', errors='ignore')

        # Split into paragraphs
        paragraphs = re.split(r'<w:p[ /]', xml)

        lines = []
        for para in paragraphs:
            # Extract all <w:t> content, join WITHOUT spaces between runs
            runs = re.findall(r'<w:t(?:\s[^>]*)?>([^<]*)</w:t>', para)
            text = ''.join(runs)
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 1:
                lines.append(text)

        return '\n'.join(lines)

    except Exception:
        # Fallback: python-docx run-level join
        try:
            import docx
            doc = docx.Document(file_path)
            lines = []
            for para in doc.paragraphs:
                text = ''.join(run.text for run in para.runs)
                text = re.sub(r'\s+', ' ', text).strip()
                if text:
                    lines.append(text)
            return '\n'.join(lines)
        except Exception:
            return ""


def _extract_pdf(file_path: str) -> str:
    text = ""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(file_path)
        for page in reader.pages:
            t = page.extract_text()
            if t:
                text += t + "\n"
        if text.strip():
            return text
    except Exception:
        pass
    try:
        import fitz
        doc = fitz.open(file_path)
        for page in doc:
            text += page.get_text() + "\n"
        return text
    except Exception:
        return ""


def _extract_txt(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _extract_csv(file_path: str) -> str:
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def _extract_with_textract(file_path: str) -> str:
    try:
        import textract
        raw = textract.process(file_path)
        return raw.decode("utf-8", errors="ignore")
    except Exception:
        return ""


_FALLBACKS = {
    "txt":  _extract_txt,
    "md":   _extract_txt,
    "py":   _extract_txt,
    "html": _extract_txt,
    "css":  _extract_txt,
    "js":   _extract_txt,
    "pdf":  _extract_pdf,
    "docx": _extract_docx,
    "doc":  _extract_docx,
    "csv":  _extract_csv,
    "json": _extract_txt,
    "xml":  _extract_txt,
}


def get_any_text_content(file_path: str) -> str:
    """
    Extract text from ANY supported file format.

    For DOCX/DOC: always use fixed XML run-joining extractor first.
    Textract for DOCX internally uses python-docx which has the same
    split-word problem. We bypass it entirely.

    For all other formats: try textract first, then format-specific fallback.
    """
    if not os.path.exists(file_path):
        return ""

    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""

    # DOCX/DOC: fixed extractor always first
    if ext in ("docx", "doc"):
        text = _extract_docx(file_path)
        if text.strip():
            return text
        return _extract_with_textract(file_path)

    # All other formats: textract then fallback
    text = _extract_with_textract(file_path)
    if not text.strip():
        fallback_fn = _FALLBACKS.get(ext)
        if fallback_fn:
            text = fallback_fn(file_path)

    return text


def is_text_extractable(file_path: str) -> bool:
    ext = file_path.rsplit(".", 1)[-1].lower() if "." in file_path else ""
    return ext in _FALLBACKS