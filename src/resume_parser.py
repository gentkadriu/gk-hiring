from typing import Optional
from docx import Document
from pypdf import PdfReader
import io

def _extract_pdf(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as fh:
        reader = PdfReader(fh)
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                continue
        return "\n".join(pages)

def _extract_docx(file_bytes: bytes) -> str:
    with io.BytesIO(file_bytes) as fh:
        doc = Document(fh)
        return "\n".join(p.text for p in doc.paragraphs)

def extract_text_from_any(uploaded_file: Optional[object]) -> str:
    if uploaded_file is None:
        return ""
    suffix = (uploaded_file.name or "").lower().rsplit(".",1)[-1]
    data = uploaded_file.read()
    if suffix == "pdf":
        return _extract_pdf(data)
    if suffix == "docx":
        return _extract_docx(data)
    return ""
