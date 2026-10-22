from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import PurePosixPath

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_PDF_PAGES = 8
MAX_DOCX_PARAGRAPHS = 1200
MAX_TEXT_CHARS = 12000
MAX_ZIP_MEMBERS = 400
MAX_ZIP_EXPANDED = 8 * 1024 * 1024
MAX_ZIP_MEMBER = 4 * 1024 * 1024
WORKER_TIMEOUT_SECONDS = 5

MESSAGES = {
    "missing_file": "Choose a PDF or DOCX résumé.",
    "unsupported_type": "Only PDF and DOCX documents are accepted.",
    "too_large": "That file is too large. Maximum size is 2 MB.",
    "malformed": "The document cannot be read. Check that it is a valid PDF or DOCX.",
    "encrypted": "Password-protected or encrypted documents cannot be processed.",
    "too_many_pages": "PDFs must contain no more than 8 pages.",
    "too_many_paragraphs": "DOCX files must contain no more than 1,200 paragraphs.",
    "unsafe_archive": "This DOCX contains unsupported or unsafe embedded content.",
    "text_too_long": "The résumé contains too much text for this tool.",
    "no_text": "No extractable text found. Scanned PDFs need a text-based copy; OCR is not supported.",
    "timeout": "The document took too long to process. Try a simpler text-based file.",
    "parse_failed": "The document could not be processed. Try another text-based file.",
}


class ResumeError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(MESSAGES[code])


@dataclass(frozen=True)
class ExtractedResume:
    text: str
    document_type: str
    pages_or_paragraphs: int


def accepted_type(filename: str, data: bytes) -> str:
    suffix = PurePosixPath(filename.replace("\\", "/")).suffix.casefold()
    if suffix not in {".pdf", ".docx"}:
        raise ResumeError("unsupported_type")
    if suffix == ".pdf" and data.startswith(b"%PDF-"):
        return "pdf"
    if suffix == ".docx" and data.startswith(b"PK\x03\x04"):
        return "docx"
    raise ResumeError("malformed")


def _pdf_text(data: bytes) -> ExtractedResume:
    from pypdf import PdfReader

    try:
        reader = PdfReader(io.BytesIO(data), strict=True)
        if reader.is_encrypted:
            raise ResumeError("encrypted")
        count = len(reader.pages)
        if not 1 <= count <= MAX_PDF_PAGES:
            raise ResumeError("too_many_pages")
        root = reader.trailer.get("/Root")
        if root and "/AA" in root:
            raise ResumeError("malformed")
        if root and "/Names" in root and "/JavaScript" in root["/Names"]:
            raise ResumeError("malformed")
        if root and "/OpenAction" in root:
            from pypdf.generic import ArrayObject, DictionaryObject, IndirectObject

            action = root["/OpenAction"]
            if not isinstance(action, ArrayObject) or len(action) < 2:
                raise ResumeError("malformed")
            target = action[0]
            if not isinstance(target, IndirectObject):
                raise ResumeError("malformed")
            page_target = target.get_object()
            if not isinstance(page_target, DictionaryObject) or page_target.get("/Type") != "/Page":
                raise ResumeError("malformed")
            if action[1] not in {
                "/XYZ", "/Fit", "/FitH", "/FitV", "/FitR", "/FitB", "/FitBH", "/FitBV"
            }:
                raise ResumeError("malformed")
        parts = []
        length = 0
        for page in reader.pages:
            extracted = page.extract_text() or ""
            length += len(extracted)
            if length > MAX_TEXT_CHARS:
                raise ResumeError("text_too_long")
            parts.append(extracted)
        text = "\n".join(parts).strip()
    except ResumeError:
        raise
    except Exception as exc:
        raise ResumeError("malformed") from exc
    if not text:
        raise ResumeError("no_text")
    return ExtractedResume(text, "pdf", count)


def _check_docx_archive(data: bytes) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            members = archive.infolist()
            if not 1 <= len(members) <= MAX_ZIP_MEMBERS:
                raise ResumeError("unsafe_archive")
            names = {item.filename for item in members}
            if not {"[Content_Types].xml", "_rels/.rels", "word/document.xml"} <= names:
                raise ResumeError("malformed")
            expanded = 0
            for item in members:
                path = PurePosixPath(item.filename)
                name = item.filename.casefold()
                if (
                    item.flag_bits & 1
                    or item.filename.startswith("/")
                    or "\\" in item.filename
                    or ".." in path.parts
                    or item.file_size > MAX_ZIP_MEMBER
                    or (item.file_size and item.compress_size == 0)
                    or (item.compress_size and item.file_size > item.compress_size * 100)
                    or name.endswith((".bin", ".exe"))
                    or name.startswith(("word/embeddings/", "word/activex/"))
                ):
                    raise ResumeError("unsafe_archive")
                expanded += item.file_size
                if expanded > MAX_ZIP_EXPANDED:
                    raise ResumeError("unsafe_archive")
    except ResumeError:
        raise
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise ResumeError("malformed") from exc


def _docx_text(data: bytes) -> ExtractedResume:
    from docx import Document
    from docx.table import Table

    _check_docx_archive(data)
    try:
        document = Document(io.BytesIO(data))
        lines = []
        count = 0
        total = 0
        for block in document.iter_inner_content():
            if isinstance(block, Table):
                texts = [
                    " ".join(cell.text.splitlines())
                    for row in block.rows for cell in row.cells
                ]
            else:
                texts = [block.text]
            for text in texts:
                count += 1
                if count > MAX_DOCX_PARAGRAPHS:
                    raise ResumeError("too_many_paragraphs")
                total += len(text)
                if total > MAX_TEXT_CHARS:
                    raise ResumeError("text_too_long")
                lines.append(text)
    except ResumeError:
        raise
    except Exception as exc:
        raise ResumeError("malformed") from exc
    text = "\n".join(lines).strip()
    if not text:
        raise ResumeError("no_text")
    return ExtractedResume(text, "docx", count)


def extract_text(data: bytes, filename: str) -> ExtractedResume:
    if not data:
        raise ResumeError("missing_file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ResumeError("too_large")
    kind = accepted_type(filename, data)
    return _pdf_text(data) if kind == "pdf" else _docx_text(data)


def _worker_limits() -> None:
    if sys.platform == "win32":
        return
    import resource

    for key, limit in (
        (resource.RLIMIT_CPU, 4),
        (resource.RLIMIT_FSIZE, 0),
        (resource.RLIMIT_CORE, 0),
    ):
        try:
            resource.setrlimit(key, (limit, limit))
        except (OSError, ValueError):
            pass
    if sys.platform.startswith("linux"):
        try:
            cap = 512 * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))
        except (OSError, ValueError):
            pass


def _worker_main(filename: str) -> int:
    _worker_limits()
    try:
        data = sys.stdin.buffer.read(MAX_UPLOAD_BYTES + 1)
        result = extract_text(data, filename)
        message = {
            "text": result.text,
            "document_type": result.document_type,
            "pages_or_paragraphs": result.pages_or_paragraphs,
        }
    except ResumeError as exc:
        message = {"error": exc.code}
    except Exception:
        message = {"error": "parse_failed"}
    sys.stdout.write(json.dumps(message, ensure_ascii=True, separators=(",", ":")))
    return 0


def process_upload(storage) -> ExtractedResume:
    if not storage or not storage.filename:
        raise ResumeError("missing_file")
    filename = storage.filename
    if PurePosixPath(filename.replace("\\", "/")).suffix.casefold() not in {
        ".pdf", ".docx"
    }:
        raise ResumeError("unsupported_type")
    data = storage.stream.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise ResumeError("missing_file")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ResumeError("too_large")
    accepted_type(filename, data)
    # Only the fixed extension reaches the worker. Never pass a user path as a command.
    suffix = PurePosixPath(filename.replace("\\", "/")).suffix.casefold()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "salary_predictor.web.resume_extract", "resume" + suffix],
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=WORKER_TIMEOUT_SECONDS,
            check=False,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise ResumeError("timeout" if isinstance(exc, subprocess.TimeoutExpired)
                          else "parse_failed") from exc
    if result.returncode != 0 or len(result.stdout) > MAX_TEXT_CHARS * 8:
        raise ResumeError("parse_failed")
    try:
        payload = json.loads(result.stdout)
    except (ValueError, UnicodeDecodeError) as exc:
        raise ResumeError("parse_failed") from exc
    if "error" in payload:
        code = payload["error"]
        raise ResumeError(code if code in MESSAGES else "parse_failed")
    if (
        not isinstance(payload.get("text"), str)
        or not 0 < len(payload["text"]) <= MAX_TEXT_CHARS
        or payload.get("document_type") not in {"pdf", "docx"}
        or not isinstance(payload.get("pages_or_paragraphs"), int)
    ):
        raise ResumeError("parse_failed")
    return ExtractedResume(
        payload["text"], payload["document_type"], payload["pages_or_paragraphs"]
    )


if __name__ == "__main__":
    raise SystemExit(_worker_main(sys.argv[1] if len(sys.argv) == 2 else ""))
