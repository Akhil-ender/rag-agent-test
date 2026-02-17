import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

# Work around pyarrow>=19 removal of PyExtensionType expected by datasets -> sentence_transformers
try:
    import pyarrow as pa

    if not hasattr(pa, "PyExtensionType"):
        pa.PyExtensionType = pa.ExtensionType  # type: ignore[attr-defined]
except Exception:
    pass

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from PIL import Image
from pdf2image import convert_from_path

from .config import Settings
from .llm import vision_ocr


@dataclass
class Chunk:
    id: str
    doc_id: str
    page: int
    text: str
    source_path: str
    ocr: bool
    md5: str


class Embedder:
    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, normalize_embeddings=False).tolist()


class Ingestor:
    def __init__(self, settings: Settings, vision_api_key: str, vision_model: str) -> None:
        self.settings = settings
        self.vision_api_key = vision_api_key
        self.vision_model = vision_model
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
        )

    def _chunk(self, chunks: List[Chunk]) -> Tuple[List[str], List[Chunk]]:
        texts: List[str] = []
        expanded: List[Chunk] = []
        for c in chunks:
            for i, piece in enumerate(self.splitter.split_text(c.text)):
                chunk_id = f"{c.id}-{i}"
                if not piece.strip():
                    continue
                expanded.append(
                    Chunk(
                        id=chunk_id,
                        doc_id=c.doc_id,
                        page=c.page,
                        text=piece,
                        source_path=c.source_path,
                        ocr=c.ocr,
                        md5=self._md5(piece),
                    )
                )
                texts.append(piece)
        return texts, expanded

    def _md5(self, text: str) -> str:
        return hashlib.md5(text.encode("utf-8")).hexdigest()

    def _ocr_page(self, image) -> str:
        import io

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return vision_ocr(buf.getvalue(), api_key=self.vision_api_key, model=self.vision_model)

    def _ingest_pdf(self, path: Path, doc_id: str) -> Tuple[List[str], List[Chunk], int]:
        reader = PdfReader(str(path))
        chunks: List[Chunk] = []
        ocr_pages = 0
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            used_ocr = False
            if len(text.strip()) < 30:
                # fallback to OCR
                images = convert_from_path(str(path), first_page=page_num, last_page=page_num)
                if images:
                    text = self._ocr_page(images[0])
                    used_ocr = True
                    ocr_pages += 1
            chunks.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=doc_id,
                    page=page_num,
                    text=text,
                    source_path=str(path),
                    ocr=used_ocr,
                    md5=self._md5(text),
                )
            )
        return self._chunk(chunks) + (ocr_pages,)

    def _ingest_image(self, path: Path, doc_id: str) -> Tuple[List[str], List[Chunk], int]:
        image = Image.open(path)
        text = self._ocr_page(image)
        chunk = Chunk(
            id=str(uuid.uuid4()),
            doc_id=doc_id,
            page=1,
            text=text,
            source_path=str(path),
            ocr=True,
            md5=self._md5(text),
        )
        texts, expanded = self._chunk([chunk])
        return texts, expanded, 1

    def ingest(self, file_path: str) -> Tuple[str, List[str], List[Chunk], int, int]:
        path = Path(file_path)
        doc_id = str(uuid.uuid4())
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            texts, expanded, ocr_pages = self._ingest_pdf(path, doc_id)
            pages = len(set(c.page for c in expanded))
            return doc_id, texts, expanded, pages, ocr_pages
        if suffix in {".png", ".jpg", ".jpeg"}:
            texts, expanded, ocr_pages = self._ingest_image(path, doc_id)
            return doc_id, texts, expanded, 1, ocr_pages
        raise ValueError(f"Unsupported file type: {suffix}")
