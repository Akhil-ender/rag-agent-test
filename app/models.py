from typing import List, Optional
from pydantic import BaseModel


class IngestResponse(BaseModel):
    doc_id: str
    chunks: int
    pages: int
    ocr_pages: int
    source_path: str


class ChatRequest(BaseModel):
    message: str
    doc_id: Optional[str] = None
    stream: bool = False


class SourceChunk(BaseModel):
    doc_id: str
    page: int
    chunk_id: str
    snippet: str
    ocr: bool


class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
