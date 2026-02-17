import os
import uuid
from functools import lru_cache

import uvicorn
from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .agent import Agent
from .config import Settings, ensure_dirs, get_settings
from .ingest import Embedder, Ingestor
from .llm import OpenAIClient
from .models import ChatRequest, ChatResponse, IngestResponse, SourceChunk
from .vectorstore import init_vectorstore


app = FastAPI(title="Document QA Agent")


@lru_cache()
def get_embedder():
    settings = get_settings()
    return Embedder(settings.embed_model)


@lru_cache()
def get_vectorstore():
    settings = get_settings()
    return init_vectorstore(settings.index_dir)


@lru_cache()
def get_llm():
    settings = get_settings()
    return OpenAIClient(settings.openai_api_key, settings.openai_model)


@lru_cache()
def get_agent():
    settings = get_settings()
    embedder = get_embedder()
    vectorstore = get_vectorstore()
    llm = get_llm()
    return Agent(llm=llm, vectorstore=vectorstore, embedder=lambda q: embedder.embed([q])[0])


def deps(settings: Settings = Depends(get_settings)):
    ensure_dirs(settings)
    return (
        settings,
        get_vectorstore(),
        get_embedder(),
        get_llm(),
        get_agent(),
        Ingestor(settings, vision_api_key=settings.openai_api_key, vision_model=settings.openai_vision_model),
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...), deps_tuple=Depends(deps)):
    settings, vectorstore, embedder, llm, agent, ingestor = deps_tuple
    if not file.filename:
        raise HTTPException(status_code=422, detail="file is required")
    suffix = os.path.splitext(file.filename)[1].lower()
    if suffix not in {".pdf", ".png", ".jpg", ".jpeg"}:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    save_name = f"{uuid.uuid4()}{suffix}"
    save_path = os.path.join(settings.uploads_dir, save_name)
    with open(save_path, "wb") as f:
        f.write(file.file.read())
    doc_id, texts, chunks, pages, ocr_pages = ingestor.ingest(save_path)
    embeddings = embedder.embed(texts)
    metas = []
    for chunk, emb in zip(chunks, embeddings):
        metas.append(
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "page": chunk.page,
                "text": chunk.text,
                "source_path": chunk.source_path,
                "ocr": chunk.ocr,
                "md5": chunk.md5,
            }
        )
    vectorstore.add(embeddings, metas)
    return IngestResponse(doc_id=doc_id, chunks=len(chunks), pages=pages, ocr_pages=ocr_pages, source_path=save_path)


@app.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, deps_tuple=Depends(deps)):
    settings, vectorstore, embedder, llm, agent, ingestor = deps_tuple
    if not payload.message:
        raise HTTPException(status_code=422, detail="message is required")
    if payload.stream:
        return StreamingResponse(agent.stream(payload.message, doc_id=payload.doc_id), media_type="text/event-stream")
    result = agent.run(payload.message, doc_id=payload.doc_id)
    sources = [SourceChunk(**s) for s in result.get("sources", [])]
    return ChatResponse(answer=result.get("answer", ""), sources=sources)


if __name__ == "__main__":
    uvicorn.run("app.server:app", host="0.0.0.0", port=8000, reload=True)
