# Document/Image QA Agent

FastAPI + LangGraph agent that ingests PDFs or images, indexes with FAISS + local sentence-transformers embeddings, and answers questions using OpenAI LLMs (GPT-4.1/4o) with optional OCR fallback.

## Setup

1) Activate env (recommended):
```bash
conda activate genai
```

2) Install deps:
```bash
pip install -r requirements.txt
```

3) Set environment variables (put in `.env` if you like):
```
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4.1
OPENAI_VISION_MODEL=gpt-4o
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
CHUNK_SIZE=800
CHUNK_OVERLAP=120
```

## Run
```bash
uvicorn app.server:app --reload --port 8000
```

## API
- `POST /ingest` (multipart `file`): returns `doc_id`, chunk count, pages, OCR pages, source path.
- `POST /chat` (json `{message, doc_id?, stream?}`): returns `answer` and source chunks.
- `GET /health`

## Notes
- Vectors and metadata stored under `data/faiss_index/`.
- Uploaded files stored under `data/uploads/`.
- OCR fallback uses vision LLM; requires OpenAI key.
- If `poppler` missing, pdf2image OCR fallback may be skipped.
# rag-agent-test
