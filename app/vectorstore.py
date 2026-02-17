import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np


class LocalFAISS:
    """Simple FAISS index with sidecar metadata stored as JSON."""

    def __init__(self, index_path: str, meta_path: Optional[str] = None) -> None:
        self.index_path = Path(index_path)
        self.meta_path = Path(meta_path) if meta_path else self.index_path.with_suffix(".json")
        self.lock = threading.Lock()
        self.index: Optional[faiss.IndexFlatIP] = None
        self.metadata: List[Dict[str, Any]] = []

    def _normalize(self, vecs: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        return vecs / norms

    def add(self, embeddings: List[List[float]], metas: List[Dict[str, Any]]) -> None:
        if len(embeddings) != len(metas):
            raise ValueError("Embeddings and metadata length mismatch")
        arr = np.array(embeddings, dtype="float32")
        arr = self._normalize(arr)
        with self.lock:
            if self.index is None:
                self.index = faiss.IndexFlatIP(arr.shape[1])
            self.index.add(arr)
            self.metadata.extend(metas)
            self.save()

    def search(self, query: List[float], k: int = 6) -> List[Tuple[float, Dict[str, Any]]]:
        if self.index is None or self.index.ntotal == 0:
            return []
        q = np.array([query], dtype="float32")
        q = self._normalize(q)
        with self.lock:
            scores, idxs = self.index.search(q, k)
        results: List[Tuple[float, Dict[str, Any]]] = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            meta = self.metadata[idx]
            results.append((float(score), meta))
        return results

    def save(self) -> None:
        if self.index is None:
            return
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)

    def load(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.meta_path.exists():
            with open(self.meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

    @property
    def size(self) -> int:
        return len(self.metadata)


def init_vectorstore(index_dir: str) -> LocalFAISS:
    index_path = os.path.join(index_dir, "index.faiss")
    vs = LocalFAISS(index_path)
    if os.path.exists(index_path):
        vs.load()
    return vs
