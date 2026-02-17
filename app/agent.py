from typing import Any, Callable, Dict, List, Optional

from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from .llm import OpenAIClient
from .vectorstore import LocalFAISS


class GraphState(Dict[str, Any]):
    question: str
    doc_id: Optional[str]
    history: List[Dict[str, str]]
    retrieved: List[Dict[str, Any]]
    answer: str
    sources: List[Dict[str, Any]]


class Agent:
    def __init__(self, llm: OpenAIClient, vectorstore: LocalFAISS, embedder: Callable[[str], List[float]]) -> None:
        self.llm = llm
        self.vectorstore = vectorstore
        self.embedder = embedder
        graph = StateGraph(GraphState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate", self._generate)
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", END)
        graph.set_entry_point("retrieve")
        self.app = graph.compile()

    def _retrieve(self, state: GraphState) -> GraphState:
        query = state["question"]
        doc_id = state.get("doc_id")
        embedding = self.embedder(query)
        results = self.vectorstore.search(embedding, k=6)
        if doc_id:
            results = [r for r in results if r[1].get("doc_id") == doc_id]
        state["retrieved"] = [meta for _, meta in results]
        return state

    def _generate(self, state: GraphState) -> GraphState:
        retrieved = state.get("retrieved", [])
        if not retrieved:
            state["answer"] = "I could not find relevant information in the indexed documents."
            state["sources"] = []
            return state
        context_lines = []
        sources = []
        for meta in retrieved:
            snippet = meta.get("text", "")[:400]
            context_lines.append(f"[doc:{meta.get('doc_id')}] page {meta.get('page')}, chunk {meta.get('id')}: {snippet}")
            sources.append(
                {
                    "doc_id": meta.get("doc_id"),
                    "page": meta.get("page"),
                    "chunk_id": meta.get("id"),
                    "snippet": snippet,
                    "ocr": meta.get("ocr", False),
                }
            )
        system_prompt = (
            "You are a concise assistant. Answer using only the provided context. "
            "Cite chunks as [doc_id:chunk_id] inline."
        )
        user_prompt = (
            "Question: "
            + state["question"]
            + "\nContext:\n"
            + "\n".join(context_lines)
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        parts = self.llm.complete(messages, stream=False)
        answer = "".join(parts)
        state["answer"] = answer
        state["sources"] = sources
        return state

    def run(self, question: str, history: Optional[List[Dict[str, str]]] = None, doc_id: Optional[str] = None) -> Dict[str, Any]:
        initial: GraphState = {
            "question": question,
            "history": history or [],
            "doc_id": doc_id,
        }
        result = self.app.invoke(initial)
        return {"answer": result.get("answer", ""), "sources": result.get("sources", [])}

    def stream(self, question: str, doc_id: Optional[str] = None):
        # manual retrieval + streamed generation to preserve sources
        embedding = self.embedder(question)
        results = self.vectorstore.search(embedding, k=6)
        if doc_id:
            results = [r for r in results if r[1].get("doc_id") == doc_id]
        metas = [meta for _, meta in results]
        if not metas:
            yield "data: " + "I could not find relevant information in the indexed documents.\\n\\n" + "\\n\\n"
            yield "data: [SOURCES] []\\n\\n"
            yield "data: [DONE]\\n\\n"
            return
        context_lines = []
        sources = []
        for meta in metas:
            snippet = meta.get("text", "")[:400]
            context_lines.append(f"[doc:{meta.get('doc_id')}] page {meta.get('page')}, chunk {meta.get('id')}: {snippet}")
            sources.append(
                {
                    "doc_id": meta.get("doc_id"),
                    "page": meta.get("page"),
                    "chunk_id": meta.get("id"),
                    "snippet": snippet,
                    "ocr": meta.get("ocr", False),
                }
            )
        system_prompt = (
            "You are a concise assistant. Answer using only the provided context. "
            "Cite chunks as [doc_id:chunk_id] inline."
        )
        user_prompt = "Question: " + question + "\\nContext:\\n" + "\\n".join(context_lines)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        for token in self.llm.complete(messages, stream=True):
            yield f"data: {token}\\n\\n"
        yield "data: [SOURCES] " + json_dumps_safe(sources) + "\\n\\n"
        yield "data: [DONE]\\n\\n"


def json_dumps_safe(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj)
    except Exception:
        return "[]"
