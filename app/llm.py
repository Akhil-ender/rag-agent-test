import base64
import json
from typing import Generator, Iterable, List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage


class OpenAIClient:
    """Wrapper around Langchain ChatOpenAI for OpenAI models."""

    def __init__(self, api_key: str, model: str) -> None:
        self.api_key = api_key
        self.model = model
        self.client = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.7,
        )
        self.streaming_client = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=0.7,
            streaming=True,
        )

    def complete(self, messages: List[dict], stream: bool = False, extra: Optional[dict] = None) -> Iterable[str]:
        """Convert dict messages to Langchain format and get completion."""
        lang_messages = self._convert_messages(messages)
        
        if stream:
            return self._stream_complete(lang_messages, extra)
        else:
            return self._non_stream_complete(lang_messages, extra)
    
    def _convert_messages(self, messages: List[dict]) -> List[BaseMessage]:
        """Convert OpenAI-style messages to Langchain message objects."""
        lang_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                lang_messages.append(SystemMessage(content=content))
            elif role == "user":
                lang_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                from langchain_core.messages import AIMessage
                lang_messages.append(AIMessage(content=content))
        return lang_messages
    
    def _stream_complete(self, messages: List[BaseMessage], extra: Optional[dict]) -> Generator[str, None, None]:
        """Stream completion from OpenAI."""
        kwargs = {}
        if extra:
            # Map common params if needed
            if "temperature" in extra:
                kwargs["temperature"] = extra["temperature"]
            if "max_tokens" in extra:
                kwargs["max_tokens"] = extra["max_tokens"]
        
        for chunk in self.streaming_client.stream(messages, **kwargs):
            if chunk.content:
                yield chunk.content
    
    def _non_stream_complete(self, messages: List[BaseMessage], extra: Optional[dict]) -> List[str]:
        """Get non-streaming completion from OpenAI."""
        kwargs = {}
        if extra:
            if "temperature" in extra:
                kwargs["temperature"] = extra["temperature"]
            if "max_tokens" in extra:
                kwargs["max_tokens"] = extra["max_tokens"]
        
        response = self.client.invoke(messages, **kwargs)
        return [response.content]


def vision_ocr(image_bytes: bytes, api_key: str, model: str, prompt: str = "Extract text from this image.") -> str:
    """Extract text from image using OpenAI vision model."""
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    vision_client = ChatOpenAI(
        api_key=api_key,
        model=model,
    )
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{b64}"},
            },
        ]
    )
    
    response = vision_client.invoke([message])
    return response.content
