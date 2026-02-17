import os
from functools import lru_cache
from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field("gpt-4.1", env="OPENAI_MODEL")
    openai_vision_model: str = Field("gpt-4o", env="OPENAI_VISION_MODEL")
    embed_model: str = Field("sentence-transformers/all-MiniLM-L6-v2", env="EMBED_MODEL")
    chunk_size: int = Field(800, env="CHUNK_SIZE")
    chunk_overlap: int = Field(120, env="CHUNK_OVERLAP")
    uploads_dir: str = Field("data/uploads", env="UPLOADS_DIR")
    index_dir: str = Field("data/faiss_index", env="INDEX_DIR")
    cache_dir: str = Field("data/cache", env="CACHE_DIR")

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()  # type: ignore[arg-type]


def ensure_dirs(settings: Settings) -> None:
    for path in (settings.uploads_dir, settings.index_dir, settings.cache_dir):
        os.makedirs(path, exist_ok=True)
