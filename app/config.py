from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openai_chat_model: str = Field(default="gpt-5.4-mini", alias="OPENAI_CHAT_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        alias="OPENAI_EMBEDDING_MODEL",
    )

    app_host: str = Field(default="127.0.0.1", alias="APP_HOST")
    app_port: int = Field(default=8000, alias="APP_PORT")
    data_dir: Path = Field(default=Path("data"), alias="DATA_DIR")

    rag_top_k: int = Field(default=5, ge=0, le=20, alias="RAG_TOP_K")
    chunk_size: int = Field(default=900, ge=200, le=4000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=160, ge=0, le=1000, alias="CHUNK_OVERLAP")
    max_output_tokens: int = Field(default=1200, ge=128, le=16000, alias="MAX_OUTPUT_TOKENS")
    max_upload_mb: int = Field(default=10, ge=1, le=100, alias="MAX_UPLOAD_MB")

    @property
    def vector_store_path(self) -> Path:
        return self.data_dir / "vector_store.json"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_directories()
    return settings

