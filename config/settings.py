"""
Application settings and configuration management.
Implements the Singleton pattern for settings access.
"""

from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings with environment variable support.
    Uses Pydantic for validation and type safety.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore"
    )

    # LLM Configuration
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    llm_model: str = "mistral:latest"
    llm_base_url: str = "http://localhost:11434"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 2048

    # Embedding Configuration
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    # Vector Store
    vector_store: str = "chromadb"
    chroma_persist_dir: str = "./chroma_db"

    # Retrieval Configuration
    top_k: int = 5
    similarity_threshold: float = 0.7

    # Chunking Configuration
    chunk_size: int = 512
    chunk_overlap: int = 50

    # Cache Configuration
    enable_cache: bool = True
    cache_ttl: int = 3600

    # API Configuration
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = True

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    # Paths
    ground_truth_path: str = "data/ground_truth.json"
    benchmark_results_path: str = "benchmarks/results.json"
    pdf_path: str = "data/research_paper.pdf"  # Single PDF (deprecated)
    pdf_paths: str = "data/research_paper.pdf,data/financial_report.pdf,data/employee_handbook.pdf"  # Comma-separated PDF paths
    
    def get_pdf_paths_list(self) -> list:
        """Get list of PDF paths from comma-separated string."""
        if self.pdf_paths:
            return [p.strip() for p in self.pdf_paths.split(",") if p.strip()]
        elif self.pdf_path:
            # Backward compatibility
            return [self.pdf_path]
        return []


@lru_cache()
def get_settings() -> Settings:
    """
    Get application settings singleton.
    Cached to avoid reloading from environment on every call.
    
    Returns:
        Settings: Application configuration instance
    """
    return Settings()
