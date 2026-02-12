# rapidindex/utils/config.py
"""Configuration management."""

import os
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class RapidIndexConfig(BaseSettings):
    """Application configuration."""
    
    # Database
    database_url: str = Field(default="sqlite:///./rapidindex.db")
    
    # Cache
    cache_dir: str = Field(default=".cache")
    cache_enabled: bool = Field(default=True)
    
    # LLM Provider (anthropic or ollama)
    llm_provider: str = Field(default="ollama")  # Changed default to free option
    
    # Anthropic (paid)
    anthropic_api_key: Optional[str] = Field(default=None)
    anthropic_model: str = Field(default="claude-sonnet-4-20250514")
    
    # Ollama (free)
    ollama_url: str = Field(default="http://localhost:11434")
    ollama_model: str = Field(default="llama3.2")
    
    # Generic LLM model (used as fallback)
    llm_model: Optional[str] = Field(default=None)
    
    # Embeddings
    embedding_model: str = Field(default="all-MiniLM-L6-v2")
    
    # Retrieval
    bm25_top_k: int = Field(default=20)
    final_top_k: int = Field(default=5)
    
    # Indexing
    batch_size: int = Field(default=10)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global config instance
config = RapidIndexConfig()


__all__ = ['config', 'RapidIndexConfig']