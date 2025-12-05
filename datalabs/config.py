"""
Datalabs - Configuration Management
Adapted for imagePreprocessor integration.
Loads configuration from parent directory's .env file.
"""

import os
from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field
from dotenv import load_dotenv

# Load .env from parent directory (imagePreprocessor root)
env_path = Path(__file__).parent.parent / '.env'
load_dotenv(env_path)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # ==========================================================================
    # API KEYS
    # ==========================================================================
    DATALAB_API_KEY: str = Field(default="", description="Datalab OCR API key")
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key for LLM")

    # ==========================================================================
    # DATALAB OCR CONFIGURATION
    # ==========================================================================
    DATALAB_BASE_URL: str = Field(
        default="https://www.datalab.to/api/v1",
        description="Datalab API base URL"
    )
    DATALAB_MAX_POLLS: int = Field(default=300, description="Max polling attempts")
    DATALAB_POLL_INTERVAL: float = Field(default=2.0, description="Seconds between polls")
    DATALAB_OUTPUT_FORMAT: str = Field(default="html", description="OCR output format")
    DATALAB_FORCE_OCR: bool = Field(default=False, description="Force OCR on all pages")
    DATALAB_USE_LLM: bool = Field(default=False, description="Use LLM enhancement in Datalab")

    # ==========================================================================
    # LLM CONFIGURATION (via OpenRouter)
    # ==========================================================================
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL"
    )

    # Vision models (for image analysis)
    # Options: meta-llama/llama-3.2-11b-vision-instruct (free), qwen/qwen-2-vl-7b-instruct,
    #          google/gemini-2.0-flash-001, google/gemini-2.5-flash-preview
    VISION_MODEL: str = Field(
        default="google/gemini-2.0-flash-001",
        description="Vision model for image analysis and classification"
    )

    # Extraction model (for document extraction - needs vision)
    EXTRACTION_MODEL: str = Field(
        default="google/gemini-2.0-flash-001",
        description="Model for entity extraction (vision-capable)"
    )

    # Text-only model for post-processing (smaller, faster)
    # Options: meta-llama/llama-3.1-8b-instruct, mistralai/mistral-7b-instruct, google/gemma-2-9b-it
    TEXT_MODEL: str = Field(
        default="meta-llama/llama-3.1-8b-instruct",
        description="Text model for validation/post-processing"
    )

    MAPPING_MODEL: str = Field(
        default="meta-llama/llama-3.1-8b-instruct",
        description="Model for schema mapping"
    )
    LLM_TEMPERATURE: float = Field(default=0.0, description="LLM temperature")
    LLM_MAX_TOKENS: int = Field(default=4000, description="Max tokens in response")

    # ==========================================================================
    # SERVER CONFIGURATION
    # ==========================================================================
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    DEBUG: bool = Field(default=False, description="Debug mode")

    # ==========================================================================
    # LOGGING CONFIGURATION
    # ==========================================================================
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_DIR: str = Field(default="./logs", description="Log files directory")
    LOG_TO_FILE: bool = Field(default=True, description="Enable file logging")
    LOG_TO_CONSOLE: bool = Field(default=True, description="Enable console logging")

    # ==========================================================================
    # PROCESSING CONFIGURATION
    # ==========================================================================
    REQUEST_TIMEOUT: int = Field(default=300, description="Request timeout in seconds")
    MAX_FILE_SIZE_MB: int = Field(default=50, description="Max upload file size in MB")

    class Config:
        env_file = str(env_path)
        env_file_encoding = "utf-8"
        extra = "ignore"

    @property
    def log_dir_path(self) -> Path:
        """Get log directory as Path object."""
        # Use absolute path relative to imagePreprocessor root
        base_dir = Path(__file__).parent.parent
        path = base_dir / self.LOG_DIR.lstrip('./')
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_file_size_bytes(self) -> int:
        """Get max file size in bytes."""
        return self.MAX_FILE_SIZE_MB * 1024 * 1024


# Global settings instance
settings = Settings()
