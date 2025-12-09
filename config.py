"""
Centralized configuration management.
All parameters are configurable via environment variables.
"""

import os
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


def get_list_from_env(key: str, default: str = "") -> List[str]:
    """Parse comma-separated environment variable into list."""
    value = os.getenv(key, default)
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def get_int(key: str, default: int) -> int:
    """Get integer from environment variable."""
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_float(key: str, default: float) -> float:
    """Get float from environment variable."""
    try:
        return float(os.getenv(key, str(default)))
    except ValueError:
        return default


def get_bool(key: str, default: bool) -> bool:
    """Get boolean from environment variable."""
    value = os.getenv(key, str(default)).lower()
    return value in ("true", "1", "yes", "on")


class Config:
    """Application configuration loaded from environment variables."""

    # ==========================================================================
    # API Keys (supports multiple keys for rotation)
    # All models use OpenRouter
    # ==========================================================================
    OPENROUTER_API_KEYS: List[str] = get_list_from_env("OPENROUTER_API_KEYS")

    # Fallback to single key if list not provided
    if not OPENROUTER_API_KEYS:
        single_key = os.getenv("OPENROUTER_API_KEY", "")
        if single_key:
            OPENROUTER_API_KEYS = [single_key]

    # ==========================================================================
    # Model Configuration (all via OpenRouter)
    # ==========================================================================
    VISION_MODEL: str = os.getenv("VISION_MODEL", "qwen/qwen2.5-vl-72b-instruct")
    CLASSIFIER_MODEL: str = os.getenv("CLASSIFIER_MODEL", "openai/gpt-4o-mini")
    EXTRACTION_MODEL: str = os.getenv("EXTRACTION_MODEL", "qwen/qwen2.5-vl-72b-instruct")

    # ==========================================================================
    # Token Limits
    # ==========================================================================
    VISION_MAX_TOKENS: int = get_int("VISION_MAX_TOKENS", 4096)
    CLASSIFIER_MAX_TOKENS: int = get_int("CLASSIFIER_MAX_TOKENS", 2048)
    EXTRACTION_MAX_TOKENS: int = get_int("EXTRACTION_MAX_TOKENS", 8192)

    # ==========================================================================
    # Thinking Budget (for models that support it)
    # ==========================================================================
    VISION_THINKING_BUDGET: Optional[int] = get_int("VISION_THINKING_BUDGET", 0) or None
    CLASSIFIER_THINKING_BUDGET: Optional[int] = get_int("CLASSIFIER_THINKING_BUDGET", 0) or None
    EXTRACTION_THINKING_BUDGET: Optional[int] = get_int("EXTRACTION_THINKING_BUDGET", 0) or None

    # ==========================================================================
    # Temperature Settings
    # ==========================================================================
    VISION_TEMPERATURE: float = get_float("VISION_TEMPERATURE", 0.1)
    CLASSIFIER_TEMPERATURE: float = get_float("CLASSIFIER_TEMPERATURE", 0.0)
    EXTRACTION_TEMPERATURE: float = get_float("EXTRACTION_TEMPERATURE", 0.1)

    # ==========================================================================
    # Retry Configuration
    # ==========================================================================
    MAX_RETRIES: int = get_int("MAX_RETRIES", 3)
    RETRY_BASE_DELAY: float = get_float("RETRY_BASE_DELAY", 1.0)  # seconds
    RETRY_MAX_DELAY: float = get_float("RETRY_MAX_DELAY", 30.0)  # seconds
    RETRY_EXPONENTIAL_BASE: float = get_float("RETRY_EXPONENTIAL_BASE", 2.0)

    # ==========================================================================
    # Concurrency & Rate Limiting
    # ==========================================================================
    MAX_CONCURRENT_REQUESTS: int = get_int("MAX_CONCURRENT_REQUESTS", 50)
    REQUESTS_PER_MINUTE_PER_KEY: int = get_int("REQUESTS_PER_MINUTE_PER_KEY", 60)
    BATCH_SIZE: int = get_int("BATCH_SIZE", 10)

    # ==========================================================================
    # Timeouts (seconds)
    # ==========================================================================
    VISION_TIMEOUT: int = get_int("VISION_TIMEOUT", 60)
    CLASSIFIER_TIMEOUT: int = get_int("CLASSIFIER_TIMEOUT", 30)
    EXTRACTION_TIMEOUT: int = get_int("EXTRACTION_TIMEOUT", 90)

    # ==========================================================================
    # Server Configuration
    # ==========================================================================
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = get_int("PORT", 5051)
    DEBUG: bool = get_bool("DEBUG", False)
    MAX_CONTENT_LENGTH: int = get_int("MAX_CONTENT_LENGTH", 16 * 1024 * 1024)  # 16MB

    # ==========================================================================
    # PDF Processing
    # ==========================================================================
    PDF_DPI_SCALE: float = get_float("PDF_DPI_SCALE", 2.0)
    PDF_MAX_PAGES: int = get_int("PDF_MAX_PAGES", 10)

    # ==========================================================================
    # Image Optimization for Vision API
    # ==========================================================================
    # Max dimension for resizing (2048 preserves more detail)
    IMAGE_MAX_DIMENSION: int = get_int("IMAGE_MAX_DIMENSION", 2048)
    # JPEG quality for compression (1-100, 92 = high quality with reasonable size)
    IMAGE_QUALITY: int = get_int("IMAGE_QUALITY", 92)
    # Enable/disable image optimization
    IMAGE_OPTIMIZATION_ENABLED: bool = get_bool("IMAGE_OPTIMIZATION_ENABLED", True)

    # ==========================================================================
    # API Base URL (OpenRouter for all models)
    # ==========================================================================
    OPENROUTER_BASE_URL: str = os.getenv(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1/chat/completions"
    )

    # ==========================================================================
    # Logging
    # ==========================================================================
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = os.getenv(
        "LOG_FORMAT",
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    @classmethod
    def validate(cls) -> List[str]:
        """Validate configuration and return list of warnings/errors."""
        issues = []

        if not cls.OPENROUTER_API_KEYS:
            issues.append("WARNING: No OpenRouter API keys configured")

        if cls.MAX_CONCURRENT_REQUESTS > 100:
            issues.append("WARNING: Very high concurrency may cause rate limiting")

        # Calculate theoretical throughput
        total_rpm = len(cls.OPENROUTER_API_KEYS) * cls.REQUESTS_PER_MINUTE_PER_KEY
        if total_rpm < 300:
            issues.append(
                f"INFO: Current config supports ~{total_rpm} requests/min. "
                f"Add more API keys for 300 docs/min target."
            )

        return issues

    @classmethod
    def print_config(cls):
        """Print current configuration (masking sensitive values)."""
        print("\n" + "=" * 60)
        print("Configuration Summary")
        print("=" * 60)
        print(f"OpenRouter Keys: {len(cls.OPENROUTER_API_KEYS)} configured")
        print(f"Vision Model: {cls.VISION_MODEL}")
        print(f"Classifier Model: {cls.CLASSIFIER_MODEL}")
        print(f"Extraction Model: {cls.EXTRACTION_MODEL}")
        print(f"Max Concurrent Requests: {cls.MAX_CONCURRENT_REQUESTS}")
        print(f"Batch Size: {cls.BATCH_SIZE}")
        print(f"Max Retries: {cls.MAX_RETRIES}")
        print("-" * 60)

        # Theoretical throughput
        total_rpm = len(cls.OPENROUTER_API_KEYS) * cls.REQUESTS_PER_MINUTE_PER_KEY
        print(f"Theoretical Max Throughput: ~{total_rpm} requests/min")
        print("=" * 60 + "\n")


# Create singleton instance
config = Config()
