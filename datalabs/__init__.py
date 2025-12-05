"""
Datalabs - Document Processing Pipeline
AI-powered document extraction for accountants.

This module provides a complete pipeline for:
- OCR (via Datalab API)
- Entity extraction (via LLM with evidence-driven prompts)
- Schema mapping (deterministic Pydantic validation)

Usage:
    from datalabs import process_document, ProcessingStatus

    result = process_document(
        file_content=file_bytes,
        filename="invoice.pdf",
        debug_mode=True
    )

    if result["status"] == ProcessingStatus.COMPLETE.value:
        extraction = result["mapped_output"]
        print(extraction)
"""

from .graph import process_document, process_document_async, get_tools
from .smart_pipeline import process_document_smart, classify_document
from .schemas import (
    ProcessingStatus,
    DocumentType,
    FieldStatus,
    ExtractedField,
    ExtractionOutput,
    RawExtractionOutput,
    ProcessingResponse,
    HealthResponse
)
from .config import settings

__version__ = "3.0.0"

__all__ = [
    # Main functions
    "process_document",
    "process_document_async",
    "get_tools",

    # Smart pipeline (no OCR needed)
    "process_document_smart",
    "classify_document",

    # Schemas
    "ProcessingStatus",
    "DocumentType",
    "FieldStatus",
    "ExtractedField",
    "ExtractionOutput",
    "RawExtractionOutput",
    "ProcessingResponse",
    "HealthResponse",

    # Config
    "settings",
]
