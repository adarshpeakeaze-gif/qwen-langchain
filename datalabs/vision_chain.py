"""
Vision Chain Pipeline
Image → Qwen VL (HTML) → LLM (Extraction)

Two-step process:
1. Qwen VL vision model converts image to HTML (what it sees)
2. Gemini extracts structured data from HTML
"""

import uuid
import time
import json
from typing import Dict, Any

from loguru import logger

from .tools import convert_image_to_html, extract_from_html


def process_vision_chain(
    image_base64: str,
    document_type: str = "auto",
    request_id: str = None
) -> Dict[str, Any]:
    """
    Process document through vision chain.

    Step 1: Image → Qwen VL → HTML
    Step 2: HTML → Gemini → Extracted Data

    Args:
        image_base64: Base64 encoded image (enhanced or binary)
        document_type: Type hint for extraction (auto, receipt, invoice, bill)
        request_id: Optional request ID for logging

    Returns:
        Complete result with HTML and extracted data
    """
    if not request_id:
        request_id = str(uuid.uuid4())[:8]

    logger.info(f"[{request_id}] Vision Chain: Starting")
    total_start = time.time()

    result = {
        "request_id": request_id,
        "status": "processing",
        "steps": {}
    }

    # =========================================================================
    # STEP 1: Image → Qwen VL → HTML
    # =========================================================================
    logger.info(f"[{request_id}] Step 1: Image → Qwen VL (HTML)")
    step1_start = time.time()

    try:
        vision_result = convert_image_to_html(image_base64)
        step1_time = int((time.time() - step1_start) * 1000)

        if vision_result.get("status") == "error":
            logger.error(f"[{request_id}] Step 1 failed: {vision_result.get('error')}")
            return {
                **result,
                "status": "failed",
                "error": vision_result.get("error"),
                "error_stage": "vision_to_html",
                "steps": {
                    "vision_to_html": {
                        "status": "failed",
                        "error": vision_result.get("error"),
                        "time_ms": step1_time,
                        "model": "qwen/qwen3-vl-8b-instruct"
                    }
                }
            }

        html_content = vision_result.get("html", "")

        result["steps"]["vision_to_html"] = {
            "status": "success",
            "model": vision_result.get("model", "qwen/qwen3-vl-8b-instruct"),
            "html": html_content,
            "char_count": len(html_content),
            "time_ms": step1_time
        }

        logger.info(f"[{request_id}] Step 1 complete: {len(html_content)} chars [{step1_time}ms]")

    except Exception as e:
        step1_time = int((time.time() - step1_start) * 1000)
        logger.error(f"[{request_id}] Step 1 exception: {e}")
        return {
            **result,
            "status": "failed",
            "error": str(e),
            "error_stage": "vision_to_html",
            "steps": {
                "vision_to_html": {
                    "status": "failed",
                    "error": str(e),
                    "time_ms": step1_time
                }
            }
        }

    # =========================================================================
    # STEP 2: HTML → Gemini → Extraction
    # =========================================================================
    logger.info(f"[{request_id}] Step 2: HTML → Gemini (Extraction)")
    step2_start = time.time()

    # Auto-detect document type from HTML if not specified
    if document_type == "auto":
        html_lower = html_content.lower()
        if "receipt" in html_lower or "total" in html_lower:
            document_type = "receipt"
        elif "invoice" in html_lower:
            document_type = "invoice"
        elif "bill" in html_lower or "account" in html_lower:
            document_type = "bill"
        else:
            document_type = "receipt"  # Default
        logger.info(f"[{request_id}] Auto-detected document type: {document_type}")

    try:
        extraction_result = extract_from_html(html_content, document_type)
        step2_time = int((time.time() - step2_start) * 1000)

        if extraction_result.get("status") == "error":
            logger.error(f"[{request_id}] Step 2 failed: {extraction_result.get('error')}")
            result["steps"]["extraction"] = {
                "status": "failed",
                "error": extraction_result.get("error"),
                "time_ms": step2_time,
                "model": "google/gemini-2.0-flash-001"
            }
            # Still return partial success with HTML
            result["status"] = "partial"
            result["html"] = html_content
            result["error"] = extraction_result.get("error")
            result["error_stage"] = "extraction"
        else:
            result["steps"]["extraction"] = {
                "status": "success",
                "model": extraction_result.get("model", "google/gemini-2.0-flash-001"),
                "time_ms": step2_time
            }

            result["status"] = "success"
            result["html"] = html_content
            result["document_type"] = extraction_result.get("document_type", document_type)
            result["extracted_data"] = extraction_result.get("extracted_data", {})
            result["confidence"] = extraction_result.get("confidence", 0.0)
            result["warnings"] = extraction_result.get("warnings", [])

            logger.info(f"[{request_id}] Step 2 complete: extracted data [{step2_time}ms]")

    except Exception as e:
        step2_time = int((time.time() - step2_start) * 1000)
        logger.error(f"[{request_id}] Step 2 exception: {e}")
        result["steps"]["extraction"] = {
            "status": "failed",
            "error": str(e),
            "time_ms": step2_time
        }
        result["status"] = "partial"
        result["html"] = html_content
        result["error"] = str(e)
        result["error_stage"] = "extraction"

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    total_time = int((time.time() - total_start) * 1000)
    result["total_time_ms"] = total_time
    result["models_used"] = {
        "vision": "qwen/qwen3-vl-8b-instruct",
        "extraction": "google/gemini-2.0-flash-001"
    }

    logger.info(f"[{request_id}] Vision Chain complete: {result['status']} [{total_time}ms]")

    return result
