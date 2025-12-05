"""
Unified Document Processing Pipeline
Follows the flow:
1. Image uploaded
2. Image processing (preprocessing)
3. Original + Binary sent to LLM for classification
4. LLM classifies category (thermal_print, a4_invoice, handprint, pos_receipt)
5. Route to type-specific extractor
"""

import uuid
import time
import json
from typing import Dict, Any

from loguru import logger

from .document_classifier import classify_document
from .tools import convert_image_to_html
from .extractors import extract_thermal, extract_invoice, extract_handprint, extract_pos
from .utils import normalize_extraction_result, get_field_summary


# Extractor mapping
EXTRACTORS = {
    "thermal_print": extract_thermal,
    "a4_invoice": extract_invoice,
    "handprint": extract_handprint,
    "pos_receipt": extract_pos
}


def process_document(
    original_base64: str,
    binary_base64: str,
    request_id: str = None
) -> Dict[str, Any]:
    """
    Process document through unified pipeline.

    Flow:
    1. Classify using BOTH original and binary images
    2. Convert binary image to HTML using Qwen VL
    3. Route HTML to type-specific extractor

    Args:
        original_base64: Original uploaded image
        binary_base64: Binary/thresholded image
        request_id: Optional request ID for logging

    Returns:
        Complete processing result
    """
    if not request_id:
        request_id = str(uuid.uuid4())[:8]

    logger.info(f"[{request_id}] Unified Pipeline: Starting")
    total_start = time.time()

    result = {
        "request_id": request_id,
        "status": "processing",
        "steps": {}
    }

    # =========================================================================
    # STEP 1: CLASSIFY (with both original and binary images)
    # =========================================================================
    logger.info(f"[{request_id}] Step 1: Classifying document (dual images)")
    step1_start = time.time()

    classification = classify_document(original_base64, binary_base64)
    step1_time = int((time.time() - step1_start) * 1000)

    if classification.get("status") == "error":
        logger.error(f"[{request_id}] Classification failed: {classification.get('error')}")
        return {
            **result,
            "status": "failed",
            "error": classification.get("error"),
            "error_stage": "classification"
        }

    category = classification.get("category", "unknown")
    confidence = classification.get("confidence", 0.0)

    result["steps"]["classification"] = {
        "status": "success",
        "category": category,
        "confidence": confidence,
        "reasoning": classification.get("reasoning", ""),
        "features": classification.get("features", {}),
        "time_ms": step1_time
    }

    logger.info(f"[{request_id}] Step 1 complete: {category} ({confidence:.2f}) [{step1_time}ms]")

    # =========================================================================
    # STEP 2: CONVERT TO HTML (using Qwen VL on binary image)
    # =========================================================================
    logger.info(f"[{request_id}] Step 2: Converting to HTML (Qwen VL)")
    step2_start = time.time()

    html_result = convert_image_to_html(binary_base64)
    step2_time = int((time.time() - step2_start) * 1000)

    if html_result.get("status") == "error":
        logger.error(f"[{request_id}] HTML conversion failed: {html_result.get('error')}")
        return {
            **result,
            "status": "failed",
            "error": html_result.get("error"),
            "error_stage": "vision_to_html",
            "classification": result["steps"]["classification"]
        }

    html_content = html_result.get("html", "")

    result["steps"]["vision_to_html"] = {
        "status": "success",
        "model": html_result.get("model", "qwen/qwen3-vl-8b-instruct"),
        "html_length": len(html_content),
        "time_ms": step2_time
    }
    result["html"] = html_content

    logger.info(f"[{request_id}] Step 2 complete: {len(html_content)} chars [{step2_time}ms]")

    # =========================================================================
    # STEP 3: EXTRACT (using type-specific extractor)
    # =========================================================================
    logger.info(f"[{request_id}] Step 3: Extracting with {category} extractor")
    step3_start = time.time()

    # Get appropriate extractor
    extractor = EXTRACTORS.get(category)

    if not extractor:
        logger.warning(f"[{request_id}] Unknown category '{category}', using thermal extractor as fallback")
        extractor = extract_thermal
        category = "thermal_print"

    extraction_result = extractor(html_content)
    step3_time = int((time.time() - step3_start) * 1000)

    if extraction_result.get("status") == "error":
        logger.error(f"[{request_id}] Extraction failed: {extraction_result.get('error')}")
        result["steps"]["extraction"] = {
            "status": "failed",
            "error": extraction_result.get("error"),
            "extractor": extraction_result.get("extractor", category),
            "time_ms": step3_time
        }
        result["status"] = "partial"
        result["error"] = extraction_result.get("error")
        result["error_stage"] = "extraction"
    else:
        # Normalize extracted data to ensure consistent field structure
        raw_extracted = extraction_result.get("extracted_data", {})
        normalized_data = normalize_extraction_result(raw_extracted)
        field_summary = get_field_summary(normalized_data)

        result["steps"]["extraction"] = {
            "status": "success",
            "extractor": extraction_result.get("extractor", category),
            "time_ms": step3_time,
            "field_summary": field_summary
        }
        result["status"] = "success"
        result["extracted_data"] = normalized_data

        logger.info(f"[{request_id}] Step 3 complete: Extracted with {category} [{step3_time}ms] "
                   f"({field_summary['fields_with_values']}/{field_summary['total_fields']} fields)")

    # =========================================================================
    # FINAL RESULT
    # =========================================================================
    total_time = int((time.time() - total_start) * 1000)

    result["document_type"] = category
    result["classification_confidence"] = confidence
    result["total_time_ms"] = total_time
    result["timing"] = {
        "classification_ms": step1_time,
        "vision_to_html_ms": step2_time,
        "extraction_ms": step3_time,
        "total_ms": total_time
    }

    logger.info(f"[{request_id}] Pipeline complete: {result['status']} ({category}) [{total_time}ms]")

    return result
