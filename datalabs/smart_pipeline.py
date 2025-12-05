"""
Smart Document Pipeline
Classify document type → Extract with built-in guardrails
No external OCR needed - uses vision LLM directly on images.
Guardrails are now built into the extraction prompt (single LLM call).
"""

import uuid
import json
import time
from typing import Dict, Any, TypedDict, Literal

from langgraph.graph import StateGraph, END
from loguru import logger

from .tools import focused_classifier, focused_extractor
from .utils import normalize_extraction_result, get_field_summary


# =============================================================================
# STATE DEFINITION
# =============================================================================

class SmartPipelineState(TypedDict):
    """State for smart classification pipeline."""

    # Request Info
    request_id: str
    filename: str
    image_base64: str

    # Classification Results
    document_type: str
    classification_confidence: float
    characteristics: dict
    classification_reasoning: str

    # Extraction Results (includes validation from built-in guardrails)
    extracted_data: dict
    extraction_type: str

    # Status
    status: str
    current_stage: str

    # Timing (in ms)
    classification_time_ms: int
    extraction_time_ms: int
    total_time_ms: int

    # Errors
    error: str
    error_stage: str


# =============================================================================
# NODE FUNCTIONS
# =============================================================================

def classify_node(state: SmartPipelineState) -> SmartPipelineState:
    """
    Classification Node - Determine document type using focused classifier.
    Uses smaller prompt for efficiency.
    """
    request_id = state["request_id"]
    logger.info(f"[{request_id}] Classify Node: Starting (focused)")
    start_time = time.time()

    try:
        state["status"] = "classifying"
        state["current_stage"] = "classification"

        # Call focused classifier
        result = focused_classifier._run(image_base64=state["image_base64"])
        parsed = json.loads(result)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Store classification results
        state["document_type"] = parsed.get("document_type", "unknown")
        state["classification_confidence"] = parsed.get("confidence", 0.0)
        state["characteristics"] = {
            "layout": parsed.get("layout", "unknown"),
            "has_letterhead": parsed.get("has_letterhead", False),
            "has_table": parsed.get("has_table", False),
            "is_handwritten": parsed.get("is_handwritten", False),
            "detected_keywords": parsed.get("detected_keywords", [])
        }
        state["classification_reasoning"] = parsed.get("reason", "")
        state["classification_time_ms"] = elapsed_ms
        state["status"] = "classified"

        logger.info(f"[{request_id}] Classify Node: {state['document_type']} "
                   f"(confidence: {state['classification_confidence']:.2f}) [{elapsed_ms}ms]")

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        state["error"] = str(e)
        state["error_stage"] = "classification"
        state["status"] = "failed"
        state["classification_time_ms"] = elapsed_ms
        logger.error(f"[{request_id}] Classify Node: Failed - {e}")

    return state


def extract_node(state: SmartPipelineState) -> SmartPipelineState:
    """
    Extraction Node - Extract fields with built-in guardrails validation.
    Single LLM call that extracts AND validates in one pass.
    """
    request_id = state["request_id"]
    logger.info(f"[{request_id}] Extract Node: Starting (type: {state['document_type']}, with guardrails)")
    start_time = time.time()

    try:
        # Skip if classification failed
        if state.get("status") == "failed":
            logger.warning(f"[{request_id}] Extract Node: Skipping (previous failure)")
            return state

        state["status"] = "extracting"
        state["current_stage"] = "extraction"

        # Call focused extractor with document type (includes guardrails in prompt)
        result = focused_extractor._run(
            image_base64=state["image_base64"],
            document_type=state["document_type"]
        )
        parsed = json.loads(result)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Normalize extraction results to ensure consistent field structure
        # Each field will have: value, confidence, isInferred
        normalized_data = normalize_extraction_result(parsed)
        field_summary = get_field_summary(normalized_data)

        # Store extraction results (normalized with consistent structure)
        state["extracted_data"] = normalized_data
        state["extraction_type"] = state["document_type"]
        state["extraction_time_ms"] = elapsed_ms
        state["total_time_ms"] = (
            state.get("classification_time_ms", 0) + elapsed_ms
        )
        state["status"] = "complete"

        # Log summary
        warnings = parsed.get("warnings", [])
        validation = parsed.get("validation", {})
        issues = validation.get("issues_found", 0)
        reconciles = validation.get("amounts_reconcile", "N/A")

        logger.info(f"[{request_id}] Extract Node: Complete [{elapsed_ms}ms, "
                   f"{issues} issues, {len(warnings)} warnings, amounts_reconcile={reconciles}, "
                   f"{field_summary['fields_with_values']}/{field_summary['total_fields']} fields]")

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        state["error"] = str(e)
        state["error_stage"] = "extraction"
        state["status"] = "failed"
        state["extraction_time_ms"] = elapsed_ms
        logger.error(f"[{request_id}] Extract Node: Failed - {e}")

    return state


# =============================================================================
# ROUTING
# =============================================================================

def should_continue_after_classify(state: SmartPipelineState) -> Literal["extract", "end"]:
    """Determine next step after classification."""
    if state.get("status") == "failed":
        return "end"
    return "extract"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def create_smart_pipeline() -> StateGraph:
    """
    Create the smart document processing pipeline.

    Pipeline: Classify (LLM #1) -> Extract+Validate (LLM #2) -> END

    Guardrails are built into the extraction prompt, so only 2 LLM calls needed.
    """
    workflow = StateGraph(SmartPipelineState)

    # Add nodes (only 2 now - guardrails is built into extract)
    workflow.add_node("classify", classify_node)
    workflow.add_node("extract", extract_node)

    # Set entry point
    workflow.set_entry_point("classify")

    # Add edges
    workflow.add_conditional_edges(
        "classify",
        should_continue_after_classify,
        {
            "extract": "extract",
            "end": END
        }
    )

    # Extract goes directly to END (guardrails built-in)
    workflow.add_edge("extract", END)

    return workflow.compile()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

smart_pipeline = create_smart_pipeline()


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def process_document_smart(
    image_base64: str,
    filename: str = "document",
    request_id: str = None
) -> Dict[str, Any]:
    """
    Process document with smart classification and extraction.

    Args:
        image_base64: Base64 encoded image
        filename: Original filename
        request_id: Optional request ID

    Returns:
        Processing result with classification and extraction
    """
    if not request_id:
        request_id = str(uuid.uuid4())[:8]

    logger.info(f"[{request_id}] Starting smart pipeline for: {filename}")

    # Initialize state
    initial_state: SmartPipelineState = {
        "request_id": request_id,
        "filename": filename,
        "image_base64": image_base64,
        "document_type": "",
        "classification_confidence": 0.0,
        "characteristics": {},
        "classification_reasoning": "",
        "extracted_data": {},
        "extraction_type": "",
        "status": "pending",
        "current_stage": "init",
        "classification_time_ms": 0,
        "extraction_time_ms": 0,
        "total_time_ms": 0,
        "error": "",
        "error_stage": ""
    }

    try:
        # Run pipeline
        final_state = smart_pipeline.invoke(initial_state)

        logger.info(f"[{request_id}] Pipeline complete: {final_state['status']}")

        return dict(final_state)

    except Exception as e:
        logger.error(f"[{request_id}] Pipeline error: {e}")
        return {
            **initial_state,
            "status": "failed",
            "error": str(e),
            "error_stage": "pipeline"
        }


# =============================================================================
# CLASSIFY ONLY (for UI preview)
# =============================================================================

def classify_document(
    image_base64: str,
    request_id: str = None
) -> Dict[str, Any]:
    """
    Classify document type only (without extraction).
    Useful for showing classification in UI before extraction.

    Args:
        image_base64: Base64 encoded image
        request_id: Optional request ID

    Returns:
        Classification result
    """
    if not request_id:
        request_id = str(uuid.uuid4())[:8]

    logger.info(f"[{request_id}] Classifying document (focused)...")
    start_time = time.time()

    try:
        result = focused_classifier._run(image_base64=image_base64)
        parsed = json.loads(result)

        elapsed_ms = int((time.time() - start_time) * 1000)

        return {
            "status": "success",
            "document_type": parsed.get("document_type", "unknown"),
            "confidence": parsed.get("confidence", 0.0),
            "characteristics": {
                "layout": parsed.get("layout", "unknown"),
                "has_letterhead": parsed.get("has_letterhead", False),
                "has_table": parsed.get("has_table", False),
                "is_handwritten": parsed.get("is_handwritten", False),
                "detected_keywords": parsed.get("detected_keywords", [])
            },
            "reasoning": parsed.get("reason", ""),
            "time_ms": elapsed_ms
        }

    except Exception as e:
        logger.error(f"[{request_id}] Classification error: {e}")
        return {
            "status": "failed",
            "error": str(e)
        }
