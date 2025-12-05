"""
Datalabs - LangGraph Workflow
Document processing pipeline using LangGraph with LangChain tools.
Updated for evidence-driven extraction with debug support.
"""

import uuid
import json
import base64
import time
from typing import Dict, Any, TypedDict, Literal, Optional
from pathlib import Path

from langgraph.graph import StateGraph, END
from loguru import logger

from .schemas import ProcessingStatus
from .tools import ocr_tool, extraction_tool, mapping_tool
from .logger import ProcessingLogger


# =============================================================================
# STATE DEFINITION
# =============================================================================

class GraphState(TypedDict):
    """State type for the processing graph."""

    # Request Info
    request_id: str
    filename: str
    file_content_base64: str
    mime_type: str

    # Debug Mode
    debug_mode: bool

    # Status Tracking
    status: str
    current_stage: str

    # OCR Output
    html_content: str
    ocr_page_count: int

    # Extraction Output (Raw from LLM)
    raw_extraction: dict
    at_a_glance: dict  # Quick summary for debugging

    # Mapping Output (Validated schema)
    mapped_output: dict
    validation_results: dict

    # Warnings and Debug Info
    warnings: list
    debug_info: dict

    # Timing
    ocr_time_ms: int
    extraction_time_ms: int
    mapping_time_ms: int
    total_time_ms: int

    # Errors
    error: str
    error_stage: str


# =============================================================================
# NODE FUNCTIONS (Using Tools)
# =============================================================================

def ocr_node(state: GraphState) -> GraphState:
    """
    OCR Node - Convert document to HTML using Datalab OCR Tool.
    """
    request_id = state["request_id"]
    logger.info(f"[{request_id}] OCR Node: Starting")
    start_time = time.time()

    try:
        state["status"] = ProcessingStatus.OCR_IN_PROGRESS.value
        state["current_stage"] = "ocr"

        # Call OCR Tool
        html_content = ocr_tool._run(
            file_content_base64=state["file_content_base64"],
            filename=state["filename"],
            output_format="html",
            force_ocr=False
        )

        elapsed_ms = int((time.time() - start_time) * 1000)
        state["html_content"] = html_content
        state["ocr_time_ms"] = elapsed_ms
        state["status"] = ProcessingStatus.OCR_COMPLETE.value

        # Debug info
        if state.get("debug_mode", True):
            state["debug_info"]["ocr"] = {
                "html_length": len(html_content),
                "time_ms": elapsed_ms
            }

        logger.info(f"[{request_id}] OCR Node: Complete ({elapsed_ms}ms, {len(html_content)} chars)")

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        state["error"] = str(e)
        state["error_stage"] = "ocr"
        state["status"] = ProcessingStatus.FAILED.value
        state["ocr_time_ms"] = elapsed_ms
        logger.error(f"[{request_id}] OCR Node: Failed - {e}")

    return state


def extraction_node(state: GraphState) -> GraphState:
    """
    Extraction Node - Extract entities using Entity Extraction Tool.
    Uses evidence-driven extraction with confidence scores.
    """
    request_id = state["request_id"]
    logger.info(f"[{request_id}] Extraction Node: Starting")
    start_time = time.time()

    try:
        # Skip if previous stage failed
        if state.get("status") == ProcessingStatus.FAILED.value:
            logger.warning(f"[{request_id}] Extraction Node: Skipping (previous failure)")
            return state

        state["status"] = ProcessingStatus.EXTRACTION_IN_PROGRESS.value
        state["current_stage"] = "extraction"

        # Call Extraction Tool with debug mode
        extraction_result = extraction_tool._run(
            html_content=state["html_content"],
            debug_mode=state.get("debug_mode", True)
        )

        # Parse result
        parsed = json.loads(extraction_result)

        elapsed_ms = int((time.time() - start_time) * 1000)

        # Store raw extraction and at-a-glance
        state["raw_extraction"] = parsed
        state["at_a_glance"] = parsed.get("at_a_glance", {})
        state["warnings"] = parsed.get("warnings", [])
        state["extraction_time_ms"] = elapsed_ms
        state["status"] = ProcessingStatus.EXTRACTION_COMPLETE.value

        # Debug info
        if state.get("debug_mode", True):
            at_a_glance = parsed.get("at_a_glance", {})
            state["debug_info"]["extraction"] = {
                "document_type": at_a_glance.get("document_type", "unknown"),
                "document_type_confidence": at_a_glance.get("document_type_confidence", 0),
                "visible_fields_count": len(at_a_glance.get("key_visible_fields", [])),
                "inferred_fields_count": len(at_a_glance.get("key_inferred_fields", [])),
                "missing_fields_count": len(at_a_glance.get("missing_critical_fields", [])),
                "warnings_count": len(parsed.get("warnings", [])),
                "time_ms": elapsed_ms
            }

        logger.info(f"[{request_id}] Extraction Node: Complete ({elapsed_ms}ms)")
        logger.info(f"[{request_id}] Document Type: {state['at_a_glance'].get('document_type', 'unknown')}")

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        state["error"] = str(e)
        state["error_stage"] = "extraction"
        state["status"] = ProcessingStatus.FAILED.value
        state["extraction_time_ms"] = elapsed_ms
        logger.error(f"[{request_id}] Extraction Node: Failed - {e}")

    return state


def mapping_node(state: GraphState) -> GraphState:
    """
    Mapping Node - Map to schema using Schema Mapping Tool.
    Deterministic mapping (no LLM) to Pydantic models.
    """
    request_id = state["request_id"]
    logger.info(f"[{request_id}] Mapping Node: Starting")
    start_time = time.time()

    try:
        # Skip if previous stage failed
        if state.get("status") == ProcessingStatus.FAILED.value:
            logger.warning(f"[{request_id}] Mapping Node: Skipping (previous failure)")
            return state

        state["status"] = ProcessingStatus.MAPPING_IN_PROGRESS.value
        state["current_stage"] = "mapping"

        # Call Mapping Tool
        mapping_result = mapping_tool._run(
            raw_extraction=json.dumps(state["raw_extraction"]),
            debug_mode=state.get("debug_mode", True)
        )

        # Parse result
        parsed = json.loads(mapping_result)

        elapsed_ms = int((time.time() - start_time) * 1000)

        if parsed.get("success", False):
            state["mapped_output"] = parsed.get("extraction", {})
            state["validation_results"] = parsed.get("extraction", {}).get("validation", {})
            state["status"] = ProcessingStatus.COMPLETE.value
        else:
            state["error"] = parsed.get("error", "Mapping failed")
            state["error_stage"] = "mapping"
            state["status"] = ProcessingStatus.FAILED.value

        state["mapping_time_ms"] = elapsed_ms
        state["total_time_ms"] = (
            state.get("ocr_time_ms", 0) +
            state.get("extraction_time_ms", 0) +
            elapsed_ms
        )

        # Debug info
        if state.get("debug_mode", True):
            state["debug_info"]["mapping"] = {
                "success": parsed.get("success", False),
                "time_ms": elapsed_ms
            }

        logger.info(f"[{request_id}] Mapping Node: Complete ({elapsed_ms}ms)")

    except Exception as e:
        elapsed_ms = int((time.time() - start_time) * 1000)
        state["error"] = str(e)
        state["error_stage"] = "mapping"
        state["status"] = ProcessingStatus.FAILED.value
        state["mapping_time_ms"] = elapsed_ms
        logger.error(f"[{request_id}] Mapping Node: Failed - {e}")

    return state


# =============================================================================
# ROUTING FUNCTIONS
# =============================================================================

def should_continue_after_ocr(state: GraphState) -> Literal["extraction", "end"]:
    """Determine next step after OCR."""
    if state.get("status") == ProcessingStatus.FAILED.value:
        return "end"
    return "extraction"


def should_continue_after_extraction(state: GraphState) -> Literal["mapping", "end"]:
    """Determine next step after extraction."""
    if state.get("status") == ProcessingStatus.FAILED.value:
        return "end"
    return "mapping"


def should_continue_after_mapping(state: GraphState) -> Literal["end"]:
    """Always end after mapping."""
    return "end"


# =============================================================================
# GRAPH BUILDER
# =============================================================================

def create_processing_graph() -> StateGraph:
    """
    Create the document processing graph with tools.

    Pipeline:
        OCR (Datalab) -> Extraction (LLM) -> Mapping (Deterministic) -> END

    Returns:
        Compiled LangGraph workflow
    """
    # Create graph
    workflow = StateGraph(GraphState)

    # Add nodes
    workflow.add_node("ocr", ocr_node)
    workflow.add_node("extraction", extraction_node)
    workflow.add_node("mapping", mapping_node)

    # Set entry point
    workflow.set_entry_point("ocr")

    # Add conditional edges
    workflow.add_conditional_edges(
        "ocr",
        should_continue_after_ocr,
        {
            "extraction": "extraction",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "extraction",
        should_continue_after_extraction,
        {
            "mapping": "mapping",
            "end": END
        }
    )

    workflow.add_conditional_edges(
        "mapping",
        should_continue_after_mapping,
        {
            "end": END
        }
    )

    return workflow.compile()


# =============================================================================
# GLOBAL GRAPH INSTANCE
# =============================================================================

processing_graph = create_processing_graph()


# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_document(
    file_content: bytes,
    filename: str,
    mime_type: str = None,
    request_id: str = None,
    debug_mode: bool = True
) -> Dict[str, Any]:
    """
    Process a document through the full pipeline.

    Args:
        file_content: Raw file bytes
        filename: Original filename
        mime_type: Optional MIME type override
        request_id: Optional request ID (generated if not provided)
        debug_mode: Enable detailed debug logging (default True)

    Returns:
        Processing result dictionary with extraction output
    """
    # Generate request ID
    if not request_id:
        request_id = str(uuid.uuid4())[:8]

    # Determine MIME type
    if not mime_type:
        ext = Path(filename).suffix.lower()
        mime_map = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tiff': 'image/tiff',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp'
        }
        mime_type = mime_map.get(ext, 'application/octet-stream')

    # Initialize logger
    proc_logger = ProcessingLogger(request_id, filename)
    proc_logger.log("INIT", "Starting document processing", {
        "filename": filename,
        "mime_type": mime_type,
        "size_bytes": len(file_content),
        "debug_mode": debug_mode
    })

    # Encode file content
    file_content_base64 = base64.b64encode(file_content).decode('utf-8')

    # Initialize state
    initial_state: GraphState = {
        "request_id": request_id,
        "filename": filename,
        "file_content_base64": file_content_base64,
        "mime_type": mime_type,
        "debug_mode": debug_mode,
        "status": ProcessingStatus.PENDING.value,
        "current_stage": "init",
        "html_content": "",
        "ocr_page_count": 0,
        "raw_extraction": {},
        "at_a_glance": {},
        "mapped_output": {},
        "validation_results": {},
        "warnings": [],
        "debug_info": {},
        "ocr_time_ms": 0,
        "extraction_time_ms": 0,
        "mapping_time_ms": 0,
        "total_time_ms": 0,
        "error": "",
        "error_stage": ""
    }

    try:
        # Run the graph
        logger.info(f"[{request_id}] Starting pipeline for: {filename}")
        final_state = processing_graph.invoke(initial_state)

        # Log completion
        success = final_state.get("status") == ProcessingStatus.COMPLETE.value
        proc_logger.complete(
            success=success,
            result=final_state.get("mapped_output"),
            error=final_state.get("error") if not success else None
        )

        # Log at-a-glance summary
        at_a_glance = final_state.get("at_a_glance", {})
        if at_a_glance:
            proc_logger.log("SUMMARY", at_a_glance.get("quick_summary", "No summary"), {
                "document_type": at_a_glance.get("document_type"),
                "confidence": at_a_glance.get("document_type_confidence"),
                "visible_fields": at_a_glance.get("key_visible_fields", []),
                "inferred_fields": at_a_glance.get("key_inferred_fields", []),
                "missing_fields": at_a_glance.get("missing_critical_fields", [])
            })

        # Log warnings
        warnings = final_state.get("warnings", [])
        if warnings:
            proc_logger.log("WARNINGS", f"{len(warnings)} warnings", {"warnings": warnings})

        return dict(final_state)

    except Exception as e:
        logger.error(f"[{request_id}] Pipeline error: {e}")
        proc_logger.complete(success=False, error=str(e))

        return {
            **initial_state,
            "status": ProcessingStatus.FAILED.value,
            "error": str(e),
            "error_stage": "pipeline"
        }


async def process_document_async(
    file_content: bytes,
    filename: str,
    mime_type: str = None,
    request_id: str = None,
    debug_mode: bool = True
) -> Dict[str, Any]:
    """
    Async wrapper for process_document.
    """
    import asyncio
    return await asyncio.to_thread(
        process_document,
        file_content,
        filename,
        mime_type,
        request_id,
        debug_mode
    )


# =============================================================================
# TOOL LIST FOR AGENTS
# =============================================================================

def get_tools():
    """
    Get list of all available tools.

    Returns:
        List of LangChain tools for use with agents
    """
    return [ocr_tool, extraction_tool, mapping_tool]
