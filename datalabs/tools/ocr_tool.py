"""
Datalabs - OCR Tool
LangChain tool for converting documents to HTML using Datalab OCR API.
"""

import asyncio
import base64
from pathlib import Path
from typing import Type

import aiohttp
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from loguru import logger

from ..config import settings


# =============================================================================
# TOOL INPUT SCHEMA
# =============================================================================

class OCRToolInput(BaseModel):
    """Input schema for the OCR tool."""
    file_content_base64: str = Field(
        description="Base64 encoded file content (PDF, JPEG, PNG, etc.)"
    )
    filename: str = Field(
        description="Original filename with extension (e.g., 'invoice.pdf')"
    )
    output_format: str = Field(
        default="html",
        description="Output format: 'html', 'markdown', or 'json'"
    )
    force_ocr: bool = Field(
        default=False,
        description="Force OCR on all pages (slower but more accurate)"
    )


# =============================================================================
# OCR TOOL IMPLEMENTATION
# =============================================================================

class DatalabOCRTool(BaseTool):
    """
    LangChain tool for OCR processing using Datalab API.

    Converts PDF, JPEG, PNG, and other image formats to structured HTML
    using Datalab's Marker OCR service.

    Features:
    - Supports multiple document formats
    - Returns structured HTML with layout preservation
    - Handles multi-page documents
    - Configurable OCR options
    """

    name: str = "datalab_ocr"
    description: str = """
    Convert a document (PDF, JPEG, PNG, TIFF, etc.) to HTML using OCR.

    Use this tool when you need to:
    - Extract text from a scanned document
    - Convert a PDF to searchable HTML
    - Process an image containing text

    Input: Base64 encoded file content and filename
    Output: HTML string with the document content
    """
    args_schema: Type[BaseModel] = OCRToolInput

    def _get_mime_type(self, filename: str) -> str:
        """Determine MIME type from filename."""
        ext = Path(filename).suffix.lower()
        mime_map = {
            '.pdf': 'application/pdf',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.tiff': 'image/tiff',
            '.tif': 'image/tiff',
            '.bmp': 'image/bmp',
            '.webp': 'image/webp',
            '.gif': 'image/gif'
        }
        return mime_map.get(ext, 'application/octet-stream')

    async def _call_datalab_api(
        self,
        file_content: bytes,
        filename: str,
        output_format: str,
        force_ocr: bool
    ) -> dict:
        """
        Call Datalab API and poll for results.

        Args:
            file_content: Raw file bytes
            filename: Original filename
            output_format: Desired output format
            force_ocr: Whether to force OCR

        Returns:
            Dict with content, page_count, and metadata
        """
        mime_type = self._get_mime_type(filename)
        headers = {"X-Api-Key": settings.DATALAB_API_KEY}

        # Prepare form data
        form_data = aiohttp.FormData()
        form_data.add_field('file', file_content, filename=filename, content_type=mime_type)
        form_data.add_field('output_format', output_format)
        form_data.add_field('force_ocr', str(force_ocr).lower())
        form_data.add_field('use_llm', str(settings.DATALAB_USE_LLM).lower())

        submit_url = f"{settings.DATALAB_BASE_URL}/marker"

        async with aiohttp.ClientSession() as session:
            # Submit document
            logger.info(f"[OCR Tool] Submitting document: {filename}")

            async with session.post(submit_url, headers=headers, data=form_data) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"Datalab API error ({response.status}): {error_text}")

                submit_result = await response.json()

            if not submit_result.get("success", False):
                raise Exception(f"Datalab submit failed: {submit_result.get('error', 'Unknown')}")

            check_url = submit_result.get("request_check_url")
            request_id = submit_result.get("request_id")
            logger.info(f"[OCR Tool] Request ID: {request_id}")

            # Poll for results
            for i in range(settings.DATALAB_MAX_POLLS):
                await asyncio.sleep(settings.DATALAB_POLL_INTERVAL)

                async with session.get(check_url, headers=headers) as response:
                    if response.status != 200:
                        continue

                    result = await response.json()
                    status = result.get("status", "")

                    if status == "complete":
                        logger.info(f"[OCR Tool] Processing complete: {request_id}")
                        return {
                            "content": result.get(output_format, result.get("html", "")),
                            "page_count": result.get("page_count", 0),
                            "metadata": result.get("meta", {}),
                            "request_id": request_id
                        }

                    elif status == "failed":
                        raise Exception(f"OCR failed: {result.get('error', 'Unknown')}")

                    logger.debug(f"[OCR Tool] Status: {status} (poll {i+1})")

            raise Exception(f"OCR timeout after {settings.DATALAB_MAX_POLLS * settings.DATALAB_POLL_INTERVAL}s")

    def _run(
        self,
        file_content_base64: str,
        filename: str,
        output_format: str = "html",
        force_ocr: bool = False
    ) -> str:
        """
        Synchronous execution of OCR tool.

        Args:
            file_content_base64: Base64 encoded file content
            filename: Original filename
            output_format: Output format (html, markdown, json)
            force_ocr: Force OCR on all pages

        Returns:
            HTML/Markdown/JSON string with document content
        """
        logger.info(f"[OCR Tool] Starting OCR for: {filename}")

        # Validate API key
        if not settings.DATALAB_API_KEY:
            raise ValueError("DATALAB_API_KEY not configured")

        # Decode file content
        try:
            file_content = base64.b64decode(file_content_base64)
        except Exception as e:
            raise ValueError(f"Invalid base64 content: {e}")

        # Run async OCR
        result = asyncio.run(self._call_datalab_api(
            file_content=file_content,
            filename=filename,
            output_format=output_format,
            force_ocr=force_ocr
        ))

        logger.info(f"[OCR Tool] Complete: {result['page_count']} pages")

        return result["content"]

    async def _arun(
        self,
        file_content_base64: str,
        filename: str,
        output_format: str = "html",
        force_ocr: bool = False
    ) -> str:
        """
        Asynchronous execution of OCR tool.
        """
        logger.info(f"[OCR Tool] Starting async OCR for: {filename}")

        if not settings.DATALAB_API_KEY:
            raise ValueError("DATALAB_API_KEY not configured")

        file_content = base64.b64decode(file_content_base64)

        result = await self._call_datalab_api(
            file_content=file_content,
            filename=filename,
            output_format=output_format,
            force_ocr=force_ocr
        )

        return result["content"]


# =============================================================================
# TOOL INSTANCE
# =============================================================================

# Create a singleton instance for easy import
ocr_tool = DatalabOCRTool()
