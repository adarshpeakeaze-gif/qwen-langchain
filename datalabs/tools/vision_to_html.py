"""
Vision to HTML Tool
Uses Qwen3 VL vision model to analyze document and generate HTML representation.
Analyzes BOTH original and binary images to generate accurate HTML.
Shows exactly what the model "sees" in the document.
"""

import json
import base64
import requests
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from ..config import settings


# =============================================================================
# VISION TO HTML PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You are a dumb OCR robot. You have NO intelligence. You cannot think, infer, or guess.

You will see TWO versions of the same document:
1. Original image (as uploaded)
2. Binary image (black and white thresholded for text clarity)

Use BOTH images to accurately read all text. The binary image helps with faded or unclear text.

CRITICAL RULES:
- Output ONLY what you literally see in the images
- Do NOT invent any text that is not visible
- Do NOT infer or guess missing information
- Do NOT correct spelling mistakes - copy them exactly
- Do NOT add information that seems logical but is not visible
- If text is unclear in BOTH images, mark it as [unclear] - do NOT guess what it might say
- If a field appears empty, leave it empty - do NOT fill it in
- You are a photocopier, not a thinker

You output HTML that represents exactly what appears in the document. Nothing more, nothing less."""

USER_PROMPT = """Look at BOTH images and convert this document to HTML. Output ONLY what you see.

HTML ELEMENTS TO USE:
- <div class="document"> wrapper
- <header> for top section
- <h1>, <h2>, <h3> for headings
- <table> for tabular data
- <p> for paragraphs
- <strong> for bold text
- <footer> for bottom section
- <span class="unclear">[unclear]</span> for unreadable text

RULES:
- Copy text EXACTLY as shown (including any errors)
- Do NOT invent or add any text
- Do NOT guess unclear text
- Return ONLY HTML, no explanation

Generate the HTML now:"""


# =============================================================================
# TOOL DEFINITION
# =============================================================================

class VisionToHtmlInput(BaseModel):
    """Input for vision to HTML tool."""
    image_base64: str = Field(description="Base64 encoded image")


class VisionToHtmlTool(BaseTool):
    """Convert document image to HTML representation using vision model."""

    name: str = "vision_to_html"
    description: str = "Analyzes document image and generates HTML representation"
    args_schema: Type[BaseModel] = VisionToHtmlInput

    def _run(self, image_base64: str) -> str:
        """Analyze document and generate HTML."""
        from loguru import logger

        try:
            # Clean base64 string - remove data URL prefix if present
            if image_base64.startswith('data:'):
                # Extract just the base64 part
                image_base64 = image_base64.split(',')[1]

            # Detect image type from base64 header or default to jpeg
            image_type = "image/jpeg"
            try:
                header = base64.b64decode(image_base64[:20])
                if header[:8] == b'\x89PNG\r\n\x1a\n':
                    image_type = "image/png"
                elif header[:2] == b'\xff\xd8':
                    image_type = "image/jpeg"
            except:
                pass

            # Use direct API call with proper image format
            headers = {
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5050",
                "X-Title": "Document Processor"
            }

            # Build the request payload with system + user prompts
            payload = {
                "model": "qwen/qwen3-vl-8b-instruct",
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": USER_PROMPT
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{image_type};base64,{image_base64}"
                                }
                            }
                        ]
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 4000
            }

            logger.info(f"Vision to HTML: Sending request to OpenRouter (image type: {image_type})")

            # Retry logic for API reliability
            max_retries = 3
            last_error = None

            for attempt in range(max_retries):
                try:
                    response = requests.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=300  # Increased timeout to 5 minutes
                    )
                    break  # Success, exit retry loop
                except requests.exceptions.RequestException as e:
                    last_error = e
                    logger.warning(f"Vision to HTML: Attempt {attempt + 1}/{max_retries} failed: {e}")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(2)  # Wait before retry
                    continue
            else:
                # All retries failed
                raise last_error or Exception("All retry attempts failed")

            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"Vision to HTML API error: {response.status_code} - {error_msg}")
                return json.dumps({
                    "status": "error",
                    "error": f"API error {response.status_code}: {error_msg}",
                    "html": f'<div class="document error"><p>API Error: {error_msg}</p></div>'
                })

            result = response.json()
            html_content = result["choices"][0]["message"]["content"].strip()

            # Clean up response - remove markdown code blocks if present
            if "```html" in html_content:
                html_content = html_content.split("```html")[1].split("```")[0].strip()
            elif "```" in html_content:
                html_content = html_content.split("```")[1].split("```")[0].strip()

            # Wrap in container with default styles if not already wrapped
            if not html_content.startswith('<div class="document'):
                html_content = f'<div class="document">{html_content}</div>'

            logger.info(f"Vision to HTML: Generated {len(html_content)} chars")

            return json.dumps({
                "status": "success",
                "html": html_content,
                "model": "qwen/qwen3-vl-8b-instruct",
                "char_count": len(html_content)
            })

        except Exception as e:
            logger.error(f"Vision to HTML error: {e}")
            return json.dumps({
                "status": "error",
                "error": str(e),
                "html": f'<div class="document error"><p>Error: {str(e)}</p></div>'
            })


# Global instance
vision_to_html = VisionToHtmlTool()


# Convenience function
def convert_image_to_html(image_base64: str) -> dict:
    """Convert document image to HTML representation."""
    result = vision_to_html._run(image_base64=image_base64)
    return json.loads(result)
