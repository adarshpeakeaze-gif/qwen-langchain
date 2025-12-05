"""
Document Classifier
Classifies documents using BOTH original and binary images for better accuracy.
Categories: thermal_print, a4_invoice, handprint, pos_receipt
"""

import json
import re
import base64
import requests
from loguru import logger

from .config import settings


SYSTEM_PROMPT = """You are a document classifier. You will see TWO versions of the same document:
1. Original image (as uploaded)
2. Binary image (black and white thresholded for text clarity)

Classify into ONE of these categories:
- thermal_print: Narrow thermal printed receipts (grocery, convenience stores)
- a4_invoice: Formal A4 invoices with letterhead, supplier/customer sections
- handprint: Handwritten documents, notes, manual receipts
- pos_receipt: Point-of-sale receipts with card payment details

Look at BOTH images to make the best classification decision."""

USER_PROMPT = """Look at both images and classify this document.

Analyze:
1. Document width/format (narrow thermal vs wide A4)
2. Font type (thermal/monospace vs printed vs handwritten)
3. Content structure (formal invoice vs receipt vs handwritten)
4. Payment info (card details suggest POS)

Return JSON only:
{{
  "category": "thermal_print|a4_invoice|handprint|pos_receipt",
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "features": {{
    "is_narrow": true/false,
    "has_letterhead": true/false,
    "is_handwritten": true/false,
    "has_card_payment": true/false
  }}
}}"""


def classify_document(original_base64: str, binary_base64: str) -> dict:
    """
    Classify document using both original and binary images.

    Args:
        original_base64: Original uploaded image
        binary_base64: Binary/thresholded image

    Returns:
        Classification result with category and confidence
    """
    logger.info("Document Classifier: Classifying with dual images")

    try:
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5050",
            "X-Title": "Document Processor"
        }

        # Prepare image URLs
        def format_image(img_base64):
            if img_base64.startswith('data:'):
                return img_base64
            return f"data:image/png;base64,{img_base64}"

        payload = {
            "model": "openai/gpt-4o",
            "messages": [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Image 1 - ORIGINAL:"},
                        {"type": "image_url", "image_url": {"url": format_image(original_base64)}},
                        {"type": "text", "text": "Image 2 - BINARY:"},
                        {"type": "image_url", "image_url": {"url": format_image(binary_base64)}},
                        {"type": "text", "text": USER_PROMPT}
                    ]
                }
            ],
            "temperature": 0.1,
            "max_tokens": 1000
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            logger.error(f"Classifier API error: {response.status_code} - {response.text}")
            return {
                "status": "error",
                "error": response.text,
                "category": "unknown"
            }

        content = response.json()["choices"][0]["message"]["content"].strip()

        logger.info(f"Classifier raw response: {content[:300]}...")

        # Clean JSON - remove markdown blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                content = parts[1].strip()
                if content.startswith("json"):
                    content = content[4:].strip()

        # Try to parse JSON
        result = None
        try:
            result = json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON from content
            json_match = re.search(r'\{[^{}]*"category"[^{}]*\}', content)
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                except:
                    pass

            if not result:
                # Try broader regex
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    try:
                        result = json.loads(json_match.group(0))
                    except:
                        pass

        if not result:
            logger.error(f"Classifier: Could not parse JSON - {content[:300]}")
            # Fallback: try to extract category from text
            content_lower = content.lower()
            if "thermal" in content_lower:
                return {"status": "success", "category": "thermal_print", "confidence": 0.5, "reasoning": "Fallback detection", "features": {}}
            elif "invoice" in content_lower or "a4" in content_lower:
                return {"status": "success", "category": "a4_invoice", "confidence": 0.5, "reasoning": "Fallback detection", "features": {}}
            elif "handwrit" in content_lower or "handprint" in content_lower:
                return {"status": "success", "category": "handprint", "confidence": 0.5, "reasoning": "Fallback detection", "features": {}}
            elif "pos" in content_lower or "card" in content_lower:
                return {"status": "success", "category": "pos_receipt", "confidence": 0.5, "reasoning": "Fallback detection", "features": {}}
            else:
                return {"status": "success", "category": "thermal_print", "confidence": 0.3, "reasoning": "Default fallback", "features": {}}

        category = result.get("category", "unknown")
        confidence = result.get("confidence", 0.0)

        logger.info(f"Document Classifier: {category} (confidence: {confidence})")

        return {
            "status": "success",
            "category": category,
            "confidence": confidence,
            "reasoning": result.get("reasoning", ""),
            "features": result.get("features", {})
        }

    except Exception as e:
        logger.error(f"Document Classifier error: {e}")
        return {
            "status": "error",
            "error": str(e),
            "category": "unknown"
        }
