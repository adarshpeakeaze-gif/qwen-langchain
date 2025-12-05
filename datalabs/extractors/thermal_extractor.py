"""
Thermal Receipt Extractor
Optimized for thermal printed receipts (narrow, monospace font, faded text).
"""

import json
import re
import requests
from loguru import logger

from ..config import settings


SYSTEM_PROMPT = """You extract data from thermal receipt HTML. Thermal receipts are narrow, use monospace font, may have faded text.

Rules:
- Extract ONLY text that exists in the HTML
- Use null for missing fields
- Copy text exactly as shown (including errors)
- Do NOT invent or guess any values"""

USER_PROMPT = """Extract data from this thermal receipt HTML:

{html_content}

Return JSON:
{{
  "supplier": {{
    "name": "store name from header",
    "address": "store address",
    "phone": "store phone"
  }},
  "document": {{
    "date": "date",
    "time": "time",
    "receipt_number": "receipt/transaction number"
  }},
  "items": [
    {{"description": "item name", "quantity": "qty", "price": "price"}}
  ],
  "amounts": {{
    "subtotal": "subtotal",
    "tax": "tax amount",
    "total": "total"
  }},
  "payment": {{
    "method": "cash/card",
    "amount_paid": "amount paid",
    "change": "change given"
  }}
}}

Return ONLY JSON, no explanation."""


def extract_thermal(html_content: str) -> dict:
    """Extract data from thermal receipt HTML."""
    logger.info("Thermal Extractor: Processing thermal receipt")

    try:
        headers = {
            "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:5050",
            "X-Title": "Document Processor"
        }

        payload = {
            "model": "openai/gpt-4.1-nano",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT.format(html_content=html_content)}
            ],
            "temperature": 0.1,
            "max_tokens": 4000
        }

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=60
        )

        if response.status_code != 200:
            logger.error(f"Thermal Extractor API error: {response.status_code}")
            return {"status": "error", "error": response.text}

        content = response.json()["choices"][0]["message"]["content"].strip()

        # Clean JSON
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        try:
            extracted = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                extracted = json.loads(json_match.group(0))
            else:
                return {"status": "error", "error": "Invalid JSON response"}

        logger.info(f"Thermal Extractor: Extracted - {extracted.get('supplier', {}).get('name')}")

        return {
            "status": "success",
            "document_type": "thermal_receipt",
            "extractor": "thermal_extractor",
            "extracted_data": extracted
        }

    except Exception as e:
        logger.error(f"Thermal Extractor error: {e}")
        return {"status": "error", "error": str(e)}
