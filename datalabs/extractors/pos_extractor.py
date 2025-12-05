"""
POS Receipt Extractor
Optimized for Point-of-Sale receipts (card payments, transaction details, merchant info).
"""

import json
import re
import requests
from loguru import logger

from ..config import settings


SYSTEM_PROMPT = """You extract data from POS (Point-of-Sale) receipt HTML. POS receipts have card payment details, transaction IDs, merchant info.

Rules:
- Extract ONLY text that exists in the HTML
- Use null for missing fields
- Copy text exactly as shown (including errors)
- Do NOT invent or guess any values
- Card numbers should be masked as shown (e.g., ****1234)"""

USER_PROMPT = """Extract data from this POS receipt HTML:

{html_content}

Return JSON:
{{
  "merchant": {{
    "name": "store/merchant name",
    "address": "address",
    "phone": "phone",
    "terminal_id": "terminal ID if shown"
  }},
  "transaction": {{
    "date": "transaction date",
    "time": "transaction time",
    "receipt_number": "receipt number",
    "order_number": "order number",
    "reference": "reference number"
  }},
  "items": [
    {{
      "description": "item name",
      "quantity": "qty",
      "price": "price"
    }}
  ],
  "amounts": {{
    "subtotal": "subtotal",
    "tax": "tax",
    "total": "total amount"
  }},
  "payment": {{
    "method": "card type (VISA/Mastercard/etc)",
    "card_number": "masked card number",
    "card_type": "Debit/Credit",
    "auth_code": "authorization code",
    "aid": "AID if shown",
    "contactless": "yes/no",
    "amount": "amount charged"
  }},
  "customer": {{
    "name": "customer name if shown",
    "account": "account number if shown"
  }}
}}

Return ONLY JSON, no explanation."""


def extract_pos(html_content: str) -> dict:
    """Extract data from POS receipt HTML."""
    logger.info("POS Extractor: Processing POS receipt")

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
            logger.error(f"POS Extractor API error: {response.status_code}")
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

        logger.info(f"POS Extractor: Extracted - merchant: {extracted.get('merchant', {}).get('name')}")

        return {
            "status": "success",
            "document_type": "pos_receipt",
            "extractor": "pos_extractor",
            "extracted_data": extracted
        }

    except Exception as e:
        logger.error(f"POS Extractor error: {e}")
        return {"status": "error", "error": str(e)}
