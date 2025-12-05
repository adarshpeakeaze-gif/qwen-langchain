"""
Handprint/Handwritten Document Extractor
Optimized for handwritten documents (variable handwriting, informal layout).
"""

import json
import re
import requests
from loguru import logger

from ..config import settings


SYSTEM_PROMPT = """You extract data from handwritten document HTML. Handwritten documents have variable text quality, informal layout.

Rules:
- Extract ONLY text that exists in the HTML
- Use null for missing fields
- Copy text exactly as shown (including errors, unclear marks)
- Mark unclear text as shown - do NOT guess
- Do NOT invent or guess any values
- Handwriting may have spelling errors - copy as-is"""

USER_PROMPT = """Extract data from this handwritten document HTML:

{html_content}

Return JSON:
{{
  "document_info": {{
    "type": "receipt/note/bill/form",
    "date": "date if shown",
    "title": "title or heading if any"
  }},
  "parties": {{
    "from": "who wrote/issued this",
    "to": "recipient if shown"
  }},
  "items": [
    {{
      "description": "item or line of text",
      "amount": "amount if shown"
    }}
  ],
  "amounts": {{
    "total": "total if shown",
    "paid": "amount paid if shown",
    "balance": "balance if shown"
  }},
  "notes": "any additional text or notes",
  "unclear_sections": ["list any text marked as unclear"]
}}

Return ONLY JSON, no explanation."""


def extract_handprint(html_content: str) -> dict:
    """Extract data from handwritten document HTML."""
    logger.info("Handprint Extractor: Processing handwritten document")

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
            logger.error(f"Handprint Extractor API error: {response.status_code}")
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

        logger.info(f"Handprint Extractor: Extracted - type: {extracted.get('document_info', {}).get('type')}")

        return {
            "status": "success",
            "document_type": "handprint",
            "extractor": "handprint_extractor",
            "extracted_data": extracted
        }

    except Exception as e:
        logger.error(f"Handprint Extractor error: {e}")
        return {"status": "error", "error": str(e)}
