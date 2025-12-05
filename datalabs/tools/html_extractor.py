"""
HTML Extractor Tool
Takes HTML from vision model and extracts structured data using LLM.
"""

import json
import re
import requests
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool

from ..config import settings


# =============================================================================
# EXTRACTION PROMPTS
# =============================================================================

SYSTEM_PROMPT = """You extract data from HTML documents. Copy text exactly as shown. Use null for missing fields."""

USER_PROMPT_TEMPLATE = """Extract all data from this HTML invoice/receipt:

{html_content}

Return JSON with this exact structure. Copy text exactly as it appears. Use null if field not found:

{{
  "supplier": {{
    "name": "company/store name from header or h1/h2",
    "address": "supplier address if shown",
    "phone": "supplier phone number",
    "email": "supplier email",
    "vat_number": "VAT registration number"
  }},
  "customer": {{
    "name": "customer/buyer name (look for 'Collected by', 'Bill to', 'Customer')",
    "address": "customer full address",
    "phone": "customer phone",
    "email": "customer email"
  }},
  "document": {{
    "type": "Invoice/Receipt/Bill",
    "number": "invoice or receipt number",
    "date": "document date",
    "time": "time if shown",
    "order_number": "order number if shown",
    "reference": "reference number if shown"
  }},
  "items": [
    {{
      "description": "item name/description",
      "quantity": "qty as shown",
      "unit_price": "price each",
      "total": "line total"
    }}
  ],
  "amounts": {{
    "goods_total": "goods/subtotal before tax",
    "vat_amount": "VAT/tax amount",
    "vat_rate": "VAT rate percentage",
    "total": "final total amount"
  }},
  "payment": {{
    "method": "payment method (card type, cash, etc)",
    "card_type": "VISA/Mastercard/etc if card payment",
    "status": "paid/outstanding amount",
    "currency": "currency used"
  }}
}}

Return ONLY the JSON. No explanation."""


# =============================================================================
# TOOL DEFINITION
# =============================================================================

class HtmlExtractorInput(BaseModel):
    """Input for HTML extractor tool."""
    html_content: str = Field(description="HTML content to extract from")
    document_type: str = Field(default="receipt", description="Type of document")


class HtmlExtractorTool(BaseTool):
    """Extract structured data from HTML using LLM."""

    name: str = "html_extractor"
    description: str = "Extracts structured data from HTML document"
    args_schema: Type[BaseModel] = HtmlExtractorInput

    def _run(self, html_content: str, document_type: str = "receipt") -> str:
        """Extract data from HTML."""
        from loguru import logger

        try:
            headers = {
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:5050",
                "X-Title": "Document Processor"
            }

            user_prompt = USER_PROMPT_TEMPLATE.format(
                html_content=html_content
            )

            payload = {
                "model": "google/gemini-2.0-flash-001",  # Fast text model for extraction
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user",
                        "content": user_prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 4000
            }

            logger.info(f"HTML Extractor: Sending to LLM")

            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=60
            )

            if response.status_code != 200:
                error_msg = response.text
                logger.error(f"HTML Extractor API error: {response.status_code} - {error_msg}")
                return json.dumps({
                    "status": "error",
                    "error": f"API error {response.status_code}: {error_msg}"
                })

            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()

            logger.debug(f"HTML Extractor raw response: {content[:500]}...")

            # Clean up response - remove markdown code blocks if present
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                parts = content.split("```")
                if len(parts) >= 2:
                    content = parts[1].strip()
                    # Remove language identifier if present (e.g., "json\n")
                    if content.startswith("json"):
                        content = content[4:].strip()

            logger.info(f"HTML Extractor cleaned JSON: {content[:200]}...")

            # Try to find JSON object in content if direct parse fails
            try:
                extracted = json.loads(content)
            except json.JSONDecodeError:
                # Try to extract JSON from content
                json_match = re.search(r'\{[\s\S]*\}', content)
                if json_match:
                    content = json_match.group(0)
                    extracted = json.loads(content)
                else:
                    raise

            # If response has direct fields (supplier, customer, etc), wrap in extracted_data
            if "supplier" in extracted or "customer" in extracted or "items" in extracted:
                extracted_data = extracted
            else:
                extracted_data = extracted.get("extracted_data", extracted)

            logger.info(f"HTML Extractor: Extracted fields - supplier: {extracted_data.get('supplier', {}).get('name')}, customer: {extracted_data.get('customer', {}).get('name')}")

            return json.dumps({
                "status": "success",
                "model": "google/gemini-2.0-flash-001",
                "extracted_data": extracted_data
            })

        except json.JSONDecodeError as e:
            logger.error(f"HTML Extractor JSON error: {e}")
            return json.dumps({
                "status": "error",
                "error": f"Invalid JSON response: {str(e)}"
            })
        except Exception as e:
            logger.error(f"HTML Extractor error: {e}")
            return json.dumps({
                "status": "error",
                "error": str(e)
            })


# Global instance
html_extractor = HtmlExtractorTool()


def extract_from_html(html_content: str, document_type: str = "receipt") -> dict:
    """Extract structured data from HTML."""
    result = html_extractor._run(html_content=html_content, document_type=document_type)
    return json.loads(result)
