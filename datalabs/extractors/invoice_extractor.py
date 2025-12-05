"""
A4 Invoice Extractor
Optimized for formal A4 invoices (letterhead, structured layout, multiple sections).
"""

import json
import re
import requests
from loguru import logger

from ..config import settings


SYSTEM_PROMPT = """You extract data from A4 invoice HTML. A4 invoices have formal layout, letterhead, supplier and customer sections.

Rules:
- Extract ONLY text that exists in the HTML
- Use null for missing fields
- Copy text exactly as shown (including errors)
- Do NOT invent or guess any values"""

USER_PROMPT = """Extract data from this A4 invoice HTML:

{html_content}

Return JSON:
{{
  "supplier": {{
    "name": "company name",
    "address": "full address",
    "phone": "phone",
    "email": "email",
    "vat_number": "VAT/tax registration number",
    "company_number": "company registration number"
  }},
  "customer": {{
    "name": "customer/buyer name",
    "address": "customer address",
    "phone": "customer phone",
    "email": "customer email",
    "account_number": "customer account number"
  }},
  "document": {{
    "type": "Invoice/Credit Note/Quote",
    "number": "invoice number",
    "date": "invoice date",
    "due_date": "payment due date",
    "order_number": "order/PO number",
    "reference": "reference"
  }},
  "items": [
    {{
      "code": "product code",
      "description": "item description",
      "quantity": "quantity",
      "unit_price": "unit price",
      "vat_rate": "VAT rate",
      "total": "line total"
    }}
  ],
  "amounts": {{
    "subtotal": "net total before tax",
    "vat_amount": "total VAT",
    "discount": "discount if any",
    "total": "gross total"
  }},
  "payment": {{
    "method": "payment method",
    "terms": "payment terms",
    "bank_details": "bank account info if shown",
    "status": "paid/unpaid/outstanding amount"
  }}
}}

Return ONLY JSON, no explanation."""


def extract_invoice(html_content: str) -> dict:
    """Extract data from A4 invoice HTML."""
    logger.info("Invoice Extractor: Processing A4 invoice")

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
            logger.error(f"Invoice Extractor API error: {response.status_code}")
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

        logger.info(f"Invoice Extractor: Extracted - supplier: {extracted.get('supplier', {}).get('name')}, customer: {extracted.get('customer', {}).get('name')}")

        return {
            "status": "success",
            "document_type": "a4_invoice",
            "extractor": "invoice_extractor",
            "extracted_data": extracted
        }

    except Exception as e:
        logger.error(f"Invoice Extractor error: {e}")
        return {"status": "error", "error": str(e)}
