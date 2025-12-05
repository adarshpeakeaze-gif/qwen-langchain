"""
Focused Field Extractor with Built-in Guardrails
Extracts document fields AND validates in a single LLM call.
No separate guardrails step needed.
"""

import json
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..config import settings


# =============================================================================
# EXTRACTION + GUARDRAILS PROMPT (Combined)
# =============================================================================

EXTRACTION_PROMPT = """You are an expert document extraction agent with strict validation rules.
Extract fields from this {document_type} document AND validate as you extract.

═══════════════════════════════════════════════════════════════════════════════
CRITICAL EXTRACTION RULES - FOLLOW EXACTLY
═══════════════════════════════════════════════════════════════════════════════

## RULE 1: ZERO HALLUCINATIONS
- Extract ONLY what you can SEE in the image
- If a field is not visible → value = null, confidence = 0
- NEVER guess, assume, or infer from external knowledge
- Partial text stays partial: "SWINDO" → "SWINDO" (NOT "Swindon")

## RULE 2: SUPPLIER NAME VALIDATION
- "STORE# 5793", "BRANCH# 123", "LOCATION# 456" are STORE IDs, NOT supplier names
  → Look for actual business name in header/footer/logo
- Full addresses with Road/Street/postal codes are NOT supplier names
  → Extract just the business name, not the address
- If only store ID visible and no business name found:
  → supplier.value = null
  → supplier.confidence = 0
  → Add warning: "Only store ID visible, no supplier name found"

## RULE 3: CUSTOMER NAME VALIDATION (CARDHOLDER TRAP)
- Names appearing near VISA, MasterCard, AMEX, card numbers (****1234) are CARDHOLDERS
- Cardholders are NOT customers - they are payment method holders
- Receipts typically do NOT have customer names
- If name appears near payment info:
  → customer.value = null
  → customer.confidence = 0
  → Add warning: "Name appears to be cardholder, not customer"

## RULE 4: CONFIDENCE SCORING (STRICT)
- 0.95-1.0: Crystal clear, explicitly labeled field (e.g., "Total: £17.50")
- 0.85-0.94: Clear but minor issues (slight blur, partially visible label)
- 0.70-0.84: Readable but requires interpretation or is INFERRED
- 0.50-0.69: Some uncertainty, context-based extraction
- 0.30-0.49: Weak evidence, partial text, guessing
- 0.0: Not visible or not found

IMPORTANT: Inferred values (is_inferred=true) can NEVER exceed 0.85 confidence

## RULE 5: INFERENCE RULES
Mark is_inferred=true when:
- Value derived from calculation (tax_rate from subtotal/tax)
- Value found in footer/watermark instead of main content
- Value derived from currency symbol (£ → GBP)
- Value interpreted from context

Always provide inference_reason explaining HOW you inferred it.

## RULE 6: AMOUNT VALIDATION
- Check if subtotal + tax_amount = total (within 0.02 tolerance)
- Set validation.amounts_reconcile = true/false
- Set validation.reconciliation_check = "X + Y = Z vs total W"
- If amounts don't reconcile, add warning

## RULE 7: CURRENCY DETECTION
- Look for explicit currency codes (GBP, USD, EUR, INR)
- If not found, infer from symbols: £→GBP, $→USD, €→EUR, ₹→INR
- If inferred: is_inferred=true, confidence max 0.85

## RULE 8: DATE FORMATTING
- Convert all dates to ISO format: YYYY-MM-DD
- Keep original format in raw_value
- Future dates are likely OCR errors → add warning

═══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT - Return valid JSON only
═══════════════════════════════════════════════════════════════════════════════

{{
  "supplier": {{
    "value": "Business Name or null",
    "confidence": 0.0-1.0,
    "is_inferred": false,
    "source_text": "exact text from document",
    "inference_reason": null
  }},
  "customer": {{
    "value": null,
    "confidence": 0,
    "is_inferred": false,
    "source_text": null,
    "inference_reason": null
  }},
  "invoice_number": {{
    "value": "string or null",
    "confidence": 0.0-1.0,
    "source_text": "exact text"
  }},
  "date": {{
    "value": "YYYY-MM-DD or null",
    "raw_value": "original format",
    "confidence": 0.0-1.0,
    "source_text": "exact text"
  }},
  "due_date": {{
    "value": "YYYY-MM-DD or null",
    "confidence": 0.0-1.0,
    "source_text": "exact text"
  }},
  "subtotal": {{
    "value": 0.00,
    "confidence": 0.0-1.0,
    "source_text": "exact text"
  }},
  "tax_amount": {{
    "value": 0.00,
    "confidence": 0.0-1.0,
    "source_text": "exact text"
  }},
  "tax_rate": {{
    "value": "20% or null",
    "confidence": 0.0-1.0,
    "is_inferred": false,
    "inference_reason": null
  }},
  "total": {{
    "value": 0.00,
    "confidence": 0.0-1.0,
    "source_text": "exact text"
  }},
  "currency": {{
    "value": "GBP/USD/EUR/INR or null",
    "confidence": 0.0-1.0,
    "is_inferred": true,
    "source_text": "£ symbol",
    "inference_reason": "Derived from £ symbol in amounts"
  }},
  "payment_method": {{
    "value": "Card/Cash/etc or null",
    "confidence": 0.0-1.0,
    "source_text": "exact text"
  }},
  "line_items": [
    {{
      "description": "Item name",
      "quantity": 1,
      "unit_price": 0.00,
      "total": 0.00,
      "confidence": 0.0-1.0
    }}
  ],
  "validation": {{
    "amounts_reconcile": true,
    "reconciliation_check": "subtotal + tax = total calculation",
    "issues_found": 0,
    "corrections_made": []
  }},
  "warnings": [
    "List any issues, validation failures, or concerns"
  ],
  "_extraction_complete": true,
  "_guardrails_applied": true
}}

═══════════════════════════════════════════════════════════════════════════════
IMPORTANT REMINDERS
═══════════════════════════════════════════════════════════════════════════════
- Return ONLY valid JSON, no markdown code blocks
- Include source_text for EVERY extracted value
- Add warnings for any issues found
- Be strict with confidence scores
- NULL is better than hallucinating"""


# =============================================================================
# TOOL DEFINITION
# =============================================================================

class ExtractorInput(BaseModel):
    """Input for field extractor."""
    image_base64: str = Field(description="Base64 encoded image")
    document_type: str = Field(default="unknown", description="Classified document type")


class FocusedExtractorTool(BaseTool):
    """Focused field extractor with built-in guardrails validation."""

    name: str = "focused_extractor"
    description: str = "Extracts and validates document fields in a single pass"
    args_schema: Type[BaseModel] = ExtractorInput

    def _run(self, image_base64: str, document_type: str = "unknown") -> str:
        """Extract and validate document fields."""
        from loguru import logger

        try:
            llm = ChatOpenAI(
                model=settings.EXTRACTION_MODEL,
                temperature=0.0,
                max_tokens=4000,
                openai_api_key=settings.OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1"
            )

            if image_base64.startswith('data:'):
                image_url = image_base64
            else:
                image_url = f"data:image/png;base64,{image_base64}"

            # Format prompt with document type
            prompt = EXTRACTION_PROMPT.format(document_type=document_type)

            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            )

            response = llm.invoke([message])
            response_text = response.content.strip()

            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)
            result['_document_type'] = document_type
            result['_extraction_complete'] = True
            result['_guardrails_applied'] = True

            # Log summary
            warnings = result.get('warnings', [])
            validation = result.get('validation', {})
            issues = validation.get('issues_found', 0)

            logger.info(f"Extraction complete for {document_type}: "
                       f"{issues} issues, {len(warnings)} warnings")

            return json.dumps(result)

        except Exception as e:
            logger.error(f"Extraction error: {e}")
            return json.dumps({
                "error": str(e),
                "_document_type": document_type,
                "_extraction_complete": False,
                "_guardrails_applied": False,
                "warnings": [f"Extraction failed: {str(e)}"]
            })


focused_extractor = FocusedExtractorTool()
