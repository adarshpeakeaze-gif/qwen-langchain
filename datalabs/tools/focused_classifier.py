"""
Focused Document Classifier
Simple, small prompt for document type classification.
"""

import json
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..config import settings

# =============================================================================
# FOCUSED CLASSIFICATION PROMPT (Small, focused)
# =============================================================================

CLASSIFIER_PROMPT = """Classify this document image into ONE type.

DOCUMENT TYPES:
- invoice: Supplier billing customer, has invoice number, line items, total
- receipt: Proof of payment, transaction number, often narrow/thermal format
- expense_receipt: Small retail purchase receipt
- credit_note: Refund/credit from supplier, negative amounts
- bank_statement: Account transactions, opening/closing balance
- purchase_order: Request to buy goods/services
- utility_bill: Electricity, water, gas, phone bill
- unknown: Cannot determine

LOOK FOR:
- Keywords: "INVOICE", "RECEIPT", "CREDIT NOTE", "STATEMENT", "ORDER"
- Layout: Narrow thermal vs full page, tables, letterhead
- Key fields: Invoice number, transaction ID, account number

OUTPUT JSON only:
{
  "document_type": "invoice|receipt|expense_receipt|credit_note|bank_statement|purchase_order|utility_bill|unknown",
  "confidence": 0.0-1.0,
  "reason": "brief explanation",
  "detected_keywords": ["list", "of", "keywords"],
  "layout": "thermal|a4|letter|small|unknown",
  "has_letterhead": true|false,
  "has_table": true|false,
  "is_handwritten": true|false
}"""


# =============================================================================
# TOOL DEFINITION
# =============================================================================

class ClassifierInput(BaseModel):
    """Input for document classifier."""
    image_base64: str = Field(description="Base64 encoded image")


class FocusedClassifierTool(BaseTool):
    """Focused document classifier with small prompt."""

    name: str = "focused_classifier"
    description: str = "Classifies document type with focused, small prompt"
    args_schema: Type[BaseModel] = ClassifierInput

    def _run(self, image_base64: str) -> str:
        """Classify document type."""
        from loguru import logger

        try:
            llm = ChatOpenAI(
                model=settings.VISION_MODEL,
                temperature=0.0,
                max_tokens=500,
                openai_api_key=settings.OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1"
            )

            if image_base64.startswith('data:'):
                image_url = image_base64
            else:
                image_url = f"data:image/png;base64,{image_base64}"

            message = HumanMessage(
                content=[
                    {"type": "text", "text": CLASSIFIER_PROMPT},
                    {"type": "image_url", "image_url": {"url": image_url}}
                ]
            )

            response = llm.invoke([message])
            response_text = response.content.strip()

            # Extract JSON
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            result = json.loads(response_text)
            logger.info(f"Classified as: {result.get('document_type')} ({result.get('confidence')})")
            return json.dumps(result)

        except Exception as e:
            logger.error(f"Classification error: {e}")
            return json.dumps({
                "document_type": "unknown",
                "confidence": 0.0,
                "reason": f"Error: {str(e)}",
                "error": str(e)
            })


focused_classifier = FocusedClassifierTool()
