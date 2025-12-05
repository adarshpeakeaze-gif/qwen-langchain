"""
Document Classifier Tool
Classifies document type from image using vision LLM.
"""

import json
import base64
from typing import Type
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..config import settings

# =============================================================================
# CLASSIFICATION PROMPT
# =============================================================================

CLASSIFICATION_PROMPT = """Analyze this document image and classify it.

TASK: Determine the document type and characteristics.

DOCUMENT TYPES:
1. **thermal_receipt** - Small thermal paper receipt (typically from POS systems, gas stations, restaurants)
   - Characteristics: Narrow width, faded/gray text, simple formatting, often has dotted lines

2. **a4_invoice** - Standard A4/Letter size printed invoice or bill
   - Characteristics: Full page, professional layout, company letterhead, structured tables

3. **handwritten** - Handwritten invoice, receipt, or bill
   - Characteristics: Pen/pencil writing, possibly on ruled paper, informal layout

4. **small_receipt** - Small printed receipt (not thermal, like carbon copy or small printed)
   - Characteristics: Small size, basic printing, may have perforated edges

5. **utility_bill** - Utility bill (electricity, water, gas, phone)
   - Characteristics: Account number, meter readings, usage graphs, payment due date

6. **bank_statement** - Bank or financial statement
   - Characteristics: Account details, transaction list, opening/closing balance

7. **other** - Other document type

RESPOND IN THIS EXACT JSON FORMAT:
{
    "document_type": "thermal_receipt|a4_invoice|handwritten|small_receipt|utility_bill|bank_statement|other",
    "confidence": 0.0-1.0,
    "characteristics": {
        "paper_size": "thermal|a4|letter|small|unknown",
        "print_quality": "thermal|printed|handwritten|mixed",
        "has_letterhead": true/false,
        "has_table": true/false,
        "is_faded": true/false,
        "language_detected": "english|hindi|mixed|other"
    },
    "reasoning": "Brief explanation of classification"
}

Be precise. Only output valid JSON."""


# =============================================================================
# TOOL DEFINITION
# =============================================================================

class ClassifierInput(BaseModel):
    """Input for document classifier."""
    image_base64: str = Field(description="Base64 encoded image")


class DocumentClassifierTool(BaseTool):
    """Tool for classifying document type from image."""

    name: str = "document_classifier"
    description: str = "Classifies document type (thermal receipt, A4 invoice, handwritten, etc.) from image"
    args_schema: Type[BaseModel] = ClassifierInput

    def _run(self, image_base64: str) -> str:
        """
        Classify document from image.

        Args:
            image_base64: Base64 encoded image

        Returns:
            JSON string with classification result
        """
        from loguru import logger

        try:
            # Initialize vision LLM
            llm = ChatOpenAI(
                model=settings.VISION_MODEL,
                temperature=0.0,
                max_tokens=1000,
                openai_api_key=settings.OPENROUTER_API_KEY,
                openai_api_base="https://openrouter.ai/api/v1"
            )

            # Prepare image for vision model
            # Handle if base64 already has data URL prefix
            if image_base64.startswith('data:'):
                image_url = image_base64
            else:
                image_url = f"data:image/png;base64,{image_base64}"

            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": CLASSIFICATION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": image_url}
                    }
                ]
            )

            # Call LLM
            logger.info("Classifying document type...")
            response = llm.invoke([message])

            # Parse response
            response_text = response.content.strip()

            # Extract JSON from response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()

            # Validate JSON
            result = json.loads(response_text)

            # Ensure required fields
            if "document_type" not in result:
                result["document_type"] = "other"
            if "confidence" not in result:
                result["confidence"] = 0.5

            logger.info(f"Classification: {result['document_type']} (confidence: {result['confidence']})")

            return json.dumps(result)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse classification response: {e}")
            return json.dumps({
                "document_type": "other",
                "confidence": 0.0,
                "characteristics": {},
                "reasoning": f"Classification failed: {str(e)}",
                "raw_response": response_text if 'response_text' in locals() else None
            })
        except Exception as e:
            logger.error(f"Classification error: {e}")
            return json.dumps({
                "document_type": "other",
                "confidence": 0.0,
                "characteristics": {},
                "reasoning": f"Error: {str(e)}"
            })


# Create singleton instance
classifier_tool = DocumentClassifierTool()
