"""
LangChain tools for document type classification.
All tools return the same standardized extraction schema.
"""

from langchain_core.tools import tool


# Standardized fields for all document types
COMMON_EXTRACTION_SCHEMA = {
    "fields_to_extract": [
        {"field": "document_type", "type": "string", "required": True},
        {"field": "document_issue_date", "type": "date", "required": True},
        {"field": "document_due_date", "type": "date", "required": False},
        {"field": "document_reference_number", "type": "string", "required": True},
        {"field": "is_document_paid", "type": "boolean", "required": False},
        {"field": "issuer_name", "type": "string", "required": True},
        {"field": "customer_name", "type": "string", "required": False},
        {"field": "issuer_vat_number", "type": "string", "required": False},
        {"field": "customer_vat_number", "type": "string", "required": False},
        {"field": "net_amount", "type": "number", "required": False},
        {"field": "vat_amount", "type": "number", "required": False},
        {"field": "gross_amount", "type": "number", "required": True},
        {"field": "currency", "type": "string", "required": True},
        {"field": "tax_lines", "type": "array", "required": False},
        {"field": "is_marketplace_document", "type": "boolean", "required": False},
        {"field": "is_cis_applicable", "type": "boolean", "required": False},
        {"field": "payment_method", "type": "string", "required": False},
        {"field": "payment_card_last_4_digits", "type": "string", "required": False},
        {"field": "document_description", "type": "string", "required": True},
        {"field": "line_items", "type": "array", "required": False},
    ],
    "output_format": {
        "document_type": "string",
        "document_issue_date": "YYYY-MM-DD or null",
        "document_due_date": "YYYY-MM-DD or null",
        "document_reference_number": "string or null",
        "is_document_paid": "boolean or null",
        "issuer_name": "string or null",
        "customer_name": "string or null",
        "issuer_vat_number": "string or null",
        "customer_vat_number": "string or null",
        "net_amount": "number or null",
        "vat_amount": "number or null",
        "gross_amount": "number or null",
        "currency": "string or null",
        "tax_lines": [{"tax_rate": "number", "net_amount": "number", "tax_amount": "number"}],
        "is_marketplace_document": "boolean or null",
        "is_cis_applicable": "boolean or null",
        "payment_method": "string or null",
        "payment_card_last_4_digits": "string or null",
        "document_description": "string",
        "line_items": [{"description": "string", "quantity": "number", "unit_price": "number", "net_amount": "number", "vat_rate": "number", "vat_amount": "number", "gross_amount": "number"}]
    }
}


def create_tool_response(doc_type: str) -> dict:
    """Create standardized response for a document type."""
    return {
        "document_type": doc_type,
        **COMMON_EXTRACTION_SCHEMA
    }


@tool
def extract_invoice_info(description: str) -> dict:
    """INVOICE - payment request for goods/services."""
    return create_tool_response("invoice")


@tool
def extract_credit_note_info(description: str) -> dict:
    """CREDIT_NOTE - refund or credit memo."""
    return create_tool_response("credit_note")


@tool
def extract_receipt_info(description: str) -> dict:
    """RECEIPT - proof of purchase/payment."""
    return create_tool_response("receipt")


@tool
def extract_cardholder_copy_info(description: str) -> dict:
    """CARDHOLDER_COPY - card payment slip."""
    return create_tool_response("cardholder_copy")


@tool
def extract_delivery_note_info(description: str) -> dict:
    """DELIVERY_NOTE - shipping/dispatch proof."""
    return create_tool_response("delivery_note")


@tool
def extract_bill_info(description: str) -> dict:
    """BILL - utility/phone/medical recurring charge."""
    return create_tool_response("bill")


@tool
def extract_bank_statement_info(description: str) -> dict:
    """BANK_STATEMENT - account transaction history."""
    return create_tool_response("bank_statement")


@tool
def extract_purchase_order_info(description: str) -> dict:
    """PURCHASE_ORDER - request to buy goods/services."""
    return create_tool_response("purchase_order")


@tool
def extract_expense_receipt_info(description: str) -> dict:
    """EXPENSE_RECEIPT - small purchase for expense reporting."""
    return create_tool_response("expense_receipt")


@tool
def extract_letter_info(description: str) -> dict:
    """LETTER - formal correspondence."""
    return create_tool_response("letter")


@tool
def extract_form_info(description: str) -> dict:
    """FORM - application/registration/tax form."""
    return create_tool_response("form")


@tool
def extract_id_document_info(description: str) -> dict:
    """ID_DOCUMENT - license/passport/ID card."""
    return create_tool_response("id_document")


@tool
def extract_contract_info(description: str) -> dict:
    """CONTRACT - legal agreement/lease."""
    return create_tool_response("contract")


@tool
def extract_generic_document_info(description: str) -> dict:
    """GENERIC - unknown/other document type."""
    return create_tool_response("generic")


# List of all extraction tools for the classifier agent
EXTRACTION_TOOLS = [
    extract_invoice_info,
    extract_credit_note_info,
    extract_receipt_info,
    extract_cardholder_copy_info,
    extract_delivery_note_info,
    extract_bill_info,
    extract_bank_statement_info,
    extract_purchase_order_info,
    extract_expense_receipt_info,
    extract_letter_info,
    extract_form_info,
    extract_id_document_info,
    extract_contract_info,
    extract_generic_document_info,
]
