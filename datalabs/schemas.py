"""
Datalabs - Pydantic Schemas
Data models for the document processing pipeline.
Updated to match the evidence-driven extraction prompt.
"""

from enum import Enum
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field


# =============================================================================
# ENUMS
# =============================================================================

class DocumentType(str, Enum):
    """Document type classification (accounting focus)."""
    INVOICE = "invoice"
    RECEIPT = "receipt"
    SALES_DOCUMENT = "sales_document"
    CREDIT_NOTE = "credit_note"
    BANK_STATEMENT = "bank_statement"
    EXPENSE_RECEIPT = "expense_receipt"
    PURCHASE_ORDER = "purchase_order"
    UNKNOWN = "unknown"


class ProcessingStatus(str, Enum):
    """Pipeline processing status."""
    PENDING = "pending"
    OCR_IN_PROGRESS = "ocr_in_progress"
    OCR_COMPLETE = "ocr_complete"
    EXTRACTION_IN_PROGRESS = "extraction_in_progress"
    EXTRACTION_COMPLETE = "extraction_complete"
    MAPPING_IN_PROGRESS = "mapping_in_progress"
    COMPLETE = "complete"
    FAILED = "failed"


class FieldStatus(str, Enum):
    """Status of an extracted field."""
    VISIBLE = "visible"      # Directly visible in document
    INFERRED = "inferred"    # Derived/calculated from other data
    MISSING = "missing"      # Not found in document


# =============================================================================
# EXTRACTED FIELD MODEL (Core building block)
# =============================================================================

class ExtractedField(BaseModel):
    """
    Base model for an extracted field with confidence and inference tracking.

    This is the core building block - every extracted value uses this structure
    to track confidence, source, and whether it was inferred.
    """
    value: Optional[Any] = Field(
        default=None,
        description="Extracted value (null if not found)"
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score 0.0-1.0"
    )
    status: FieldStatus = Field(
        default=FieldStatus.MISSING,
        description="visible | inferred | missing"
    )
    is_inferred: bool = Field(
        default=False,
        description="True if value was derived/calculated, not directly visible"
    )
    source_text: Optional[str] = Field(
        default=None,
        description="Exact OCR text this was extracted from"
    )
    inference_reason: Optional[str] = Field(
        default=None,
        description="Explanation of how value was inferred (required if is_inferred=true)"
    )

    class Config:
        use_enum_values = True


# =============================================================================
# ADDRESS AND PARTY MODELS
# =============================================================================

class AddressInfo(BaseModel):
    """Address information."""
    full_text: Optional[str] = Field(default=None, description="Full address as text block")
    street: Optional[str] = Field(default=None)
    city: Optional[str] = Field(default=None)
    postal_code: Optional[str] = Field(default=None)
    country: Optional[str] = Field(default=None)


class PartyInfo(BaseModel):
    """Information about a party (supplier/customer)."""
    name: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Party name"
    )
    address: Optional[AddressInfo] = Field(
        default=None,
        description="Address details"
    )
    vat_number: ExtractedField = Field(
        default_factory=ExtractedField,
        description="VAT/Tax ID"
    )
    email: Optional[str] = Field(default=None)
    phone: Optional[str] = Field(default=None)


# =============================================================================
# FINANCIAL MODELS
# =============================================================================

class TaxBreakdown(BaseModel):
    """Tax breakdown by rate."""
    rate: float = Field(description="Tax rate percentage")
    net_amount: float = Field(description="Net amount at this rate")
    tax_amount: float = Field(description="Tax amount at this rate")


class TaxInfo(BaseModel):
    """Comprehensive tax information."""
    total_tax: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Total tax/VAT amount"
    )
    tax_rate: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Tax rate (e.g., '20%')"
    )
    tax_breakdown: List[TaxBreakdown] = Field(
        default_factory=list,
        description="Breakdown by rate"
    )
    is_reverse_charge: bool = Field(
        default=False,
        description="Whether reverse charge applies"
    )
    tax_scheme: Optional[str] = Field(
        default=None,
        description="Tax scheme type"
    )


class MonetaryAmount(BaseModel):
    """Monetary amount with currency."""
    amount: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Amount value"
    )
    currency: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Currency code (GBP, USD, EUR, etc.)"
    )


class LineItem(BaseModel):
    """Individual line item from document."""
    description: str = Field(description="Item description")
    quantity: float = Field(default=1.0)
    unit_price: float = Field(default=0.0)
    amount: float = Field(default=0.0)
    vat_rate: Optional[str] = Field(default=None, description="VAT rate e.g. '20%'")
    is_inferred: bool = Field(default=False, description="Whether any values were inferred")


class LineItemsContainer(BaseModel):
    """Container for line items with metadata."""
    value: List[LineItem] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    status: FieldStatus = Field(default=FieldStatus.MISSING)

    class Config:
        use_enum_values = True


# =============================================================================
# BANK DETAILS
# =============================================================================

class BankDetails(BaseModel):
    """Bank account details for payment."""
    account_number: ExtractedField = Field(default_factory=ExtractedField)
    sort_code: ExtractedField = Field(default_factory=ExtractedField)
    iban: ExtractedField = Field(default_factory=ExtractedField)
    account_name: ExtractedField = Field(default_factory=ExtractedField)


# =============================================================================
# VALIDATION RESULTS
# =============================================================================

class ValidationResult(BaseModel):
    """Validation checks performed on extracted data."""
    amounts_reconcile: bool = Field(
        default=False,
        description="Whether subtotal + tax = total"
    )
    reconciliation_check: Optional[str] = Field(
        default=None,
        description="Details of reconciliation calculation"
    )
    date_valid: bool = Field(
        default=True,
        description="Whether dates are logically valid"
    )
    supplier_has_evidence: bool = Field(
        default=False,
        description="Whether supplier was found with evidence"
    )
    customer_has_evidence: bool = Field(
        default=False,
        description="Whether customer was found with evidence"
    )


# =============================================================================
# AT-A-GLANCE SUMMARY
# =============================================================================

class AtAGlance(BaseModel):
    """Quick summary of document classification and extraction status."""
    document_type: str = Field(
        default="unknown",
        description="Classified document type"
    )
    document_type_confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence in document type classification"
    )
    document_type_reason: Optional[str] = Field(
        default=None,
        description="Why this document type was chosen"
    )
    key_visible_fields: List[str] = Field(
        default_factory=list,
        description="Fields that were clearly visible"
    )
    key_inferred_fields: List[str] = Field(
        default_factory=list,
        description="Fields that were inferred"
    )
    missing_critical_fields: List[str] = Field(
        default_factory=list,
        description="Expected fields that were not found"
    )
    quick_summary: Optional[str] = Field(
        default=None,
        description="1-2 sentence description of the document"
    )


# =============================================================================
# RAW EXTRACTION OUTPUT (Direct from LLM)
# =============================================================================

class RawExtractionOutput(BaseModel):
    """
    Raw extraction output matching the LLM prompt structure.
    This is what comes directly from the extraction LLM.
    """
    at_a_glance: AtAGlance = Field(
        default_factory=AtAGlance,
        description="Quick summary and classification"
    )

    # Core extracted fields
    supplier: ExtractedField = Field(default_factory=ExtractedField)
    customer: ExtractedField = Field(default_factory=ExtractedField)
    invoice_number: ExtractedField = Field(default_factory=ExtractedField)
    invoice_date: ExtractedField = Field(default_factory=ExtractedField)
    invoice_date_iso: ExtractedField = Field(default_factory=ExtractedField)
    due_date: ExtractedField = Field(default_factory=ExtractedField)
    subtotal: ExtractedField = Field(default_factory=ExtractedField)
    tax_amount: ExtractedField = Field(default_factory=ExtractedField)
    tax_rate: ExtractedField = Field(default_factory=ExtractedField)
    total: ExtractedField = Field(default_factory=ExtractedField)
    currency: ExtractedField = Field(default_factory=ExtractedField)
    payment_method: ExtractedField = Field(default_factory=ExtractedField)
    payment_reference: ExtractedField = Field(default_factory=ExtractedField)

    # Complex structures
    line_items: LineItemsContainer = Field(default_factory=LineItemsContainer)
    bank_details: BankDetails = Field(default_factory=BankDetails)

    # Validation and warnings
    validation: ValidationResult = Field(default_factory=ValidationResult)
    warnings: List[str] = Field(default_factory=list)

    # Debug info
    raw_ocr_text: Optional[str] = Field(
        default=None,
        description="Full OCR text for debugging"
    )


# =============================================================================
# MAPPED EXTRACTION OUTPUT (After schema mapping)
# =============================================================================

class ExtractionOutput(BaseModel):
    """
    Complete extraction output after mapping to standardized schema.
    This combines raw extraction with additional processing.
    """
    # Summary
    at_a_glance: AtAGlance = Field(
        default_factory=AtAGlance,
        description="Quick summary and classification"
    )

    # Document Classification
    document_type: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Type of document"
    )

    # Parties
    supplier: PartyInfo = Field(
        default_factory=PartyInfo,
        description="Supplier/vendor information"
    )
    customer: PartyInfo = Field(
        default_factory=PartyInfo,
        description="Customer/buyer information"
    )

    # Amounts
    subtotal: MonetaryAmount = Field(
        default_factory=MonetaryAmount,
        description="Subtotal before tax"
    )
    gross_amount: MonetaryAmount = Field(
        default_factory=MonetaryAmount,
        description="Gross/total amount"
    )

    # Tax Information
    tax_info: TaxInfo = Field(
        default_factory=TaxInfo,
        description="Tax/VAT information"
    )

    # Dates
    document_date: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Document/invoice date (ISO format)"
    )
    due_date: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Payment due date"
    )

    # Reference Numbers
    document_number: ExtractedField = Field(
        default_factory=ExtractedField,
        description="Invoice/receipt number"
    )

    # Line Items
    line_items: LineItemsContainer = Field(
        default_factory=LineItemsContainer,
        description="Individual line items"
    )

    # Bank Details
    bank_details: BankDetails = Field(
        default_factory=BankDetails,
        description="Bank account details for payment"
    )

    # Payment Information
    payment_method: ExtractedField = Field(default_factory=ExtractedField)
    payment_reference: ExtractedField = Field(default_factory=ExtractedField)

    # Validation
    validation: ValidationResult = Field(
        default_factory=ValidationResult,
        description="Data validation results"
    )

    # Quality Tracking
    warnings: List[str] = Field(
        default_factory=list,
        description="Issues, low confidence fields, ambiguities"
    )

    # Debug
    raw_ocr_text: Optional[str] = Field(
        default=None,
        description="Original OCR text for debugging"
    )


# =============================================================================
# API REQUEST/RESPONSE MODELS
# =============================================================================

class ProcessingResponse(BaseModel):
    """API response for document processing."""
    success: bool = Field(description="Whether processing succeeded")
    request_id: str = Field(description="Request identifier")
    status: ProcessingStatus = Field(description="Processing status")

    # Results
    extraction: Optional[ExtractionOutput] = Field(
        default=None,
        description="Extracted data"
    )
    raw_extraction: Optional[RawExtractionOutput] = Field(
        default=None,
        description="Raw extraction before mapping (for debugging)"
    )
    html_content: Optional[str] = Field(
        default=None,
        description="HTML from OCR (if requested)"
    )

    # Timing
    ocr_time_ms: int = Field(default=0)
    extraction_time_ms: int = Field(default=0)
    mapping_time_ms: int = Field(default=0)
    total_time_ms: int = Field(default=0)

    # Errors
    error: Optional[str] = Field(default=None)
    error_stage: Optional[str] = Field(default=None)

    # Debug
    debug_info: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional debug information"
    )

    class Config:
        use_enum_values = True


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(default="healthy")
    version: str = Field(default="3.0.0")
    datalab_configured: bool = Field(default=False)
    llm_configured: bool = Field(default=False)
