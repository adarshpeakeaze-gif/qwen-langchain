"""
Datalabs - Schema Mapping Tool
LangChain tool for mapping raw extraction to standardized schema.
Handles the new evidence-driven extraction format with confidence scores.
"""

import json
from pathlib import Path
from typing import Type, Any, Dict, Optional, List

from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from loguru import logger

from ..config import settings
from ..schemas import (
    ExtractionOutput,
    RawExtractionOutput,
    PartyInfo,
    AddressInfo,
    TaxInfo,
    MonetaryAmount,
    ExtractedField,
    FieldStatus,
    TaxBreakdown,
    LineItem,
    LineItemsContainer,
    BankDetails,
    ValidationResult,
    AtAGlance
)


# =============================================================================
# TOOL INPUT SCHEMA
# =============================================================================

class MappingToolInput(BaseModel):
    """Input schema for the mapping tool."""
    raw_extraction: str = Field(
        description="JSON string with raw extraction from extraction tool"
    )
    debug_mode: bool = Field(
        default=True,
        description="Enable detailed debug logging"
    )


# =============================================================================
# MAPPING TOOL IMPLEMENTATION
# =============================================================================

class SchemaMappingTool(BaseTool):
    """
    LangChain tool for mapping raw extraction to standardized schema.

    Takes the output from EntityExtractionTool (evidence-driven format) and maps it to
    Pydantic-validated schema with proper typing.

    This tool does NOT use an LLM - it's a deterministic mapping operation.

    Features:
    - Direct field mapping without LLM
    - Preserves confidence scores and inference flags
    - Validates against Pydantic models
    - Debug logging for troubleshooting
    """

    name: str = "schema_mapping"
    description: str = """
    Map raw extraction data to standardized Pydantic schema.

    Use this tool when you need to:
    - Convert raw extraction to typed schema
    - Validate data structure
    - Prepare data for downstream systems

    Input: JSON string from extraction tool
    Output: Pydantic-validated schema as JSON

    Note: This is a deterministic mapping - no LLM involved.
    """
    args_schema: Type[BaseModel] = MappingToolInput

    def _build_extracted_field(self, data: Any) -> ExtractedField:
        """Build ExtractedField from raw data."""
        if data is None:
            return ExtractedField(
                value=None,
                confidence=0.0,
                status=FieldStatus.MISSING,
                is_inferred=False,
                source_text=None,
                inference_reason=None
            )

        if isinstance(data, dict):
            # Map status string to enum
            status_str = data.get("status", "missing")
            try:
                status = FieldStatus(status_str)
            except ValueError:
                status = FieldStatus.MISSING

            return ExtractedField(
                value=data.get("value"),
                confidence=float(data.get("confidence", 0.0)),
                status=status,
                is_inferred=bool(data.get("is_inferred", False)),
                source_text=data.get("source_text"),
                inference_reason=data.get("inference_reason")
            )

        # If it's a simple value, wrap it
        return ExtractedField(
            value=data,
            confidence=0.5,
            status=FieldStatus.VISIBLE,
            is_inferred=False,
            source_text=str(data),
            inference_reason=None
        )

    def _build_address(self, data: Any) -> Optional[AddressInfo]:
        """Build AddressInfo from raw data."""
        if not data:
            return None

        if isinstance(data, dict):
            return AddressInfo(
                full_text=data.get("full_text"),
                street=data.get("street"),
                city=data.get("city"),
                postal_code=data.get("postal_code"),
                country=data.get("country")
            )

        return AddressInfo(full_text=str(data))

    def _build_party_info(self, name_field: Any, address_data: Any = None) -> PartyInfo:
        """Build PartyInfo from name field and optional address."""
        return PartyInfo(
            name=self._build_extracted_field(name_field),
            address=self._build_address(address_data),
            vat_number=ExtractedField(),  # VAT usually not in basic extraction
            email=None,
            phone=None
        )

    def _build_monetary_amount(self, amount_field: Any, currency_field: Any) -> MonetaryAmount:
        """Build MonetaryAmount from amount and currency fields."""
        return MonetaryAmount(
            amount=self._build_extracted_field(amount_field),
            currency=self._build_extracted_field(currency_field)
        )

    def _build_tax_info(self, tax_amount: Any, tax_rate: Any) -> TaxInfo:
        """Build TaxInfo from tax fields."""
        return TaxInfo(
            total_tax=self._build_extracted_field(tax_amount),
            tax_rate=self._build_extracted_field(tax_rate),
            tax_breakdown=[],
            is_reverse_charge=False,
            tax_scheme=None
        )

    def _build_line_items(self, data: Any) -> LineItemsContainer:
        """Build LineItemsContainer from raw data."""
        if not data:
            return LineItemsContainer()

        if isinstance(data, dict):
            items = data.get("value", [])
            status_str = data.get("status", "missing")
            try:
                status = FieldStatus(status_str)
            except ValueError:
                status = FieldStatus.MISSING

            line_items = []
            for item in items:
                if isinstance(item, dict):
                    line_items.append(LineItem(
                        description=item.get("description", ""),
                        quantity=float(item.get("quantity", 1.0)),
                        unit_price=float(item.get("unit_price", 0.0)),
                        amount=float(item.get("amount", 0.0)),
                        vat_rate=item.get("vat_rate"),
                        is_inferred=bool(item.get("is_inferred", False))
                    ))

            return LineItemsContainer(
                value=line_items,
                confidence=float(data.get("confidence", 0.0)),
                status=status
            )

        return LineItemsContainer()

    def _build_bank_details(self, data: Any) -> BankDetails:
        """Build BankDetails from raw data."""
        if not data or not isinstance(data, dict):
            return BankDetails()

        return BankDetails(
            account_number=self._build_extracted_field(data.get("account_number")),
            sort_code=self._build_extracted_field(data.get("sort_code")),
            iban=self._build_extracted_field(data.get("iban")),
            account_name=self._build_extracted_field(data.get("account_name"))
        )

    def _build_validation(self, data: Any) -> ValidationResult:
        """Build ValidationResult from raw data."""
        if not data or not isinstance(data, dict):
            return ValidationResult()

        return ValidationResult(
            amounts_reconcile=bool(data.get("amounts_reconcile", False)),
            reconciliation_check=data.get("reconciliation_check"),
            date_valid=bool(data.get("date_valid", True)),
            supplier_has_evidence=bool(data.get("supplier_has_evidence", False)),
            customer_has_evidence=bool(data.get("customer_has_evidence", False))
        )

    def _build_at_a_glance(self, data: Any) -> AtAGlance:
        """Build AtAGlance from raw data."""
        if not data or not isinstance(data, dict):
            return AtAGlance()

        return AtAGlance(
            document_type=data.get("document_type", "unknown"),
            document_type_confidence=float(data.get("document_type_confidence", 0.0)),
            document_type_reason=data.get("document_type_reason"),
            key_visible_fields=data.get("key_visible_fields", []),
            key_inferred_fields=data.get("key_inferred_fields", []),
            missing_critical_fields=data.get("missing_critical_fields", []),
            quick_summary=data.get("quick_summary")
        )

    def _build_extraction_output(self, raw_data: Dict) -> ExtractionOutput:
        """Build ExtractionOutput from raw extraction data."""
        extracted = raw_data.get("extracted", {})
        at_a_glance = raw_data.get("at_a_glance", {})

        # Build document type field from at_a_glance
        doc_type_field = ExtractedField(
            value=at_a_glance.get("document_type", "unknown"),
            confidence=float(at_a_glance.get("document_type_confidence", 0.0)),
            status=FieldStatus.INFERRED if at_a_glance.get("document_type_confidence", 0) < 0.9 else FieldStatus.VISIBLE,
            is_inferred=True,
            source_text=None,
            inference_reason=at_a_glance.get("document_type_reason")
        )

        # Get currency for monetary amounts
        currency_field = extracted.get("currency", {})

        return ExtractionOutput(
            at_a_glance=self._build_at_a_glance(at_a_glance),
            document_type=doc_type_field,
            supplier=self._build_party_info(extracted.get("supplier")),
            customer=self._build_party_info(extracted.get("customer")),
            subtotal=self._build_monetary_amount(
                extracted.get("subtotal"),
                currency_field
            ),
            gross_amount=self._build_monetary_amount(
                extracted.get("total"),
                currency_field
            ),
            tax_info=self._build_tax_info(
                extracted.get("tax_amount"),
                extracted.get("tax_rate")
            ),
            document_date=self._build_extracted_field(
                extracted.get("invoice_date_iso") or extracted.get("invoice_date")
            ),
            due_date=self._build_extracted_field(extracted.get("due_date")),
            document_number=self._build_extracted_field(extracted.get("invoice_number")),
            line_items=self._build_line_items(raw_data.get("line_items")),
            bank_details=self._build_bank_details(raw_data.get("bank_details")),
            payment_method=self._build_extracted_field(extracted.get("payment_method")),
            payment_reference=self._build_extracted_field(extracted.get("payment_reference")),
            validation=self._build_validation(raw_data.get("validation")),
            warnings=raw_data.get("warnings", []),
            raw_ocr_text=raw_data.get("raw_ocr_text")
        )

    def _log_mapping_debug(self, raw_data: Dict, output: ExtractionOutput, request_id: str = ""):
        """Log detailed mapping debug information."""
        prefix = f"[{request_id}] " if request_id else ""

        logger.info(f"{prefix}[Mapping Debug] Document type: {output.document_type.value}")
        logger.info(f"{prefix}[Mapping Debug] Supplier: {output.supplier.name.value} "
                   f"(conf={output.supplier.name.confidence})")
        logger.info(f"{prefix}[Mapping Debug] Customer: {output.customer.name.value} "
                   f"(conf={output.customer.name.confidence})")
        logger.info(f"{prefix}[Mapping Debug] Total: {output.gross_amount.amount.value} "
                   f"{output.gross_amount.currency.value}")

        # Log field counts
        at_a_glance = output.at_a_glance
        logger.debug(f"{prefix}[Mapping Debug] Visible fields: {len(at_a_glance.key_visible_fields)}")
        logger.debug(f"{prefix}[Mapping Debug] Inferred fields: {len(at_a_glance.key_inferred_fields)}")
        logger.debug(f"{prefix}[Mapping Debug] Missing fields: {len(at_a_glance.missing_critical_fields)}")

        # Log validation
        validation = output.validation
        logger.debug(f"{prefix}[Mapping Debug] Amounts reconcile: {validation.amounts_reconcile}")
        logger.debug(f"{prefix}[Mapping Debug] Reconciliation: {validation.reconciliation_check}")

        # Log warnings count
        if output.warnings:
            logger.warning(f"{prefix}[Mapping Debug] {len(output.warnings)} warnings present")

    def _run(
        self,
        raw_extraction: str,
        debug_mode: bool = True,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """
        Map raw extraction to standardized schema.

        Args:
            raw_extraction: JSON string from extraction tool
            debug_mode: Enable detailed debug logging

        Returns:
            JSON string with mapped and validated data
        """
        logger.info("[Mapping Tool] Starting schema mapping")

        # Parse input
        try:
            raw_data = json.loads(raw_extraction)
        except json.JSONDecodeError as e:
            logger.error(f"[Mapping Tool] Invalid JSON input: {e}")
            return json.dumps({
                "success": False,
                "error": f"Invalid JSON input: {e}"
            }, indent=2)

        # Check for extraction errors
        if "error" in raw_data:
            logger.warning(f"[Mapping Tool] Extraction had error: {raw_data['error']}")
            # Still try to map what we can

        try:
            # Build the output model
            extraction_output = self._build_extraction_output(raw_data)

            # Debug logging
            if debug_mode:
                self._log_mapping_debug(raw_data, extraction_output)

            logger.info("[Mapping Tool] Mapping complete")

            # Return full result
            result = {
                "success": True,
                "extraction": extraction_output.model_dump()
            }

            return json.dumps(result, indent=2)

        except Exception as e:
            logger.error(f"[Mapping Tool] Error during mapping: {e}")
            import traceback
            logger.error(f"[Mapping Tool] Traceback: {traceback.format_exc()}")

            return json.dumps({
                "success": False,
                "error": str(e),
                "raw_data_keys": list(raw_data.keys()) if isinstance(raw_data, dict) else "not a dict"
            }, indent=2)

    async def _arun(
        self,
        raw_extraction: str,
        debug_mode: bool = True
    ) -> str:
        """Async execution of mapping tool."""
        import asyncio
        return await asyncio.to_thread(self._run, raw_extraction, debug_mode)


# =============================================================================
# TOOL INSTANCE
# =============================================================================

mapping_tool = SchemaMappingTool()
