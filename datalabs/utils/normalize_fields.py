"""
Field Normalization Utility
Ensures all extraction fields have consistent structure:
- value (actual value or null)
- confidence (0.0-1.0)
- isInferred (true/false)
"""

from typing import Dict, Any, Optional
from loguru import logger


# =============================================================================
# FIELD SCHEMA DEFINITION
# =============================================================================

# Define expected fields and their default structure
FIELD_SCHEMA = {
    "supplier": {
        "name": None,
        "address": None,
        "phone": None,
        "email": None,
        "vat_number": None
    },
    "customer": {
        "name": None,
        "address": None,
        "phone": None,
        "email": None
    },
    "document": {
        "type": None,
        "number": None,
        "date": None,
        "time": None,
        "order_number": None,
        "reference": None,
        "receipt_number": None
    },
    "amounts": {
        "subtotal": None,
        "goods_total": None,
        "tax": None,
        "tax_amount": None,
        "vat_amount": None,
        "vat_rate": None,
        "total": None,
        "discount": None
    },
    "payment": {
        "method": None,
        "card_type": None,
        "status": None,
        "currency": None,
        "amount_paid": None,
        "change": None
    },
    "items": []
}


def create_normalized_field(value: Any, confidence: float = 0.0, is_inferred: bool = False) -> Dict[str, Any]:
    """
    Create a normalized field structure.

    Args:
        value: The field value (or None)
        confidence: Confidence score (0.0-1.0)
        is_inferred: Whether the value was inferred

    Returns:
        Normalized field dict
    """
    return {
        "value": value,
        "confidence": confidence,
        "isInferred": is_inferred
    }


def normalize_simple_value(value: Any) -> Dict[str, Any]:
    """
    Convert a simple value to normalized structure.
    If value is already a dict with expected keys, preserve it.

    Args:
        value: Simple value or existing normalized dict

    Returns:
        Normalized field dict
    """
    if value is None:
        return create_normalized_field(None, 0.0, False)

    # Already normalized (has value, confidence, isInferred)
    if isinstance(value, dict):
        if "value" in value and "confidence" in value:
            # Ensure isInferred key exists (handle is_inferred vs isInferred)
            is_inferred = value.get("isInferred", value.get("is_inferred", False))
            return {
                "value": value.get("value"),
                "confidence": float(value.get("confidence", 0.0)),
                "isInferred": bool(is_inferred)
            }
        else:
            # It's a dict but not normalized - treat the whole dict as value
            return create_normalized_field(value, 0.8, False)

    # Simple value - assume high confidence direct extraction
    return create_normalized_field(value, 0.9, False)


def normalize_nested_object(data: Optional[Dict], schema: Dict) -> Dict[str, Any]:
    """
    Normalize a nested object (like supplier, customer, amounts).

    Args:
        data: The input data dict
        schema: Expected fields for this object

    Returns:
        Normalized object with all fields present
    """
    if data is None:
        data = {}

    result = {}

    for field_name, default_value in schema.items():
        if field_name in data:
            result[field_name] = normalize_simple_value(data[field_name])
        else:
            result[field_name] = create_normalized_field(None, 0.0, False)

    return result


def normalize_line_items(items: Optional[list]) -> Dict[str, Any]:
    """
    Normalize line items array.

    Args:
        items: List of line items

    Returns:
        Normalized items structure
    """
    if not items:
        return {
            "value": [],
            "confidence": 0.0,
            "isInferred": False
        }

    normalized_items = []
    total_confidence = 0.0

    for item in items:
        if isinstance(item, dict):
            normalized_item = {
                "description": normalize_simple_value(item.get("description")),
                "quantity": normalize_simple_value(item.get("quantity", item.get("qty"))),
                "unit_price": normalize_simple_value(item.get("unit_price", item.get("price"))),
                "amount": normalize_simple_value(item.get("amount", item.get("total"))),
                "vat_rate": normalize_simple_value(item.get("vat_rate"))
            }
            normalized_items.append(normalized_item)
            # Average confidence from item fields
            item_conf = sum([
                normalized_item["description"]["confidence"],
                normalized_item["quantity"]["confidence"],
                normalized_item["unit_price"]["confidence"],
                normalized_item["amount"]["confidence"]
            ]) / 4
            total_confidence += item_conf

    avg_confidence = total_confidence / len(items) if items else 0.0

    return {
        "value": normalized_items,
        "confidence": round(avg_confidence, 2),
        "isInferred": False
    }


def normalize_extraction_result(extracted_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize extraction result to ensure consistent field structure.

    All fields will have:
    - value: The actual value or null
    - confidence: 0.0-1.0 score
    - isInferred: boolean flag

    Args:
        extracted_data: Raw extraction result

    Returns:
        Normalized extraction result
    """
    if not extracted_data:
        extracted_data = {}

    logger.debug(f"Normalizing extraction result with {len(extracted_data)} top-level keys")

    result = {
        "supplier": normalize_nested_object(
            extracted_data.get("supplier"),
            FIELD_SCHEMA["supplier"]
        ),
        "customer": normalize_nested_object(
            extracted_data.get("customer"),
            FIELD_SCHEMA["customer"]
        ),
        "document": normalize_nested_object(
            extracted_data.get("document"),
            FIELD_SCHEMA["document"]
        ),
        "amounts": normalize_nested_object(
            extracted_data.get("amounts"),
            FIELD_SCHEMA["amounts"]
        ),
        "payment": normalize_nested_object(
            extracted_data.get("payment"),
            FIELD_SCHEMA["payment"]
        ),
        "items": normalize_line_items(
            extracted_data.get("items", [])
        )
    }

    # Preserve any additional top-level fields that aren't in schema
    for key, value in extracted_data.items():
        if key not in result:
            if isinstance(value, dict):
                # Normalize each field in the dict
                result[key] = {
                    k: normalize_simple_value(v) for k, v in value.items()
                }
            elif isinstance(value, list):
                result[key] = {
                    "value": value,
                    "confidence": 0.8,
                    "isInferred": False
                }
            else:
                result[key] = normalize_simple_value(value)

    logger.debug(f"Normalized result has {len(result)} sections")

    return result


# =============================================================================
# EXTRACTED_FIELDS SCHEMA (for app.py extract_document_entities format)
# =============================================================================

EXTRACTED_FIELDS_SCHEMA = {
    "supplier": None,
    "customer": None,
    "supplier_address": None,
    "supplier_vat_id": None,
    "customer_address": None,
    "invoice_number": None,
    "date": None,
    "due_date": None,
    "currency": None,
    "subtotal": None,
    "tax_amount": None,
    "tax_rate": None,
    "total": None,
    "payment_terms": None,
    "payment_method": None
}


def normalize_extracted_fields(data: Optional[Dict]) -> Dict[str, Any]:
    """
    Normalize extracted_fields structure from extract_document_entities.

    Ensures all fields have consistent structure:
    - value
    - confidence (0.0-1.0)
    - isInferred (boolean)

    Args:
        data: Raw extracted_fields dict

    Returns:
        Normalized extracted_fields with all expected keys
    """
    if data is None:
        data = {}

    result = {}

    for field_name in EXTRACTED_FIELDS_SCHEMA.keys():
        if field_name in data:
            field_data = data[field_name]
            if isinstance(field_data, dict) and "value" in field_data:
                # Already has structure, normalize it
                result[field_name] = {
                    "value": field_data.get("value"),
                    "confidence": float(field_data.get("confidence", 0.0)),
                    "isInferred": bool(field_data.get("is_inferred", field_data.get("isInferred", False))),
                    "source_text": field_data.get("source_text"),
                    "inference_reason": field_data.get("inference_reason"),
                    "raw_value": field_data.get("raw_value")
                }
            else:
                # Simple value - wrap it
                result[field_name] = create_normalized_field(field_data, 0.8, False)
        else:
            # Missing field - create null entry
            result[field_name] = create_normalized_field(None, 0.0, False)

    # Preserve any additional fields not in schema
    for key, value in data.items():
        if key not in result:
            if isinstance(value, dict) and "value" in value:
                result[key] = {
                    "value": value.get("value"),
                    "confidence": float(value.get("confidence", 0.0)),
                    "isInferred": bool(value.get("is_inferred", value.get("isInferred", False))),
                    "source_text": value.get("source_text"),
                    "inference_reason": value.get("inference_reason")
                }
            else:
                result[key] = normalize_simple_value(value)

    return result


def normalize_entity_extraction_result(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize the full extraction result from extract_document_entities.

    Expected input structure:
    {
        "document_analysis": {...},
        "extracted_fields": {...},
        "line_items": {...},
        "validation": {...}
    }

    Args:
        data: Full extraction result

    Returns:
        Normalized extraction result with consistent field structure
    """
    if not data:
        return {
            "document_analysis": {
                "type": {"value": None, "confidence": 0.0, "isInferred": False},
                "type_confidence": {"value": 0.0, "confidence": 0.0, "isInferred": False},
                "overall_quality": {"value": None, "confidence": 0.0, "isInferred": False}
            },
            "extracted_fields": normalize_extracted_fields({}),
            "line_items": normalize_line_items([]),
            "validation": {
                "amounts_reconcile": False,
                "reconciliation_detail": None,
                "warnings": []
            }
        }

    result = {}

    # Normalize document_analysis
    doc_analysis = data.get("document_analysis", {})
    result["document_analysis"] = {
        "type": normalize_simple_value(doc_analysis.get("type")),
        "type_confidence": normalize_simple_value(doc_analysis.get("type_confidence")),
        "overall_quality": normalize_simple_value(doc_analysis.get("overall_quality"))
    }

    # Normalize extracted_fields
    result["extracted_fields"] = normalize_extracted_fields(data.get("extracted_fields", {}))

    # Normalize line_items
    line_items_data = data.get("line_items", {})
    if isinstance(line_items_data, dict):
        items = line_items_data.get("items", [])
        result["line_items"] = {
            "items": normalize_line_items(items),
            "count": normalize_simple_value(line_items_data.get("count", len(items))),
            "confidence": normalize_simple_value(line_items_data.get("confidence", 0.8))
        }
    else:
        result["line_items"] = {
            "items": normalize_line_items([]),
            "count": {"value": 0, "confidence": 0.0, "isInferred": False},
            "confidence": {"value": 0.0, "confidence": 0.0, "isInferred": False}
        }

    # Keep validation as-is (it's metadata, not extracted data)
    result["validation"] = data.get("validation", {
        "amounts_reconcile": False,
        "reconciliation_detail": None,
        "warnings": []
    })

    return result


def get_field_summary(normalized_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get a summary of extracted fields.

    Args:
        normalized_data: Normalized extraction result

    Returns:
        Summary with field counts and confidence averages
    """
    total_fields = 0
    fields_with_values = 0
    total_confidence = 0.0
    inferred_count = 0

    def count_fields(obj, prefix=""):
        nonlocal total_fields, fields_with_values, total_confidence, inferred_count

        if isinstance(obj, dict):
            if "value" in obj and "confidence" in obj:
                # This is a normalized field
                total_fields += 1
                if obj["value"] is not None:
                    fields_with_values += 1
                    total_confidence += obj["confidence"]
                if obj.get("isInferred", False):
                    inferred_count += 1
            else:
                # Recurse into nested dict
                for key, value in obj.items():
                    count_fields(value, f"{prefix}.{key}")

    count_fields(normalized_data)

    return {
        "total_fields": total_fields,
        "fields_with_values": fields_with_values,
        "fields_null": total_fields - fields_with_values,
        "inferred_count": inferred_count,
        "average_confidence": round(total_confidence / fields_with_values, 2) if fields_with_values > 0 else 0.0
    }
