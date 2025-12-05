"""
Document Type Specific Extractors
Each extractor has custom rules optimized for that document type.
"""

from .thermal_extractor import extract_thermal
from .invoice_extractor import extract_invoice
from .handprint_extractor import extract_handprint
from .pos_extractor import extract_pos

__all__ = [
    "extract_thermal",
    "extract_invoice",
    "extract_handprint",
    "extract_pos"
]
