"""
Datalabs - Logging System
Comprehensive logging with file and console output.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from loguru import logger

from .config import settings


def setup_logger(request_id: Optional[str] = None) -> None:
    """
    Configure the logging system.

    Args:
        request_id: Optional request ID for log file naming
    """
    # Remove default handler
    logger.remove()

    # Console logging format
    console_format = (
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # File logging format (more detailed)
    file_format = (
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    )

    # Add console handler
    if settings.LOG_TO_CONSOLE:
        logger.add(
            sys.stderr,
            format=console_format,
            level=settings.LOG_LEVEL,
            colorize=True
        )

    # Add file handler
    if settings.LOG_TO_FILE:
        log_dir = settings.log_dir_path

        # General application log
        logger.add(
            log_dir / "app.log",
            format=file_format,
            level=settings.LOG_LEVEL,
            rotation="10 MB",
            retention="7 days",
            compression="zip"
        )

        # Error-only log
        logger.add(
            log_dir / "error.log",
            format=file_format,
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )


class ProcessingLogger:
    """
    Logger for tracking individual document processing.
    Creates a detailed log file for each processing request.
    """

    def __init__(self, request_id: str, filename: str):
        """
        Initialize processing logger.

        Args:
            request_id: Unique request identifier
            filename: Original filename being processed
        """
        self.request_id = request_id
        self.filename = filename
        self.start_time = datetime.now()
        self.entries: list = []

        # Create request-specific log file
        self.log_file = settings.log_dir_path / f"{request_id}_{self._safe_filename()}.log"

    def _safe_filename(self) -> str:
        """Create safe filename from original."""
        safe = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.filename)
        return safe[:50]  # Limit length

    def log(self, stage: str, message: str, data: Optional[dict] = None, level: str = "INFO"):
        """
        Log a processing step.

        Args:
            stage: Processing stage name (e.g., "OCR", "EXTRACTION", "MAPPING")
            message: Log message
            data: Optional additional data
            level: Log level
        """
        timestamp = datetime.now()
        elapsed = (timestamp - self.start_time).total_seconds()

        entry = {
            "timestamp": timestamp.isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "stage": stage,
            "level": level,
            "message": message,
            "data": data
        }
        self.entries.append(entry)

        # Log to main logger
        log_msg = f"[{self.request_id}] [{stage}] {message}"
        if data:
            log_msg += f" | {data}"

        getattr(logger, level.lower(), logger.info)(log_msg)

        # Write to request-specific file
        self._write_to_file(entry)

    def _write_to_file(self, entry: dict):
        """Write entry to request log file."""
        with open(self.log_file, "a", encoding="utf-8") as f:
            line = (
                f"[{entry['timestamp']}] "
                f"[+{entry['elapsed_seconds']:.3f}s] "
                f"[{entry['stage']}] "
                f"[{entry['level']}] "
                f"{entry['message']}"
            )
            if entry.get("data"):
                line += f"\n    DATA: {entry['data']}"
            f.write(line + "\n")

    def log_reasoning(self, stage: str, reasoning: str):
        """
        Log LLM reasoning output.

        Args:
            stage: Processing stage
            reasoning: LLM reasoning text
        """
        self.log(stage, "REASONING", {"reasoning": reasoning})

        # Also write reasoning to dedicated section in log file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*60}\n")
            f.write(f"REASONING ({stage}):\n")
            f.write(f"{'='*60}\n")
            f.write(reasoning)
            f.write(f"\n{'='*60}\n\n")

    def complete(self, success: bool, result: Optional[dict] = None, error: Optional[str] = None):
        """
        Mark processing as complete.

        Args:
            success: Whether processing succeeded
            result: Final result if successful
            error: Error message if failed
        """
        elapsed = (datetime.now() - self.start_time).total_seconds()

        if success:
            self.log("COMPLETE", f"Processing completed in {elapsed:.2f}s", {"success": True})
        else:
            self.log("COMPLETE", f"Processing failed after {elapsed:.2f}s", {"success": False, "error": error}, "ERROR")

        # Write summary to file
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'#'*60}\n")
            f.write(f"# PROCESSING SUMMARY\n")
            f.write(f"{'#'*60}\n")
            f.write(f"Request ID: {self.request_id}\n")
            f.write(f"Filename: {self.filename}\n")
            f.write(f"Success: {success}\n")
            f.write(f"Total Time: {elapsed:.2f}s\n")
            if error:
                f.write(f"Error: {error}\n")
            f.write(f"{'#'*60}\n")


# Initialize logger on module import
setup_logger()
