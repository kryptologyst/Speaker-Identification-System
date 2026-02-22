"""Logging utilities."""

import logging
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Setup logging configuration.
    
    Args:
        level: Logging level.
        log_dir: Directory for log files.
        log_file: Log file name.
    """
    # Create log directory if specified
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file and log_dir:
        file_path = Path(log_dir) / log_file
        file_handler = logging.FileHandler(file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def get_logger(name: str) -> logging.Logger:
    """Get logger instance.
    
    Args:
        name: Logger name.
        
    Returns:
        Logger instance.
    """
    return logging.getLogger(name)


def anonymize_log_message(message: str) -> str:
    """Anonymize log message by removing potential PII.
    
    Args:
        message: Original log message.
        
    Returns:
        Anonymized log message.
    """
    # Simple anonymization - remove common PII patterns
    import re
    
    # Remove email addresses
    message = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', message)
    
    # Remove phone numbers
    message = re.sub(r'\b\d{3}-\d{3}-\d{4}\b', '[PHONE]', message)
    message = re.sub(r'\(\d{3}\)\s*\d{3}-\d{4}', '[PHONE]', message)
    
    # Remove SSN patterns
    message = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN]', message)
    
    return message
