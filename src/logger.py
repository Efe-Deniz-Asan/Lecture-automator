# ------------------------------------------------------------------------------
#  Copyright (c) 2025 Efe Deniz Asan <asan.efe.deniz@gmail.com>
#  All Rights Reserved.
#
#  NOTICE:  All information contained herein is, and remains the property of
#  Efe Deniz Asan. The intellectual and technical concepts contained herein
#  are proprietary to Efe Deniz Asan and are protected by trade secret or
#  copyright law. Dissemination of this information or reproduction of this
#  material is strictly forbidden unless prior written permission is obtained
#  from Efe Deniz Asan or via email at <asan.efe.deniz@gmail.com>.
# ------------------------------------------------------------------------------

"""
Centralized logging configuration for Lecture Automator.
Provides consistent logging across all modules.
"""
import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name (typically __name__ from calling module)
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.INFO)
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # File handler with rotation
    file_handler = RotatingFileHandler(
        log_dir / "lecture_automator.log",
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=3,
        encoding='utf-8'
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler with simpler format
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        '%(levelname)s: %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
