# rapidindex/utils/logging.py
"""Logging configuration."""

import sys
from loguru import logger

# Remove default handler
logger.remove()

# Add custom handler
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)

# Add file handler
logger.add(
    "rapidindex.log",
    rotation="10 MB",
    retention="7 days",
    level="DEBUG"
)


__all__ = ['logger']