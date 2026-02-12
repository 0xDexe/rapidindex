# rapidindex/core/exceptions.py
"""
Custom exceptions for RapidIndex.

This module defines all custom exceptions used throughout the application,
organized by category for better error handling and debugging.
"""

from typing import Optional


class RapidIndexError(Exception):
    """Base exception for all RapidIndex errors."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self):
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


# ==================== Parsing Exceptions ====================

class ParseError(RapidIndexError):
    """Raised when document parsing fails."""
    pass


class UnsupportedFormatError(ParseError):
    """Raised when file format is not supported."""
    pass


class FileNotFoundError(ParseError):
    """Raised when file cannot be found."""
    pass


class FileSizeExceededError(ParseError):
    """Raised when file size exceeds limits."""
    pass


class CorruptedFileError(ParseError):
    """Raised when file is corrupted or unreadable."""
    pass


# ==================== Indexing Exceptions ====================

class IndexError(RapidIndexError):
    """Base exception for indexing errors."""
    pass


class IndexNotFoundError(IndexError):
    """Raised when index doesn't exist."""
    pass


class IndexingFailedError(IndexError):
    """Raised when indexing operation fails."""
    pass


class DuplicateDocumentError(IndexError):
    """Raised when trying to index duplicate document."""
    pass


# ==================== Search Exceptions ====================

class SearchError(RapidIndexError):
    """Base exception for search errors."""
    pass


class InvalidQueryError(SearchError):
    """Raised when search query is invalid."""
    pass


class SearchTimeoutError(SearchError):
    """Raised when search operation times out."""
    pass


class NoResultsError(SearchError):
    """Raised when search returns no results."""
    pass


# ==================== LLM Exceptions ====================

class LLMError(RapidIndexError):
    """Base exception for LLM-related errors."""
    pass


class LLMAPIError(LLMError):
    """Raised when LLM API call fails."""
    pass


class LLMTimeoutError(LLMError):
    """Raised when LLM API call times out."""
    pass


class LLMRateLimitError(LLMError):
    """Raised when LLM API rate limit is hit."""
    pass


class LLMTokenLimitError(LLMError):
    """Raised when token limit is exceeded."""
    pass


class InvalidPromptError(LLMError):
    """Raised when prompt is invalid."""
    pass


# ==================== Cache Exceptions ====================

class CacheError(RapidIndexError):
    """Base exception for cache errors."""
    pass


class CacheConnectionError(CacheError):
    """Raised when cache connection fails."""
    pass


class CacheKeyError(CacheError):
    """Raised when cache key is invalid."""
    pass


# ==================== Storage Exceptions ====================

class StorageError(RapidIndexError):
    """Base exception for storage errors."""
    pass


class DatabaseError(StorageError):
    """Raised when database operation fails."""
    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""
    pass


class MigrationError(DatabaseError):
    """Raised when database migration fails."""
    pass


# ==================== API Exceptions ====================

class APIError(RapidIndexError):
    """Base exception for API errors."""
    pass


class ValidationError(APIError):
    """Raised when request validation fails."""
    pass


class AuthenticationError(APIError):
    """Raised when authentication fails."""
    pass


class AuthorizationError(APIError):
    """Raised when authorization fails."""
    pass


class RateLimitError(APIError):
    """Raised when rate limit is exceeded."""
    pass


# ==================== MCP Exceptions ====================

class MCPError(RapidIndexError):
    """Base exception for MCP-related errors."""
    pass


class MCPConnectionError(MCPError):
    """Raised when MCP connection fails."""
    pass


class MCPProtocolError(MCPError):
    """Raised when MCP protocol violation occurs."""
    pass


class MCPToolError(MCPError):
    """Raised when MCP tool execution fails."""
    pass


# ==================== Configuration Exceptions ====================

class ConfigError(RapidIndexError):
    """Base exception for configuration errors."""
    pass


class InvalidConfigError(ConfigError):
    """Raised when configuration is invalid."""
    pass


class MissingConfigError(ConfigError):
    """Raised when required configuration is missing."""
    pass


# Export all exceptions
__all__ = [
    # Base
    'RapidIndexError',
    
    # Parsing
    'ParseError',
    'UnsupportedFormatError',
    'FileNotFoundError',
    'FileSizeExceededError',
    'CorruptedFileError',
    
    # Indexing
    'IndexError',
    'IndexNotFoundError',
    'IndexingFailedError',
    'DuplicateDocumentError',
    
    # Search
    'SearchError',
    'InvalidQueryError',
    'SearchTimeoutError',
    'NoResultsError',
    
    # LLM
    'LLMError',
    'LLMAPIError',
    'LLMTimeoutError',
    'LLMRateLimitError',
    'LLMTokenLimitError',
    'InvalidPromptError',
    
    # Cache
    'CacheError',
    'CacheConnectionError',
    'CacheKeyError',
    
    # Storage
    'StorageError',
    'DatabaseError',
    'DatabaseConnectionError',
    'MigrationError',
    
    # API
    'APIError',
    'ValidationError',
    'AuthenticationError',
    'AuthorizationError',
    'RateLimitError',
    
    # MCP
    'MCPError',
    'MCPConnectionError',
    'MCPProtocolError',
    'MCPToolError',
    
    # Config
    'ConfigError',
    'InvalidConfigError',
    'MissingConfigError',
]