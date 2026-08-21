"""Custom exception hierarchy for the summarization service."""

from __future__ import annotations


class SummarizerError(Exception):
    """Base exception for all service-specific errors."""


class UnsupportedFileTypeError(SummarizerError):
    """Raised when a document type has no registered loader."""


class DocumentLoadError(SummarizerError):
    """Raised when a document cannot be read or parsed."""


class SummarizationError(SummarizerError):
    """Raised when the summarization pipeline fails."""


class ConfigurationError(SummarizerError):
    """Raised when configuration is invalid or missing."""