"""Retry decorator for transient summarization failures."""

from __future__ import annotations

import time
from collections.abc import Callable
from functools import wraps
from typing import TypeVar

from doc_summarizer.config.logging_config import get_logger
from doc_summarizer.core.exceptions import SummarizationError

logger = get_logger(__name__)

T = TypeVar("T")


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Retry a function on SummarizationError with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: object, **kwargs: object) -> T:
            last_exc: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except SummarizationError as exc:
                    last_exc = exc
                    if attempt < max_attempts:
                        delay = base_delay * (2 ** (attempt - 1))
                        logger.warning(
                            "Attempt %d/%d failed: %s. Retrying in %.1fs.",
                            attempt, max_attempts, exc, delay,
                        )
                        time.sleep(delay)
            assert last_exc is not None
            raise last_exc

        return wrapper

    return decorator