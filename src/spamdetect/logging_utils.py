"""Logging helpers (library code logs; only CLIs configure handlers)."""

import logging


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging for command-line entry points."""
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
