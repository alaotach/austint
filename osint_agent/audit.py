"""Shared append-only JSONL audit logging utilities."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


def write_audit_event(
    event_type: str,
    details: dict[str, Any],
    audit_path: str | Path | None = None,
    callback: Callable[[str, dict[str, Any]], None] | None = None,
) -> None:
    """Emit an audit record to an optional callback and/or append-only JSONL file."""
    if callback is not None:
        try:
            callback(event_type, details)
        except Exception as err:
            logger.debug("Audit callback failed for %s: %s", event_type, err)

    path = Path(audit_path or "audit.jsonl")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event_type,
            "data": details,
            "details": details,
        }
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.debug("Could not write audit event to %s: %s", path, exc)


class AuditLogger:
    """Append-only audit logger bound to a specific JSONL file path."""

    def __init__(self, log_path: str | Path = "audit.jsonl") -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log_event(self, event_type: str, details: dict[str, Any]) -> None:
        """Append an event record as a JSON line."""
        write_audit_event(event_type, details, audit_path=self.log_path)
