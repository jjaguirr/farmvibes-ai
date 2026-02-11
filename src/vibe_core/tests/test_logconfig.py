# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging

import pytest

from vibe_core.logconfig import (
    JSON_FORMAT,
    RequestContext,
    RequestContextFilter,
)


def _capture_record(message: str) -> logging.LogRecord:
    """Emit one log record through a Capture handler and return it."""
    records = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord):
            records.append(record)

    logger = logging.getLogger("test_logconfig_capture")
    logger.setLevel(logging.DEBUG)
    h = Capture()
    h.addFilter(RequestContextFilter())
    logger.addHandler(h)
    try:
        logger.info(message)
    finally:
        logger.removeHandler(h)
    return records[0]


def test_json_format_contains_request_fields():
    assert "request_id" in JSON_FORMAT
    assert "request_path" in JSON_FORMAT
    assert "duration_ms" in JSON_FORMAT


def test_request_context_filter_defaults_to_empty():
    RequestContext.clear()
    record = _capture_record("hello")
    assert record.request_id == ""
    assert record.request_path == ""
    assert record.duration_ms == ""


def test_request_context_filter_injects_values():
    RequestContext.set("req-123", "/v0/health", "42")
    try:
        record = _capture_record("hello")
        assert record.request_id == "req-123"
        assert record.request_path == "/v0/health"
        assert record.duration_ms == "42"
    finally:
        RequestContext.clear()


def test_request_context_clear_resets_to_empty():
    RequestContext.set("req-999", "/v0/liveness", "5")
    RequestContext.clear()
    record = _capture_record("hello")
    assert record.request_id == ""
    assert record.request_path == ""
    assert record.duration_ms == ""
