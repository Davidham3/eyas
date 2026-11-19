"""EYAS_LOG_LEVEL env switch: case-insensitive named levels, invalid values fail fast."""

from __future__ import annotations

import logging

import pytest

from eyas.logging import resolve_log_level


def test_unset_defaults_to_info() -> None:
    """Unset env takes the documented default (INFO)."""
    assert resolve_log_level({}) == logging.INFO


def test_none_reads_os_environ_unset_is_info(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("EYAS_LOG_LEVEL", raising=False)
    assert resolve_log_level() == logging.INFO


def test_explicit_env_mapping_is_case_insensitive() -> None:
    assert resolve_log_level({"EYAS_LOG_LEVEL": "debug"}) == logging.DEBUG
    assert resolve_log_level({"EYAS_LOG_LEVEL": "WARNING"}) == logging.WARNING


def test_os_environ_monkeypatch_level(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EYAS_LOG_LEVEL", "ERROR")
    assert resolve_log_level() == logging.ERROR


def test_invalid_value_fails_fast_naming_accepted_set() -> None:
    with pytest.raises(ValueError) as excinfo:
        resolve_log_level({"EYAS_LOG_LEVEL": "verbose"})
    message = str(excinfo.value)
    assert "EYAS_LOG_LEVEL" in message
    assert "DEBUG" in message
