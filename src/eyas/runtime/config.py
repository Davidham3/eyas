"""Shared config assembly utilities (used by both the CLI and the engine runtime): dotted-path overrides, plugin imports."""

from __future__ import annotations

import importlib
import sys
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ParsedValue:
    """One validated ``--set`` item: the raw key, its dotted parts, and the YAML-typed value."""

    key: str
    parts: tuple[str, ...]
    value: Any


def _validate_override(key: str, raw_value: str) -> ParsedValue:
    """Boundary check for one ``key=value`` item: an empty key raises, and the value is
    YAML-parsed to get its types — malformed YAML raises a :class:`ValueError` naming the
    offending item (an actionable message, not the bare scanner error that points at a
    buffer offset nobody can see)."""
    if not key:
        raise ValueError(f"--set must be in key=value form: {key!r}={raw_value!r}")
    try:
        value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"--set value is not valid YAML: {key}={raw_value!r} ({exc})") from exc
    return ParsedValue(key=key, parts=tuple(key.split(".")), value=value)


def _descend(node: dict, part: str, key: str) -> tuple[dict, bool]:
    """One path step: return the mapping child ``part`` and whether this step filled an
    open slot. A null-valued optional key (``judge: null`` — the slot exists but is
    disabled) is materialised into a mapping on descent; a ``params`` section is
    additionally the open kwargs namespace (assembled as ``cls(**params)``), so absent as
    well as null it is materialised — a typo there dies at startup with the constructor's
    TypeError, still fail fast, and pointing a demo config at another backend (fresh
    params keys the template never listed) is a first-class journey. Missing keys
    elsewhere stay an error (typo protection)."""
    filled = False
    if part in node and node[part] is None:
        node[part] = {}  # null-valued optional section: fill the slot
        filled = True
    elif part == "params" and part not in node:
        node[part] = {}  # open kwargs namespace: absent == empty
        filled = True
    if not isinstance(node.get(part), dict):
        raise ValueError(f"--set path does not exist: {key!r} ('{part}' is not a mapping)")
    return node[part], filled


def _apply_override(config: dict, parsed: ParsedValue) -> None:
    """Pure application of one validated override: descend the dotted path, then set the
    leaf. The leaf may be created inside a params map (open namespace) or a map this path
    just filled from a null/absent slot; anywhere else it must pre-exist."""
    node = config
    parts = parsed.parts
    materialised = False  # True once this path descended into a filled-in slot
    for part in parts[:-1]:
        node, filled = _descend(node, part, parsed.key)
        materialised = materialised or filled
    open_leaf = materialised or (len(parts) >= 2 and parts[-2] == "params")
    if parts[-1] not in node and not open_leaf:
        raise ValueError(f"--set path does not exist: {parsed.key!r}")
    node[parts[-1]] = parsed.value


def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """``a.b.c=value`` dotted-path overrides (values are YAML-parsed to get their types);
    errors when the path does not exist. Atomic: every item is validated first —
    at the boundary (:func:`_validate_override`) and by a dry-run application on a
    deep copy — so a malformed item or a bad path anywhere in the list never leaves
    the config half-edited (no earlier override applied, no null slot materialised)."""
    parsed: list[ParsedValue] = []
    for item in overrides:
        key, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"--set must be in key=value form: {item!r}")
        parsed.append(_validate_override(key, raw_value))
    probe = deepcopy(config)
    for p in parsed:
        _apply_override(probe, p)  # dry run: path errors surface before any real mutation
    for p in parsed:
        _apply_override(config, p)
    return config


def load_plugins(config: dict) -> None:
    """Import the module paths in the config ``plugins`` array (the registration entry
    point for user plugins).

    Modules resolve relative to the CWD (the convention is to launch from the repo
    root); fail fast on import failure or a bad key shape.
    """
    plugins = config.get("plugins")
    if plugins is None:
        return
    if not isinstance(plugins, list) or not all(isinstance(m, str) and m for m in plugins):
        raise ValueError(f"plugins must be a list of non-empty strings, got {plugins!r}")
    cwd = str(Path.cwd())
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    for module_name in plugins:
        importlib.import_module(module_name)
