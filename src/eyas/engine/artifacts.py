"""Artifact pipeline (§9): typed, hash-addressed cross-method artifact store.

Methods ``emit`` key intermediate artifacts into the ``ArtifactStore``
(addressed by ``producer.kind``, deduplicated by content hash) and downstream
methods ``get`` them by address. At assembly time ``validate_assembly`` checks
that the declared dependencies exist (consumes ⊆ available). Optional
persistence to disk (run_dir/artifacts/<producer>.<kind>.json) makes artifacts
part of the self-contained run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

from eyas.engine.domain import Artifact

__all__ = ["ArtifactStore"]


class ArtifactStore:
    """Read/write container of address → Artifact; when ``path`` is set, emit
    persists synchronously (idempotent, keyed by digest)."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._items: dict[str, Artifact] = {}
        self._dir = Path(path) if path is not None else None

    def emit(self, artifact: Artifact) -> Artifact:
        self._items[artifact.address()] = artifact
        if self._dir is not None:
            self._dir.mkdir(parents=True, exist_ok=True)
            target = self._dir / f"{artifact.address().replace('/', '_')}.json"
            target.write_text(
                json.dumps(artifact.to_json(), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )
        return artifact

    def get(self, address: str) -> Artifact | None:
        return self._items.get(address)

    def load(self) -> None:
        """Reload persisted artifacts from the store's directory (``--resume``).

        Reads every ``<address>.json`` the emit path wrote back into memory so
        ``get`` / ``available`` / ``manifest`` / ``in`` work after a restart.
        A pathless store or a missing directory is a no-op; a corrupt file
        raises (a broken durable is loud). Payloads are JSON-native by
        convention — emit's ``default=str`` fallback means exotic payload
        members may not round-trip exactly (see ``Artifact.from_json``)."""
        if self._dir is None or not self._dir.is_dir():
            return
        for path in sorted(self._dir.glob("*.json")):
            artifact = Artifact.from_json(json.loads(path.read_text(encoding="utf-8")))
            self._items[artifact.address()] = artifact

    def available(self) -> set[str]:
        return set(self._items)

    def manifest(self) -> dict[str, str]:
        """address → digest (goes into result.json / report; auditable and deduplicable)."""
        return {addr: art.digest for addr, art in self._items.items()}

    def __contains__(self, address: object) -> bool:
        return address in self._items
