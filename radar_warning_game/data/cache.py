"""Disk cache management with hashed filenames for UI date-blinding.

S3 object keys contain dates (``YYYY/MM/DD/...``). To prevent the date from leaking
into the cache directory (where a player might glimpse it), we store every cached
file under ``<sha1-of-key>.<suffix>`` on disk and keep an in-memory map from hash
back to the real key for the lifetime of the round. The map is cleared at round end.

This is honor-system blinding (per plan §4b) — a determined user can still find
the date by inspecting their own S3 traffic or the Level 2 binary headers.
"""

from __future__ import annotations

import hashlib
import logging
import os
import time
from datetime import timedelta
from pathlib import Path
from threading import RLock

log = logging.getLogger(__name__)

DEFAULT_CACHE_ROOT = Path(os.path.expanduser("~/.radaranalysiscorp/cache"))

# Periodic cleanup: at startup, purge files whose last-modification time is
# older than this. Keeps the cache from growing indefinitely across many rounds.
DEFAULT_CACHE_MAX_AGE = timedelta(days=30)


def purge_cache_older_than(root: Path = DEFAULT_CACHE_ROOT,
                            max_age: timedelta = DEFAULT_CACHE_MAX_AGE) -> int:
    """Delete cache files older than ``max_age``. Returns count of files removed.

    Best-effort: any per-file error is logged and the rest continues. Safe to
    call on startup; idempotent if nothing is old.
    """
    if not root.exists():
        return 0
    cutoff = time.time() - max_age.total_seconds()
    n_removed = 0
    n_failed = 0
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink()
                n_removed += 1
        except OSError as e:  # noqa: BLE001
            log.debug("Cache cleanup: skipped %s (%s)", path.name, e)
            n_failed += 1
    if n_removed or n_failed:
        log.info("Cache cleanup: removed %d files (%d skipped) under %s",
                 n_removed, n_failed, root)
    return n_removed


class HashedCache:
    """File-system cache keyed by an arbitrary string identifier, stored under sha1.

    Thread-safe for typical concurrent-download usage.
    """

    def __init__(self, root: Path | None = None, suffix: str = ".bin") -> None:
        self.root = (root or DEFAULT_CACHE_ROOT).resolve()
        self.suffix = suffix
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._key_by_hash: dict[str, str] = {}

    def _hash(self, key: str) -> str:
        return hashlib.sha1(key.encode("utf-8")).hexdigest()

    def path(self, key: str) -> Path:
        h = self._hash(key)
        with self._lock:
            self._key_by_hash[h] = key
        return self.root / f"{h}{self.suffix}"

    def exists(self, key: str) -> bool:
        p = self.path(key)
        return p.exists() and p.stat().st_size > 0

    def temp_path(self, key: str) -> Path:
        """Path for an in-progress download. Caller renames to final on completion."""
        return self.path(key).with_suffix(self.suffix + ".part")

    def finalize(self, key: str) -> Path:
        """Atomic rename from temp_path to path. Returns the final path."""
        tmp = self.temp_path(key)
        final = self.path(key)
        tmp.rename(final)
        return final

    def clear_key_map(self) -> None:
        """Drop the in-memory hash→key reverse map (call at round end)."""
        with self._lock:
            self._key_by_hash.clear()

    def lookup_key(self, hashed: str) -> str | None:
        """Reverse-lookup the original key for a hashed filename. Returns None if unknown."""
        with self._lock:
            return self._key_by_hash.get(hashed)
