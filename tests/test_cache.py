"""Unit tests for HashedCache (UI date-blinding)."""

from __future__ import annotations

from pathlib import Path

import pytest

from radar_warning_game.data.cache import HashedCache


@pytest.fixture
def tmp_cache(tmp_path: Path):
    return HashedCache(tmp_path, suffix=".ar2v")


def test_path_uses_sha1_filename(tmp_cache):
    p = tmp_cache.path("2013/05/20/KTLX/KTLX20130520_204812_V06")
    # Filename should be sha1-hex + suffix; no date substring
    assert p.suffix == ".ar2v"
    assert len(p.stem) == 40   # sha1 hex
    assert "2013" not in p.name
    assert "20130520" not in p.name


def test_exists_before_and_after_write(tmp_cache):
    key = "test/key/foo"
    assert tmp_cache.exists(key) is False
    tmp_cache.path(key).write_bytes(b"hello")
    assert tmp_cache.exists(key) is True


def test_exists_empty_file_is_false(tmp_cache):
    """Zero-byte files (e.g. botched downloads) should not count as cached."""
    key = "test/empty"
    tmp_cache.path(key).write_bytes(b"")
    assert tmp_cache.exists(key) is False


def test_temp_path_distinct_from_final(tmp_cache):
    key = "test/key"
    assert tmp_cache.temp_path(key) != tmp_cache.path(key)
    assert tmp_cache.temp_path(key).suffix == ".part"


def test_finalize_renames_temp_to_final(tmp_cache):
    key = "test/key"
    tmp_cache.temp_path(key).write_bytes(b"payload")
    final = tmp_cache.finalize(key)
    assert final.exists()
    assert final.read_bytes() == b"payload"


def test_lookup_key_reverse_map(tmp_cache):
    key = "2013/05/20/KTLX/KTLX20130520_204812_V06"
    p = tmp_cache.path(key)
    hashed = p.stem
    assert tmp_cache.lookup_key(hashed) == key


def test_clear_key_map_drops_lookup(tmp_cache):
    key = "test/key"
    tmp_cache.path(key)
    hashed = tmp_cache.path(key).stem
    tmp_cache.clear_key_map()
    assert tmp_cache.lookup_key(hashed) is None
