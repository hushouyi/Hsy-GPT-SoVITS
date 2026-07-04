"""
Mapping file manager.

Reads/writes mapping.txt which tracks recording state for each sentence.
Uses in-memory cache with async flush to disk for performance.

Format:
    idx|text|wav_path|confirmed|duration_sec|recorded_at
"""

import os
import threading
import time
from typing import Dict, List, Optional


class MappingEntry:
    """Single mapping entry for one sentence."""
    def __init__(self, idx: int = 0, text: str = "", wav_path: str = "",
                 confirmed: bool = False, duration_sec: float = 0,
                 recorded_at: str = "0"):
        self.idx = idx
        self.text = text
        self.wav_path = wav_path
        self.confirmed = confirmed
        self.duration_sec = duration_sec
        self.recorded_at = recorded_at

    def to_line(self) -> str:
        confirmed_str = "yes" if self.confirmed else "no"
        return f"{self.idx}|{self.text}|{self.wav_path}|{confirmed_str}|{self.duration_sec}|{self.recorded_at}"

    @classmethod
    def from_line(cls, line: str) -> Optional['MappingEntry']:
        line = line.strip()
        if not line:
            return None
        parts = line.split('|', 5)
        if len(parts) < 4:
            return None
        try:
            idx = int(parts[0])
            text = parts[1]
            wav_path = parts[2]
            confirmed = parts[3].strip().lower() == 'yes'
            duration_sec = float(parts[4]) if len(parts) > 4 and parts[4] else 0.0
            recorded_at = parts[5] if len(parts) > 5 else "0"
            return cls(idx, text, wav_path, confirmed, duration_sec, recorded_at)
        except (ValueError, IndexError):
            return None


class MappingManager:
    """
    Manages mapping.txt with in-memory cache.
    Reads from disk on load(), writes back on flush().
    """

    def __init__(self, filepath: str = None):
        self.filepath = filepath
        self._cache = {}  # Dict[int, MappingEntry]
        self._dirty = False
        self._lock = threading.Lock()
        self._last_flush_time = 0

    def load(self, filepath: str = None) -> Dict[int, MappingEntry]:
        """Load mapping from disk into cache. Returns the cache dict."""
        path = filepath or self.filepath
        if not path:
            return self._cache

        self.filepath = path
        cache = {}

        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('idx|'):
                        continue
                    entry = MappingEntry.from_line(line)
                    if entry:
                        cache[entry.idx] = entry

        with self._lock:
            self._cache = cache
            self._dirty = False

        return cache

    def get(self, idx: int) -> Optional[MappingEntry]:
        """Get entry by sentence index."""
        with self._lock:
            return self._cache.get(idx)

    def get_all(self) -> Dict[int, MappingEntry]:
        """Get all entries."""
        with self._lock:
            return dict(self._cache)

    def update(self, idx: int, entry: MappingEntry) -> None:
        """Update or insert an entry."""
        with self._lock:
            self._cache[idx] = entry
            self._dirty = True

    def update_field(self, idx: int, **kwargs) -> None:
        """Update specific fields of an entry."""
        with self._lock:
            if idx not in self._cache:
                self._cache[idx] = MappingEntry(idx=idx)
            entry = self._cache[idx]
            for key, val in kwargs.items():
                setattr(entry, key, val)
            self._dirty = True

    def get_confirmed(self) -> List[MappingEntry]:
        """Get all entries with confirmed=True."""
        with self._lock:
            return [e for e in self._cache.values() if e.confirmed and e.wav_path]

    def get_category_stats(self, category_map: Dict[str, tuple]) -> Dict[str, dict]:
        """
        Get stats per category.
        category_map: {category_name: (start_idx, end_idx)}
        Returns: {category_name: {"total": N, "confirmed": N, "met": bool}}
        """
        with self._lock:
            stats = {}
            for cat_name, (start, end) in category_map.items():
                confirmed = 0
                total = end - start + 1
                for idx in range(start, end + 1):
                    entry = self._cache.get(idx)
                    if entry and entry.confirmed and entry.wav_path:
                        confirmed += 1
                stats[cat_name] = {
                    "total": total,
                    "confirmed": confirmed,
                    "met": confirmed >= 5
                }
            return stats

    def flush(self) -> bool:
        """Write cache to disk if dirty. Returns True if written."""
        if not self.filepath:
            return False

        with self._lock:
            if not self._dirty:
                return False
            # Sort by idx
            sorted_entries = sorted(self._cache.values(), key=lambda e: e.idx)
            try:
                os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
                with open(self.filepath, 'w', encoding='utf-8') as f:
                    f.write("idx|text|wav_path|confirmed|duration_sec|recorded_at\n")
                    for entry in sorted_entries:
                        f.write(entry.to_line() + '\n')
                self._dirty = False
                self._last_flush_time = time.time()
                return True
            except Exception as e:
                print(f"[WARN] Failed to write mapping: {e}")
                return False

    def auto_flush(self) -> None:
        """Call this periodically to flush pending writes."""
        if self._dirty and (time.time() - self._last_flush_time > 1.0):
            self.flush()

    @property
    def is_dirty(self) -> bool:
        return self._dirty
