"""Per-machine rolling history buffer for streaming feature computation.

Rolling means/stds/z-scores need recent history, not just the current
reading. The architecture diagram's "Feature Store" is the production
answer (Redis / a time-series store shared across API replicas); this is
an in-process equivalent sized for a single-instance deployment or local
development. Swapping in a real feature store later means replacing this
class's storage backend only — ``AnomalyPredictor`` depends on the
``append`` / ``get_history`` interface, not on dict-of-deques internals.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque

import pandas as pd

from src.data.schema import ID_COLUMNS, SENSOR_COLUMNS


class FeatureStateStore:
    def __init__(self, history_window: int = 96, sensor_columns: list[str] | None = None):
        self.history_window = history_window
        self.sensor_columns = sensor_columns or SENSOR_COLUMNS
        self._buffers: dict[str, deque] = defaultdict(lambda: deque(maxlen=history_window))
        self._lock = threading.Lock()

    def append(self, machine_id: str, reading: dict) -> None:
        with self._lock:
            self._buffers[machine_id].append(reading)

    def get_history(self, machine_id: str) -> pd.DataFrame:
        with self._lock:
            rows = list(self._buffers[machine_id])
        columns = ID_COLUMNS + self.sensor_columns
        if not rows:
            return pd.DataFrame(columns=columns)
        df = pd.DataFrame(rows)
        return df.sort_values("timestamp").reset_index(drop=True)

    def machine_count(self) -> int:
        with self._lock:
            return len(self._buffers)

    def reading_count(self, machine_id: str) -> int:
        with self._lock:
            return len(self._buffers[machine_id])
