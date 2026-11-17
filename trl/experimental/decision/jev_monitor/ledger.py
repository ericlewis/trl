# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import threading
import time
from pathlib import Path

from .digest import json_safe


class Ledger:
    """
    Append-only record of what the monitor saw and decided, one strict JSON object per line.

    Args:
        path (`str` or `os.PathLike`):
            File to append to. Parent directories are created on the first write.
    """

    def __init__(self, path: str | os.PathLike):
        self.path = Path(path)
        self._lock = threading.Lock()

    def append(self, kind: str, step: int, payload: dict) -> None:
        """Append one record `{"kind", "step", "time", "payload"}` and flush it to disk."""
        record = {"kind": str(kind), "step": int(step), "time": time.time(), "payload": json_safe(payload)}
        line = json.dumps(record, allow_nan=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())

    def read(self) -> list[dict]:
        """Return every record in order. Lines that are not valid JSON objects, such as a torn last write, are skipped."""
        with self._lock:
            return self._read()

    def _read(self) -> list[dict]:
        if not self.path.exists():
            return []
        records = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue
                if isinstance(record, dict):
                    records.append(record)
        return records

    def truncate_after(self, step: int) -> None:
        """Drop every record whose step is greater than `step`, as when training resumes from an earlier checkpoint."""
        with self._lock:
            if not self.path.exists():
                return
            kept = [r for r in self._read() if isinstance(r.get("step"), int) and r["step"] <= step]
            tmp = self.path.with_name(self.path.name + ".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                for record in kept:
                    f.write(json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
