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
import math
from typing import Any

import numpy as np


def json_safe(value: Any, nonfinite: set[str] | None = None, _key: str | None = None) -> Any:
    """
    Convert `value` to something `json.dumps(..., allow_nan=False)` accepts.

    Non-finite floats become `None`; when `nonfinite` is given, the name of the dictionary key they were found under
    is added to it. Tuples become lists, dictionary keys become strings, NumPy scalars and arrays become Python
    values, and anything else unknown becomes its `str()`.
    """
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isfinite(value):
            return value
        if nonfinite is not None and _key is not None:
            nonfinite.add(_key)
        return None
    if isinstance(value, dict):
        return {str(k): json_safe(v, nonfinite, str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v, nonfinite, _key) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist(), nonfinite, _key)
    return str(value)


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def _slope(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope of `(step, value)` points, `None` when there are fewer than two distinct steps."""
    if len(points) < 2:
        return None
    xs = np.array([p[0] for p in points], dtype=np.float64)
    ys = np.array([p[1] for p in points], dtype=np.float64)
    xs = xs - xs.mean()
    denom = float((xs * xs).sum())
    if denom == 0.0:
        return None
    slope = float((xs * (ys - ys.mean())).sum() / denom)
    return slope if math.isfinite(slope) else None


def _size(digest: dict) -> int:
    return len(json.dumps(digest, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def build_digest(
    history: list[dict],
    *,
    step: int,
    max_steps: int,
    objective: dict[str, float],
    scales: dict[str, float],
    sentinels: dict[str, bool],
    control: list[dict] | None = None,
    last_n: int = 8,
    budget_bytes: int = 8192,
) -> dict:
    """
    Summarise the evaluation history of a run as a small, strictly JSON-safe dictionary.

    Args:
        history:
            One dictionary per evaluation, in order, each with a `"step"` and numeric metrics.
        step (`int`):
            Current training step.
        max_steps (`int`):
            Planned number of training steps.
        objective (`dict[str, float]`):
            Weight of each metric in the run's objective. The best step is the evaluation with the highest weighted
            sum over the objective metrics (evaluations where one is missing or non-finite are skipped).
        scales (`dict[str, float]`):
            Head scales to report, such as the learned temperatures.
        sentinels (`dict[str, bool]`):
            Named boolean flags to report.
        control (`list[dict]`, *optional*):
            History of a matched control arm, reported as its latest evaluation and best step.
        last_n (`int`, *optional*, defaults to `8`):
            Number of latest evaluations the per-metric slopes are computed over.
        budget_bytes (`int`, *optional*, defaults to `8192`):
            Upper bound on the size of the compact JSON encoding. Oldest history entries are dropped first, then the
            control arm and the slopes, so the bound always holds.

    Returns:
        `dict`: keys `step`, `max_steps`, `current`, `best_step`, `best_score`, `slopes`, `confidence_gap`,
        `objective`, `scales`, `sentinels`, `control`, `nonfinite`, `history`, `history_dropped`. The confidence gap is
        the current `confidence_gap` metric when there is one, otherwise `mean_confidence - accuracy`.
    """
    nonfinite: set[str] = set()
    clean = [json_safe(dict(h), nonfinite) for h in history]
    objective = {str(k): float(v) for k, v in objective.items()}

    def score(entry: dict) -> float | None:
        total = 0.0
        for name, weight in objective.items():
            value = _finite(entry.get(name))
            if value is None:
                return None
            total += weight * value
        return total if objective else None

    best_step, best_score = None, None
    for entry in clean:
        s = score(entry)
        if s is not None and (best_score is None or s > best_score):
            best_step, best_score = entry.get("step"), s

    current = {k: v for k, v in (clean[-1] if clean else {}).items() if k != "step"}
    window = clean[-max(int(last_n), 1) :]
    slopes: dict[str, float | None] = {}
    for name in sorted({k for e in window for k in e if k != "step"}):
        points = [(_finite(e.get("step")), _finite(e.get(name))) for e in window]
        slopes[name] = _slope([(x, y) for x, y in points if x is not None and y is not None])

    gap = _finite(current.get("confidence_gap"))
    if gap is None and _finite(current.get("mean_confidence")) is not None and _finite(current.get("accuracy")) is not None:
        gap = current["mean_confidence"] - current["accuracy"]

    control_summary = None
    if control:
        control_clean = [json_safe(dict(h), nonfinite) for h in control]
        scored = [(score(e), e.get("step")) for e in control_clean]
        scored = [(s, st) for s, st in scored if s is not None]
        control_summary = {
            "current": {k: v for k, v in control_clean[-1].items() if k != "step"},
            "step": control_clean[-1].get("step"),
            "best_step": max(scored, key=lambda p: p[0])[1] if scored else None,
        }

    digest: dict[str, Any] = {
        "step": int(step),
        "max_steps": int(max_steps),
        "current": current,
        "best_step": best_step,
        "best_score": best_score,
        "slopes": slopes,
        "confidence_gap": gap,
        "objective": objective,
        "scales": json_safe(scales, nonfinite),
        "sentinels": json_safe(sentinels),
        "control": control_summary,
        "nonfinite": [],
        "history": clean,
        "history_dropped": 0,
    }

    def finish() -> dict:
        digest["nonfinite"] = sorted(nonfinite)
        digest["history_dropped"] = len(clean) - len(digest["history"])
        return digest

    # Drop the oldest history first, in halving chunks so a very long history stays cheap.
    finish()
    while _size(digest) > budget_bytes and digest["history"]:
        excess = _size(digest) - budget_bytes
        per_entry = max(_size({"h": digest["history"]}) // len(digest["history"]), 1)
        drop = max(min(excess // per_entry, len(digest["history"])), 1)
        digest["history"] = digest["history"][drop:]
        finish()
    # Then the optional and derived parts, largest first, so the bound holds whatever the caller passed.
    for key in ("control", "slopes", "current", "scales", "sentinels", "objective"):
        if _size(digest) <= budget_bytes:
            break
        digest[key] = None if key == "control" else {}
    if _size(digest) > budget_bytes:
        digest = {
            "step": digest["step"],
            "max_steps": digest["max_steps"],
            "best_step": best_step,
            "nonfinite": [],
            "history": [],
            "history_dropped": len(clean),
            "truncated": True,
        }
        digest["nonfinite"] = sorted(nonfinite)
        while _size(digest) > budget_bytes and digest["nonfinite"]:
            digest["nonfinite"].pop()
        for fallback in ({"step": digest["step"], "truncated": True}, {"truncated": True}, {}):
            if _size(digest) <= budget_bytes:
                break
            digest = fallback
    return digest
