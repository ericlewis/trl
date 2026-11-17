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


# Smallest `budget_bytes` honoured by `build_digest`: smaller budgets are clamped to it, so that even the most reduced
# digest (steps, best step, a few flagged names, drop count) always fits.
MIN_BUDGET_BYTES = 512
_MAX_INT = 2**53


# Containers nested deeper than this are dropped by `json_safe`, so that no encoder or decoder ever recurses too deep
MAX_DEPTH = 64
# Integers wider than this are dropped by `json_safe`: they fit no float, and very long ones no JSON encoder
_MAX_INT_BITS = 1024


def json_safe(value: Any, nonfinite: set[str] | None = None, _key: str | None = None, _depth: int = 0) -> Any:
    """
    Convert `value` to something `json.dumps(..., allow_nan=False)` accepts, without ever raising.

    Non-finite floats, integers wider than 1024 bits and containers nested more than `MAX_DEPTH` (64) levels deep
    become `None`; when `nonfinite` is given, the name of the dictionary key they were found under is added to it.
    Tuples and sets become lists, dictionary keys become strings, NumPy scalars and arrays become Python values, and
    anything else unknown becomes its `str()` (or `None` if that fails).
    """

    def dropped():
        if nonfinite is not None and _key is not None:
            nonfinite.add(_key)
        return None

    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, (int, np.integer)):
        value = int(value)
        return value if value.bit_length() <= _MAX_INT_BITS else dropped()
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else dropped()
    if isinstance(value, (dict, list, tuple, set, frozenset, np.ndarray)) and _depth >= MAX_DEPTH:
        return dropped()
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            try:
                name = str(k)
            except Exception:  # noqa: BLE001 - a key whose `__str__` raises
                name = f"<{type(k).__name__}>"
            out[name] = json_safe(v, nonfinite, name, _depth + 1)
        return out
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(v, nonfinite, _key, _depth + 1) for v in value]
    if isinstance(value, np.ndarray):
        try:
            return json_safe(value.tolist(), nonfinite, _key, _depth)
        except Exception:  # noqa: BLE001 - an object array whose items cannot be listed
            return dropped()
    try:
        return str(value)
    except Exception:  # noqa: BLE001 - a `__str__` that raises
        return None


def _finite(value: Any) -> float | None:
    """`value` as a finite float, or `None` when it is not a number, not finite, or does not fit a float."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        return None
    try:
        value = float(value)
    except (OverflowError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _step(value: Any, name: str, nonfinite: set[str]) -> int | None:
    """A step as a bounded int; garbage (NaN, inf, huge, not a number) becomes `None` and `name` is flagged."""
    number = _finite(value)
    if number is None or abs(number) >= _MAX_INT:
        nonfinite.add(name)
        return None
    return int(number)


def _slope(points: list[tuple[float, float]]) -> float | None:
    """Least-squares slope of `(step, value)` points, `None` when there are fewer than two distinct steps."""
    if len(points) < 2:
        return None
    with np.errstate(all="ignore"):
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


def _short_names(names: set[str] | list[str], limit: int = 16, width: int = 48) -> list[str]:
    return [n[:width] for n in sorted(names)[:limit]]


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
    Summarise the evaluation history of a run as a small, strictly JSON-safe dictionary. Numeric garbage (NaN, inf,
    overflow, integers wider than 1024 bits) and values nested more than 64 levels deep never raise: they become `None`
    and the name they came from is listed under `"nonfinite"`.

    Args:
        history:
            One dictionary per evaluation, in order, each with a `"step"` and numeric metrics.
        step (`int`):
            Current training step.
        max_steps (`int`):
            Planned number of training steps.
        objective (`dict[str, float]`):
            Weight of each metric in the run's objective. The best step is the evaluation with the highest weighted sum
            over the objective metrics (evaluations where one is missing or non-finite are skipped, as are terms with a
            non-finite weight).
        scales (`dict[str, float]`):
            Head scales to report, such as the learned temperatures.
        sentinels (`dict[str, bool]`):
            Named boolean flags to report.
        control (`list[dict]`, *optional*):
            History of a matched control arm, reported as its latest evaluation and best step.
        last_n (`int`, *optional*, defaults to `8`):
            Number of latest evaluations the per-metric slopes are computed over.
        budget_bytes (`int`, *optional*, defaults to `8192`):
            Upper bound on the size of the compact JSON encoding, clamped up to `MIN_BUDGET_BYTES` (512). Oldest
            history entries are dropped first, then the control arm, slopes, current values, scales, sentinels and
            objective, then the list of flagged names, and finally everything but a few fields, so the bound always
            holds.

    Returns:
        `dict`: keys `step`, `max_steps`, `current`, `best_step`, `best_score`, `slopes`, `confidence_gap`,
        `objective`, `scales`, `sentinels`, `control`, `nonfinite`, `history`, `history_dropped`. The confidence gap is
        the current `confidence_gap` metric when there is one, otherwise `mean_confidence - accuracy`.
    """
    budget_bytes = max(int(budget_bytes), MIN_BUDGET_BYTES)
    nonfinite: set[str] = set()
    clean = [json_safe(dict(h), nonfinite) for h in history]
    weights: dict[str, float | None] = {}
    for name, weight in objective.items():
        weights[str(name)] = _finite(weight)
        if weights[str(name)] is None:
            nonfinite.add(str(name))

    def score(entry: dict) -> float | None:
        if not weights:
            return None
        total = 0.0
        with np.errstate(all="ignore"):
            for name, weight in weights.items():
                value = _finite(entry.get(name))
                if weight is None:
                    continue
                if value is None:
                    return None
                total += weight * value
        if not math.isfinite(total):
            nonfinite.add("objective_score")
            return None
        return total

    best_step, best_score = None, None
    for entry in clean:
        s = score(entry)
        if s is not None and (best_score is None or s > best_score):
            best_step, best_score = entry.get("step"), s
    best_score = _finite(best_score)
    best_step = None if best_step is None else _step(best_step, "step", nonfinite)

    current = {k: v for k, v in (clean[-1] if clean else {}).items() if k != "step"}
    window = clean[-max(int(last_n) if _finite(last_n) is not None else 8, 1) :]
    slopes: dict[str, float | None] = {}
    for name in sorted({k for e in window for k in e if k != "step"}):
        points = [(_finite(e.get("step")), _finite(e.get(name))) for e in window]
        slopes[name] = _slope([(x, y) for x, y in points if x is not None and y is not None])

    gap = _finite(current.get("confidence_gap"))
    confidence, accuracy = _finite(current.get("mean_confidence")), _finite(current.get("accuracy"))
    if gap is None and confidence is not None and accuracy is not None:
        gap = _finite(confidence - accuracy)

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
        "step": _step(step, "step", nonfinite),
        "max_steps": _step(max_steps, "max_steps", nonfinite),
        "current": current,
        "best_step": best_step,
        "best_score": best_score,
        "slopes": slopes,
        "confidence_gap": gap,
        "objective": weights,
        "scales": json_safe(scales, nonfinite),
        "sentinels": json_safe(sentinels, nonfinite),
        "control": control_summary,
        "nonfinite": [],
        "history": clean,
        "history_dropped": 0,
    }

    def finish() -> dict:
        digest["nonfinite"] = sorted(nonfinite)
        digest["history_dropped"] = len(clean) - len(digest["history"])
        return digest

    # Drop the oldest history first, in chunks so a very long history stays cheap.
    finish()
    while _size(digest) > budget_bytes and digest["history"]:
        excess = _size(digest) - budget_bytes
        per_entry = max(_size({"h": digest["history"]}) // len(digest["history"]), 1)
        drop = max(min(excess // per_entry, len(digest["history"])), 1)
        digest["history"] = digest["history"][drop:]
        finish()
    # Then the other parts, largest-first in practice, so the bound holds whatever the caller passed.
    for key in ("control", "slopes", "current", "scales", "sentinels", "objective"):
        if _size(digest) <= budget_bytes:
            break
        digest[key] = None if key == "control" else {}
    if _size(digest) > budget_bytes:
        digest["nonfinite"] = _short_names(nonfinite)
    if _size(digest) > budget_bytes:
        digest = {
            "step": digest["step"],
            "max_steps": digest["max_steps"],
            "best_step": best_step,
            "nonfinite": _short_names(nonfinite, limit=4, width=24),
            "history": [],
            "history_dropped": len(clean),
            "truncated": True,
        }
        for fallback in ({"truncated": True}, {}):  # unreachable above the minimum budget; kept as a last resort
            if _size(digest) <= budget_bytes:
                break
            digest = fallback
    return digest
