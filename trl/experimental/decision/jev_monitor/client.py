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

import http.client
import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from transformers.decision import DecisionConfig, parse_questions

from .digest import json_safe


MAX_QUESTIONS = 64
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

# preset -> (url, model, environment variable holding the API key)
PRESETS: dict[str, tuple[str, str | None, str | None]] = {
    "typesafe": ("https://api.typesafe.ai/v1/systemone", "jev-1.13.0", "TYPESAFE_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/alpha/decisions", "typesafe/jev-1.13", "OPENROUTER_API_KEY"),
    "local": ("http://127.0.0.1:8000/v1/systemone", None, None),
}


def validate_questions(questions: dict) -> None:
    """
    Check typed questions before they are sent.

    Raises:
        `ValueError`: if there are more than 64 questions, or any question is not a valid `noul`, `choice` or `score`
        question according to `transformers.decision.parse_questions`.
    """
    if not isinstance(questions, dict):
        raise ValueError("`questions` must be a dict mapping a name to a question.")
    if len(questions) > MAX_QUESTIONS:
        raise ValueError(f"At most {MAX_QUESTIONS} questions per request, got {len(questions)}.")
    # `labels` of `None` lifts the default per-type option caps: the monitor sends small, hand-written questions.
    parse_questions(questions, DecisionConfig(labels={"noul": None, "choice": None, "score": None}))


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


class SystemOneClient:
    """
    Minimal client of a System One decision endpoint that never raises into the caller.

    The API key is read from an environment variable at request time and is never accepted as an argument, logged or
    included in `last_error`.

    Args:
        preset (`str`, *optional*):
            One of `"typesafe"`, `"openrouter"` or `"local"`; sets the URL, the model and the key variable.
        url (`str`, *optional*):
            Endpoint, overriding the preset's.
        model (`str`, *optional*):
            Model name, overriding the preset's. Required when the preset does not set one.
        api_key_env (`str`, *optional*):
            Name of the environment variable holding the key, overriding the preset's. No key is sent without one.
        timeout (`float`, *optional*, defaults to `10.0`):
            Seconds allowed for each attempt, connection and body included.
        retries (`int`, *optional*, defaults to `2`):
            Extra attempts after a connection error or an HTTP 429, 500, 502, 503 or 504.
        max_response_bytes (`int`, *optional*, defaults to `262144`):
            Larger responses are rejected.
        breaker_threshold (`int`, *optional*, defaults to `3`):
            After this many consecutive failed calls to `ask`, further calls return `None` without any request.
    """

    _BACKOFF = 0.25  # seconds, doubled per retry

    def __init__(
        self,
        preset: str | None = None,
        *,
        url: str | None = None,
        model: str | None = None,
        api_key_env: str | None = None,
        timeout: float = 10.0,
        retries: int = 2,
        max_response_bytes: int = 262144,
        breaker_threshold: int = 3,
    ):
        p_url, p_model, p_env = (None, None, None)
        if preset is not None:
            if preset not in PRESETS:
                raise ValueError(f"Unknown preset {preset!r}, expected one of {sorted(PRESETS)}.")
            p_url, p_model, p_env = PRESETS[preset]
        self.url = url or p_url
        self.model = model or p_model
        self.api_key_env = api_key_env or p_env
        if not self.url:
            raise ValueError("Give a `preset` or a `url`.")
        if urllib.parse.urlparse(self.url).scheme not in ("http", "https"):
            raise ValueError("`url` must be http or https.")
        if not self.model:
            raise ValueError("Give a `model`.")
        self.timeout = float(timeout)
        self.retries = max(int(retries), 0)
        self.max_response_bytes = int(max_response_bytes)
        self.breaker_threshold = int(breaker_threshold)
        self.last_error: str | None = None
        self._failures = 0
        self._lock = threading.Lock()
        self._opener = urllib.request.build_opener(_NoRedirect)

    def _key(self) -> str | None:
        return os.environ.get(self.api_key_env) or None if self.api_key_env else None

    def _redact(self, text: str) -> str:
        key = self._key()
        if key:
            text = text.replace(key, "[REDACTED]")
        return re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text)

    def _fail(self, message: str) -> None:
        self.last_error = self._redact(message)[:500]

    def ask(self, state, questions: dict) -> dict | None:
        """
        Post `state` and typed `questions`, and return the `answers` mapping, or `None` on any failure.

        The reason of the latest failure is in `last_error`.
        """
        try:
            return self._ask(state, questions)
        except BaseException as e:  # noqa: BLE001 - nothing may escape into the training loop
            if isinstance(e, (KeyboardInterrupt, SystemExit)):
                raise
            try:
                self._fail(f"{type(e).__name__}: {e}")
            except Exception:
                self.last_error = "unexpected error"
            return None

    def _ask(self, state, questions: dict) -> dict | None:
        with self._lock:
            if self._failures >= self.breaker_threshold:
                self._fail("circuit open after repeated failures")
                return None
        try:
            validate_questions(questions)
            body = json.dumps(
                {"model": self.model, "state": json_safe(state), "questions": json_safe(questions)},
                allow_nan=False,
            ).encode("utf-8")
        except (ValueError, TypeError, RecursionError) as e:
            self._fail(f"invalid request: {e}")
            return None
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if self.api_key_env:
            key = self._key()
            if not key:
                self._fail(f"environment variable {self.api_key_env} is not set")
                return None
            headers["Authorization"] = f"Bearer {key}"

        answers = self._post(body, headers)
        with self._lock:
            self._failures = 0 if answers is not None else self._failures + 1
        return answers

    def _post(self, body: bytes, headers: dict) -> dict | None:
        start = time.monotonic()
        deadline = start + self.timeout * (self.retries + 1) + 3
        for attempt in range(self.retries + 1):
            retryable = True
            try:
                return self._attempt(body, headers)
            except urllib.error.HTTPError as e:
                self._fail(f"HTTP {e.code}")
                retryable = e.code in _RETRY_STATUSES
                e.close()
            except _Rejected as e:
                self._fail(str(e))
                retryable = False
            except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
                reason = getattr(e, "reason", e)
                self._fail(f"connection error: {type(reason).__name__}: {reason}")
            if not retryable or attempt == self.retries:
                return None
            delay = min(self._BACKOFF * 2**attempt, 2.0)
            if time.monotonic() + delay >= deadline:
                return None
            time.sleep(delay)
        return None

    def _attempt(self, body: bytes, headers: dict) -> dict:
        request = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        deadline = time.monotonic() + self.timeout
        with self._opener.open(request, timeout=self.timeout) as response:
            length = response.headers.get("Content-Length")
            if length is not None and length.isdigit() and int(length) > self.max_response_bytes:
                raise _Rejected("response too large")
            chunks, total = [], 0
            while True:
                if time.monotonic() > deadline:
                    raise _Rejected("response too slow")
                chunk = response.read(min(65536, self.max_response_bytes + 1 - total))
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > self.max_response_bytes:
                    raise _Rejected("response too large")
        try:
            payload = json.loads(b"".join(chunks).decode("utf-8"), parse_constant=_reject_constant)
        except (ValueError, RecursionError) as e:
            raise _Rejected(f"response is not valid JSON ({type(e).__name__})") from None
        if not isinstance(payload, dict):
            raise _Rejected("response is not a JSON object")
        answers = payload.get("answers")
        if not isinstance(answers, dict):
            raise _Rejected("response has no `answers` mapping")
        return answers


class _Rejected(Exception):
    """A response that arrived but cannot be used; never retried."""


def _reject_constant(name: str):
    raise ValueError(f"non-finite constant {name}")
