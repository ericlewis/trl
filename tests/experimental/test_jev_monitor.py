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

import importlib.util
import json
import math
import os
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import torch

from trl.experimental.decision.jev_monitor import (
    MIN_BUDGET_BYTES,
    Ledger,
    SystemOneClient,
    build_digest,
    validate_questions,
)


def _has_decision():
    try:
        return importlib.util.find_spec("transformers.decision") is not None
    except Exception:
        return False


# Only the tests that validate questions need `transformers.decision`; the digest and ledger tests do not.
requires_decision = pytest.mark.skipif(not _has_decision(), reason="transformers.decision is not available")

Q = {"q": {"type": "noul", "instructions": "Is the run healthy?"}}
GOOD = {"model": "m", "answers": {"q": {"type": "noul", "noul": 0.75}}, "usage": {"tokens": 3}}
OBJ = {"accuracy": 1.0, "ece": -1.0}
ENV = "JEV_TEST_KEY"
VERBS = ("GET", "POST", "HEAD", "PUT", "DELETE", "PATCH", "OPTIONS")


class Server:
    """Local HTTP server with a script of behaviours, one per request (the last repeats); records every verb."""

    def __init__(self):
        self.requests = []
        self.script = [("json", 200, GOOD)]
        self.release = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def handle_any(self):
                length = int(self.headers.get("Content-Length", 0) or 0)
                body = self.rfile.read(length) if length else b""
                outer.requests.append(
                    {"method": self.command, "headers": dict(self.headers), "body": body, "path": self.path}
                )
                index = min(len(outer.requests) - 1, len(outer.script) - 1)
                outer.behave(self, outer.script[index])

        for verb in VERBS:
            setattr(Handler, f"do_{verb}", Handler.handle_any)

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        self.url = f"{self.base}/v1/systemone"
        self.thread = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        self.thread.start()

    def respond(self, *script):
        self.script = list(script)
        self.requests.clear()

    def behave(self, h, step):
        kind = step[0]
        if kind == "json":
            raw = json.dumps(step[2]).encode()
            h.send_response(step[1])
            h.send_header("Content-Type", "application/json")
            h.send_header("Content-Length", str(len(raw)))
            h.end_headers()
            h.wfile.write(raw)
        elif kind == "slow_json":
            time.sleep(step[1])
            self.behave(h, ("json", 200, GOOD))
        elif kind == "raw":
            h.send_response(200)
            h.send_header("Content-Length", str(len(step[1])))
            h.end_headers()
            h.wfile.write(step[1])
        elif kind == "raw_no_length":
            h.send_response(200)
            h.end_headers()
            h.wfile.write(step[1])
        elif kind == "redirect":
            h.send_response(step[2])
            h.send_header("Location", step[1])
            h.send_header("Content-Length", "0")
            h.end_headers()
        elif kind == "close_mid_body":
            h.send_response(200)
            h.send_header("Content-Length", "1000")
            h.end_headers()
            h.wfile.write(b'{"answ')
            h.close_connection = True
        elif kind == "close_before_response":
            h.close_connection = True
        elif kind == "hang":
            self.release.wait(5)
        elif kind == "drip_body":
            h.send_response(200)
            h.send_header("Content-Length", "100000")
            h.end_headers()
            self._drip(h, b" ", 0.05)
        elif kind == "drip_headers":
            self._drip(h, b"HTTP/1.0 200 OK\r\nX-Pad: " + b"a" * 2000, 0.05)
        elif kind == "status":
            h.send_response(step[1])
            h.send_header("Content-Length", "0")
            h.end_headers()

    def _drip(self, h, data, interval):
        try:
            for i in range(10000):
                if self.release.is_set():
                    return
                h.wfile.write(data[i : i + 1] if i < len(data) else b"a")
                h.wfile.flush()
                time.sleep(interval)
        except OSError:
            pass
        h.close_connection = True

    def close(self):
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
    if not _has_decision():
        pytest.skip("transformers.decision is not available")
    s = Server()
    yield s
    s.close()


@pytest.fixture
def other_server():
    s = Server()
    yield s
    s.close()


@pytest.fixture(autouse=True)
def fast_backoff(monkeypatch):
    monkeypatch.setattr(SystemOneClient, "_BACKOFF", 0.01)
    monkeypatch.delenv(ENV, raising=False)
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "no_proxy", "NO_PROXY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def client(server, monkeypatch):
    def make(key=None, **kwargs):
        if key is not None:
            monkeypatch.setenv(ENV, key)
        kwargs.setdefault("timeout", 0.5)
        return SystemOneClient(url=server.url, model="m", api_key_env=ENV, **kwargs)

    return make


def test_parses_real_answer_shape(server, client):
    server.respond(("json", 200, GOOD))
    c = client(key="secret")
    assert c.ask({}, Q) == {"q": {"type": "noul", "noul": 0.75}}
    request = server.requests[0]
    assert request["headers"]["Authorization"] == "Bearer secret"
    assert request["headers"]["Content-Type"] == "application/json"
    body = json.loads(request["body"])
    assert body["model"] == "m" and body["questions"] == Q and body["state"] == {}


def test_choice_and_score_shapes_pass_through(server, client):
    answers = {
        "c": {"type": "choice", "choice": "a", "confidence": 0.6, "probabilities": {"a": 0.6, "b": 0.4}},
        "s": {"type": "score", "score": 3.2, "legend": {"1": "bad"}, "confidence": 0.5, "probabilities": {"0": 1.0}},
    }
    server.respond(("json", 200, {"model": "m", "answers": answers, "usage": {}}))
    assert client(key="k").ask({}, Q) == answers


REDIRECTS = [301, 302, 303, 307, 308]
# urllib follows these for a POST (turning it into a GET) unless told not to; it refuses 307 and 308 on its own.
URLLIB_FOLLOWS = {301, 302, 303}


@pytest.mark.parametrize("status", REDIRECTS)
def test_redirect_is_refused_and_key_not_forwarded(server, other_server, client, status):
    server.respond(("redirect", other_server.url, status))
    c = client(key="secret")
    assert c.ask({}, Q) is None
    assert len(server.requests) == 1
    assert other_server.requests == []  # no request of any verb, so no header could have been forwarded
    assert "secret" not in c.last_error


@pytest.mark.parametrize("status", sorted(URLLIB_FOLLOWS))
def test_redirect_instrument_is_sensitive(server, other_server, status):
    # Self-check: with urllib's default opener (redirects on) the same servers DO show the forwarded key.
    server.respond(("redirect", other_server.url, status))
    request = urllib.request.Request(server.url, data=b"{}", headers={"Authorization": "Bearer secret"}, method="POST")
    with urllib.request.build_opener().open(request, timeout=2) as response:
        response.read()
    assert len(other_server.requests) == 1
    assert other_server.requests[0]["headers"]["Authorization"] == "Bearer secret"


@pytest.mark.parametrize(
    "behaviour",
    [
        "close_mid_body",
        "close_before_response",
        "not_json",
        "json_list",
        "answers_not_dict",
        "nan_constant",
        "hang",
        "oversized_body",
        "oversized_streaming",
        "http_500_then_500_then_500",
    ],
)
def test_failures_return_none_and_never_raise(server, client, behaviour):
    script = {
        "close_mid_body": [("close_mid_body",)],
        "close_before_response": [("close_before_response",)],
        "not_json": [("raw", b"<html>nope</html>")],
        "json_list": [("json", 200, [1, 2])],
        "answers_not_dict": [("json", 200, {"answers": [1]})],
        "nan_constant": [("raw", b'{"answers": {"q": NaN}}')],
        "hang": [("hang",)],
        "oversized_body": [("raw", b'{"answers": {}, "pad": "' + b"x" * 5000 + b'"}')],
        "oversized_streaming": [("raw_no_length", b'{"answers": {}, "pad": "' + b"x" * 5000 + b'"}')],
        "http_500_then_500_then_500": [("status", 500)],
    }[behaviour]
    server.respond(*script)
    c = client(key="secret", timeout=0.3, retries=1, max_response_bytes=1000)
    t0 = time.monotonic()
    assert c.ask({"a": 1}, Q) is None
    assert time.monotonic() - t0 < 4
    assert c.last_error and "secret" not in c.last_error
    if behaviour.startswith("oversized"):
        assert "too large" in c.last_error


@pytest.mark.parametrize("behaviour", ["hang", "drip_body", "drip_headers"])
def test_total_time_is_bounded(server, client, behaviour):
    server.respond((behaviour,))
    timeout, retries = 0.3, 1
    c = client(key="k", timeout=timeout, retries=retries)
    t0 = time.monotonic()
    assert c.ask({}, Q) is None
    elapsed = time.monotonic() - t0
    assert elapsed < timeout * (retries + 1) + 0.6  # far inside the documented timeout * (retries + 1) + 3
    assert c.last_error
    # The abandoned worker threads are daemons and leave the client's state alone.
    state = (c.last_error, c._failures, c.breaker_state)
    server.release.set()
    time.sleep(0.2)
    assert (c.last_error, c._failures, c.breaker_state) == state
    assert all(t.daemon for t in threading.enumerate() if t.name == "systemone-request")


def test_500_exhausts_retries(server, client):
    server.respond(("status", 500))
    assert client(key="k").ask({}, Q) is None
    assert len(server.requests) == 3  # 1 + retries


def test_401_is_not_retried(server, client):
    server.respond(("status", 401))
    c = client(key="k")
    assert c.ask({}, Q) is None
    assert len(server.requests) == 1
    assert c.last_error == "HTTP 401"


def test_503_is_retried_then_succeeds(server, client):
    server.respond(("status", 503), ("json", 200, GOOD))
    assert client(key="k").ask({}, Q) == GOOD["answers"]
    assert len(server.requests) == 2


class FakeClock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


def breaker_client(client, **kwargs):
    c = client(key="k", retries=0, breaker_threshold=2, breaker_cooldown=60.0, **kwargs)
    c._clock = FakeClock()
    return c


def test_breaker_opens_and_stays_open_during_cooldown(server, client):
    server.respond(("status", 401))
    c = breaker_client(client)
    assert c.breaker_state == "closed"
    for _ in range(2):
        assert c.ask({}, Q) is None
    assert c.breaker_state == "open" and len(server.requests) == 2
    c._clock.now += 59
    assert c.ask({}, Q) is None
    assert len(server.requests) == 2 and "circuit open" in c.last_error
    assert c.breaker_state == "open"


def test_breaker_sends_one_probe_after_cooldown_and_success_closes(server, client):
    server.respond(("status", 401))
    c = breaker_client(client)
    c.ask({}, Q)
    c.ask({}, Q)
    server.respond(("json", 200, GOOD))
    c._clock.now += 61
    assert c.breaker_state == "half_open"
    assert c.ask({}, Q) == GOOD["answers"]
    assert len(server.requests) == 1
    assert c.breaker_state == "closed" and c._failures == 0
    assert c.ask({}, Q) is not None and len(server.requests) == 2


def test_breaker_failed_probe_reopens_with_fresh_cooldown(server, client):
    server.respond(("status", 401))
    c = breaker_client(client)
    c.ask({}, Q)
    c.ask({}, Q)
    c._clock.now += 61
    assert c.ask({}, Q) is None and len(server.requests) == 3  # the probe
    assert c.breaker_state == "open"
    c._clock.now += 59  # 59s after the probe, not after the first opening
    assert c.ask({}, Q) is None and len(server.requests) == 3
    c._clock.now += 2
    assert c.breaker_state == "half_open"
    assert c.ask({}, Q) is None and len(server.requests) == 4


def test_breaker_half_open_allows_a_single_concurrent_probe(server, client):
    server.respond(("status", 401))
    c = breaker_client(client, **{})
    c.timeout = 3.0
    c.ask({}, Q)
    c.ask({}, Q)
    server.respond(("slow_json", 0.4))
    c._clock.now += 61
    results = []
    threads = [threading.Thread(target=lambda: results.append(c.ask({}, Q))) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(server.requests) == 1
    assert sum(r is not None for r in results) == 1
    assert c.breaker_state == "closed"


def test_breaker_reset(server, client):
    server.respond(("status", 401))
    c = breaker_client(client)
    c.ask({}, Q)
    c.ask({}, Q)
    assert c.breaker_state == "open"
    c.reset()
    assert c.breaker_state == "closed"
    c.ask({}, Q)
    assert len(server.requests) == 3


def test_unexpected_exception_counts_as_failure(server, client, monkeypatch):
    c = breaker_client(client)
    monkeypatch.setattr(c, "_post", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    assert c.ask({}, Q) is None and c._failures == 1
    assert c.ask({}, Q) is None
    assert c.breaker_state == "open"
    assert server.requests == []


def test_unexpected_exception_in_probe_reopens(server, client, monkeypatch):
    c = breaker_client(client)
    server.respond(("status", 401))
    c.ask({}, Q)
    c.ask({}, Q)
    c._clock.now += 61
    monkeypatch.setattr(c, "_post", lambda *a: (_ for _ in ()).throw(RuntimeError("boom")))
    assert c.ask({}, Q) is None
    assert c.breaker_state == "open" and not c._probing


def test_key_only_from_environment(server, monkeypatch):
    with pytest.raises(TypeError):
        SystemOneClient(url=server.url, model="m", api_key="secret")
    with pytest.raises(TypeError):
        SystemOneClient("local", model="m", key="secret")
    c = SystemOneClient(url=server.url, model="m", api_key_env=ENV)
    assert c.ask({}, Q) is None
    assert ENV in c.last_error
    assert server.requests == []
    monkeypatch.setenv(ENV, "from-env")
    assert c.ask({}, Q) == GOOD["answers"]
    assert server.requests[0]["headers"]["Authorization"] == "Bearer from-env"


def test_presets():
    c = SystemOneClient("typesafe")
    assert (c.url, c.model, c.api_key_env) == (
        "https://api.typesafe.ai/v1/systemone",
        "jev-1.13.0",
        "TYPESAFE_API_KEY",
    )
    c = SystemOneClient("openrouter")
    assert (c.url, c.model, c.api_key_env) == (
        "https://openrouter.ai/api/alpha/decisions",
        "typesafe/jev-1.13",
        "OPENROUTER_API_KEY",
    )
    c = SystemOneClient("local", model="mine")
    assert (c.url, c.model, c.api_key_env) == ("http://127.0.0.1:8000/v1/systemone", "mine", None)
    with pytest.raises(ValueError):
        SystemOneClient("nope")
    with pytest.raises(ValueError):
        SystemOneClient("local")  # no model


def test_local_preset_sends_no_authorization(server):
    c = SystemOneClient(url=server.url, model="m", timeout=0.5)
    assert c.ask({}, Q) is not None
    assert "Authorization" not in server.requests[0]["headers"]


def test_key_is_never_sent_over_plain_http_to_a_remote_host(monkeypatch):
    monkeypatch.setenv(ENV, "topsecret")
    c = SystemOneClient(url="http://example.com/v1/systemone", model="m", api_key_env=ENV)
    monkeypatch.setattr(c, "_post", lambda *a: pytest.fail("a request was made"))
    assert c.ask({}, Q) is None
    assert "non-loopback" in c.last_error and "topsecret" not in c.last_error


@requires_decision
@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "[::1]"])
def test_loopback_hosts_may_carry_a_key_over_http(monkeypatch, host):
    monkeypatch.setenv(ENV, "k")
    c = SystemOneClient(url=f"http://{host}:9/x", model="m", api_key_env=ENV, timeout=0.3, retries=0)
    posts = []
    monkeypatch.setattr(c, "_post", lambda *a: posts.append(1))
    c.ask({}, Q)
    assert posts == [1]


def test_https_remote_host_may_carry_a_key(monkeypatch):
    monkeypatch.setenv(ENV, "k")
    c = SystemOneClient("typesafe")
    c.api_key_env = ENV
    posts = []
    monkeypatch.setattr(c, "_post", lambda *a: posts.append(1))
    c.ask({}, Q)
    assert posts == [1]


def test_env_proxies_are_ignored_for_keyed_requests_by_default(server, other_server, client, monkeypatch):
    monkeypatch.setenv("http_proxy", other_server.base)
    c = client(key="k")
    assert c.ask({}, Q) == GOOD["answers"]
    assert len(server.requests) == 1 and other_server.requests == []


@pytest.mark.parametrize("key", [None, "k"])
def test_loopback_never_goes_through_a_proxy(server, other_server, client, monkeypatch, key):
    # Even when asked to honour proxies: a key sent to a loopback host over HTTP would reach the proxy in clear text
    monkeypatch.setenv("http_proxy", other_server.base)
    c = client(key=key, use_env_proxies=True)
    if key is None:
        c.api_key_env = None
    assert c.ask({}, Q) == GOOD["answers"]
    assert len(server.requests) == 1 and other_server.requests == []


@requires_decision
@pytest.mark.parametrize(
    ("use_env_proxies", "key", "expected"), [(True, "k", "env"), (False, "k", "direct"), (False, None, "env")]
)
def test_proxy_choice_for_a_remote_host(monkeypatch, use_env_proxies, key, expected):
    if key is not None:
        monkeypatch.setenv(ENV, key)
    c = SystemOneClient(
        url="https://judge.example.com/v1",
        model="m",
        api_key_env=ENV if key else None,
        use_env_proxies=use_env_proxies,
    )
    openers = []
    monkeypatch.setattr(c, "_post", lambda body, headers, opener: openers.append(opener))
    c.ask({}, Q)
    assert openers == [c._opener_env if expected == "env" else c._opener_direct]


def test_key_is_redacted_from_errors(server, client, monkeypatch):
    c = client(key="topsecret")
    monkeypatch.setattr(
        c, "_attempt", lambda *a: (_ for _ in ()).throw(RuntimeError("boom Bearer topsecret topsecret"))
    )
    assert c.ask({}, Q) is None
    assert "topsecret" not in c.last_error and "REDACTED" in c.last_error


def test_request_is_strict_json(server, client):
    c = client(key="k")
    assert c.ask({"loss": float("nan"), "lr": float("inf"), "ok": [1.5]}, Q) is not None

    def refuse(name):
        raise AssertionError(name)

    body = json.loads(server.requests[0]["body"], parse_constant=refuse)
    assert body["state"] == {"loss": None, "lr": None, "ok": [1.5]}


def test_invalid_questions_make_no_request(server, client):
    c = client(key="k")
    assert c.ask({}, {"q": {"type": "bogus", "instructions": "x"}}) is None
    assert c.ask({}, {f"q{i}": Q["q"] for i in range(65)}) is None
    assert server.requests == [] and "invalid request" in c.last_error
    assert c._failures == 0


def test_unserialisable_state_never_raises(server, client):
    class Weird:
        def __str__(self):
            raise RuntimeError("no")

    assert client(key="k").ask({"x": Weird()}, Q) == GOOD["answers"]  # sent as null, never raised
    assert json.loads(server.requests[0]["body"])["state"] == {"x": None}


def history_of(n):
    return [
        {"step": 25 * (i + 1), "accuracy": 0.5 + i / (2 * n), "ece": 0.2 - i / (10 * n), "loss": 1.0 / (i + 1)}
        for i in range(n)
    ]


def compact(d):
    return json.dumps(d, separators=(",", ":"), allow_nan=False)


def test_digest_respects_budget():
    d = build_digest(history_of(500), step=12500, max_steps=20000, objective=OBJ, scales={}, sentinels={})
    assert len(compact(d).encode()) <= 8192 and d["history_dropped"] > 0
    assert d["history"][-1]["step"] == 12500  # the newest evaluation is kept
    assert d["history_dropped"] + len(d["history"]) == 500


@pytest.mark.parametrize("budget", [-5, 0, 1, 50, 300, 1000, 4096])
def test_digest_budget_always_holds(budget):
    d = build_digest(
        history_of(200),
        step=5000,
        max_steps=9000,
        objective=OBJ,
        scales={"temperature": 1.5},
        sentinels={"a": True},
        control=history_of(200),
        budget_bytes=budget,
    )
    assert len(compact(d).encode()) <= max(budget, MIN_BUDGET_BYTES)


def test_digest_minimum_budget_is_documented_and_holds_the_reduced_digest():
    assert MIN_BUDGET_BYTES == 512 and "512" in build_digest.__doc__
    d = build_digest(history_of(3), step=1, max_steps=2, objective=OBJ, scales={}, sentinels={}, budget_bytes=1)
    assert len(compact(d).encode()) <= MIN_BUDGET_BYTES


def test_digest_adversarial_inputs_stay_within_budget():
    long = "m" * 20000
    cases = {
        "long metric names": [{"step": i, long + str(i): 1.0, "accuracy": 0.5} for i in range(5)],
        "non-ascii": [{"step": i, "точность": 0.5, "精度" * 50: 0.1} for i in range(100)],
        "huge nonfinite": [{"step": 1, **{f"bad_{i}_{'x' * 30}": float("nan") for i in range(3000)}}],
    }
    for name, history in cases.items():
        for budget in (MIN_BUDGET_BYTES, 2048, 8192):
            d = build_digest(
                history,
                step=5,
                max_steps=10,
                objective={"accuracy": 1.0, "точность": 1.0},
                scales={f"head_{i}_é": float(i) for i in range(400)},
                sentinels={"é" * 100: True},
                control=history,
                budget_bytes=budget,
            )
            assert len(compact(d).encode()) <= budget, name
            assert len(json.dumps(d, separators=(",", ":")).encode()) <= budget, name  # ASCII-escaped form too


def test_digest_content():
    h = history_of(10)
    d = build_digest(
        h, step=250, max_steps=1000, objective=OBJ, scales={"t": 2.0}, sentinels={"flat": False}, last_n=4
    )
    assert d["best_step"] == 250 and d["current"]["accuracy"] == h[-1]["accuracy"]
    assert d["slopes"]["accuracy"] > 0 and d["slopes"]["ece"] < 0
    assert d["scales"] == {"t": 2.0} and d["sentinels"] == {"flat": False}
    assert d["history_dropped"] == 0 and d["nonfinite"] == []
    h[-1]["mean_confidence"] = 0.9
    d = build_digest(h, step=250, max_steps=1000, objective=OBJ, scales={}, sentinels={})
    assert d["confidence_gap"] == pytest.approx(0.9 - h[-1]["accuracy"])


def test_digest_control():
    d = build_digest(
        history_of(5), step=125, max_steps=1000, objective=OBJ, scales={}, sentinels={}, control=history_of(3)
    )
    assert d["control"]["step"] == 75 and d["control"]["best_step"] == 75


def test_digest_nonfinite():
    d = build_digest(
        [{"step": 5, "accuracy": float("nan")}],
        step=5,
        max_steps=10,
        objective=OBJ,
        scales={"t": float("inf")},
        sentinels={},
    )
    json.dumps(d, allow_nan=False)
    assert "accuracy" in d["nonfinite"] and "t" in d["nonfinite"]
    assert d["best_step"] is None


def test_digest_empty_history():
    d = build_digest([], step=0, max_steps=10, objective=OBJ, scales={}, sentinels={})
    json.dumps(d, allow_nan=False)
    assert d["best_step"] is None and d["history"] == []


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -float("inf")])
def test_digest_nonfinite_objective_weight(weight):
    d = build_digest(
        history_of(4), step=100, max_steps=200, objective={"accuracy": weight, "ece": -1.0}, scales={}, sentinels={}
    )
    json.dumps(d, allow_nan=False)
    assert "accuracy" in d["nonfinite"] and d["objective"]["accuracy"] is None
    assert d["best_step"] is not None  # the remaining finite term still scores


def test_digest_overflowing_weighted_sum():
    history = [{"step": 1, "accuracy": 1e308, "ece": 1e308}, {"step": 2, "accuracy": 1.0, "ece": 0.0}]
    d = build_digest(history, step=2, max_steps=3, objective={"accuracy": 10.0, "ece": 10.0}, scales={}, sentinels={})
    json.dumps(d, allow_nan=False)
    assert "objective_score" in d["nonfinite"]
    assert d["best_step"] == 2 and d["best_score"] == 10.0


def test_digest_huge_integers_do_not_raise():
    history = [{"step": 1, "accuracy": 10**400}, {"step": 2, "accuracy": 0.5}]
    d = build_digest(history, step=2, max_steps=3, objective={"accuracy": 1.0}, scales={"s": 10**400}, sentinels={})
    json.dumps(d, allow_nan=False)
    assert d["best_step"] == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 10**30, "soon", None])
def test_digest_garbage_steps(bad):
    d = build_digest([{"step": bad, "accuracy": 0.5}], step=bad, max_steps=bad, objective=OBJ, scales={}, sentinels={})
    json.dumps(d, allow_nan=False)
    assert d["step"] is None and d["max_steps"] is None
    assert "step" in d["nonfinite"] and "max_steps" in d["nonfinite"]


def test_digest_nan_best_score_and_gap():
    history = [{"step": 1, "accuracy": 0.5, "ece": 0.1, "mean_confidence": float("nan")}]
    d = build_digest(history, step=1, max_steps=2, objective=OBJ, scales={}, sentinels={})
    json.dumps(d, allow_nan=False)
    assert d["confidence_gap"] is None and d["best_score"] == pytest.approx(0.4)


def test_ledger_roundtrip_and_truncate(tmp_path):
    ledger = Ledger(tmp_path / "sub" / "ledger.jsonl")
    assert ledger.read() == []
    for step in (10, 20, 30):
        assert ledger.append("ask", step, {"loss": float("nan"), "step": step}) is True
    records = ledger.read()
    assert [r["step"] for r in records] == [10, 20, 30]
    assert records[0]["kind"] == "ask" and records[0]["payload"] == {"loss": None, "step": 10}
    aside = ledger.truncate_after(20)
    assert [r["step"] for r in ledger.read()] == [10, 20]
    assert aside.parent == ledger.path.parent and aside.name.startswith("ledger.moved-")
    assert [r["step"] for r in Ledger(aside).read()] == [30]  # moved, not deleted
    assert ledger.truncate_after(20) is None  # nothing to move: no file
    ledger.append("ask", 25, {})
    assert [r["step"] for r in ledger.read()] == [10, 20, 25]
    second = ledger.truncate_after(10)
    assert second != aside and [r["step"] for r in Ledger(aside).read()] == [30]  # never overwritten
    assert Ledger(tmp_path / "missing.jsonl").truncate_after(0) is None


def test_ledger_torn_last_line_does_not_swallow_the_next_record(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.append("ask", 1, {})
    with open(ledger.path, "ab") as f:
        f.write(b'{"kind": "ask", "step": 2, "paylo')  # a crash mid-write: no trailing newline
    assert len(ledger.read()) == 1
    assert ledger.append("ask", 3, {"x": 1}) is True
    assert [r["step"] for r in ledger.read()] == [1, 3]
    assert ledger.append("ask", 4, {}) is True
    assert [r["step"] for r in ledger.read()] == [1, 3, 4]


def test_ledger_read_tolerates_undecodable_bytes(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    ledger.append("ask", 1, {})
    with open(ledger.path, "ab") as f:
        f.write(b"\xff\xfe\x80 not json\n")
        f.write(b'{"kind": "x", "step": 2, "payload": {"s": "\xff"}}\n')
    records = ledger.read()
    assert [r["step"] for r in records] == [1, 2]
    aside = ledger.truncate_after(1)
    assert [r["step"] for r in ledger.read()] == [1]
    assert b"not json" in aside.read_bytes()  # an unreadable line is moved aside too, not lost


def test_ledger_append_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    ledger = Ledger(blocker / "sub" / "l.jsonl")  # the parent is a file
    assert ledger.append("ask", 1, {}) is False
    assert ledger.last_error
    good = Ledger(tmp_path / "good.jsonl")
    assert good.append("ask", 1, {"bad": object()}) is True  # unserialisable payloads become strings
    assert good.read()[0]["payload"]["bad"].startswith("<object object")


@requires_decision
def test_validate_questions_limits():
    validate_questions(Q)
    validate_questions({f"q{i}": Q["q"] for i in range(64)})
    with pytest.raises(ValueError):
        validate_questions({f"q{i}": Q["q"] for i in range(65)})
    with pytest.raises(ValueError):
        validate_questions({"q": {"type": "bogus", "instructions": "x"}})
    many = {"q": {"type": "choice", "instructions": "x", "criteria": {f"o{i}": "d" for i in range(30)}}}
    validate_questions(many)  # the default 26-option cap does not apply


# -- JevMonitorCallback -------------------------------------------------------------------------------------------

CQ = {
    "stop": {"type": "noul", "instructions": "Should training stop now?"},
    "trend": {
        "type": "choice",
        "instructions": "Where is the run heading?",
        "criteria": {"better": None, "flat": None, "worse": None},
    },
    "health": {"type": "score", "instructions": "How healthy is the run?", "criteria": ["bad", "ok", "good"]},
}
EVAL = {"eval_loss": 0.5, "eval_accuracy": 0.8, "eval_selection_score": 0.7, "eval_runtime": 0.1, "epoch": 1.0}
FLAGS = ("should_save", "should_evaluate", "should_epoch_stop")


def stop_answer(p):
    return {"stop": {"type": "noul", "noul": p}}


class FakeBackend:
    """A backend callable: `answers` is a mapping, or a function of the 1-based call number; optional delay or gate."""

    def __init__(self, answers=None, delay=0.0, gate=None, error=None):
        self.answers = stop_answer(1.0) if answers is None else answers
        self.delay, self.gate, self.error = delay, gate, error
        self.calls = []
        self.started = threading.Event()  # set when a request has reached the backend

    def __call__(self, state, questions):
        self.calls.append(state)
        self.started.set()
        if self.gate is not None:
            self.gate.wait(5)
        if self.delay:
            time.sleep(self.delay)
        if self.error is not None:
            raise self.error
        return self.answers(len(self.calls)) if callable(self.answers) else self.answers


class FakeTrainer:
    """Records saves; writes a weights file unless `write=False`; its accelerator says whether a step was skipped."""

    def __init__(self, control=None, fail=False, write=True):
        self.callback_handler = type("Handler", (), {"callbacks": []})()
        self.accelerator = type("Accelerator", (), {"optimizer_step_was_skipped": False})()
        self.saved, self.stop_flag_at_save = [], []
        self.control, self.fail, self.write = control, fail, write

    def add_callback(self, callback):
        self.callback_handler.callbacks.append(callback)

    def save_model(self, output_dir=None, _internal_call=False):
        self.saved.append(output_dir)
        if self.control is not None:
            self.stop_flag_at_save.append(self.control.should_training_stop)
        if self.fail:
            raise OSError("disk full")
        if self.write:
            os.makedirs(output_dir, exist_ok=True)
            with open(os.path.join(output_dir, "model.safetensors"), "wb") as f:
                f.write(b"weights")


class TinyModel(torch.nn.Module):
    """A backbone (its `base_model`), a head, and one scale parameter."""

    def __init__(self, scale=1.0):
        super().__init__()
        self.backbone = torch.nn.Linear(2, 2)
        self.head = torch.nn.Linear(2, 2)
        self.logit_scale = torch.nn.Parameter(torch.tensor(float(scale)))

    @property
    def base_model(self):
        return self.backbone

    def set_grads(self, value=0.5):
        for p in self.parameters():
            p.grad = torch.full_like(p, value)


class Run:
    """Drives one callback through the events of a training run, as the Trainer would."""

    def __init__(self, callback, tmp_path, *, rank0=True, max_steps=100, logging_steps=10, model=None, fp16=False):
        from transformers import TrainerControl, TrainerState

        self.cb, self.model = callback, model
        args = {"output_dir": str(tmp_path), "logging_steps": logging_steps, "fp16": fp16}
        self.args = type("Args", (), args)()
        self.state = TrainerState(max_steps=max_steps, is_world_process_zero=rank0)
        self.control = TrainerControl()

    def begin(self, step=0):
        self.state.global_step = step
        self.cb.on_train_begin(self.args, self.state, self.control, model=self.model)
        return self

    def evaluate(self, step, wait=True, **metrics):
        self.state.global_step = step
        self.cb.on_evaluate(self.args, self.state, self.control, metrics={**EVAL, **metrics}, model=self.model)
        if wait:
            assert self.cb.wait(5)
        return self.control.should_training_stop

    def optimizer_step(self, step):
        """One optimizer step from global step `step`: the hooks around `optimizer.step()`, as the Trainer calls them."""
        self.state.global_step = step
        self.cb.on_step_begin(self.args, self.state, self.control)
        self.cb.on_pre_optimizer_step(self.args, self.state, self.control, model=self.model)
        self.cb.on_optimizer_step(self.args, self.state, self.control, model=self.model)
        self.state.global_step = step + 1
        self.cb.on_step_end(self.args, self.state, self.control)

    def end(self):
        self.cb.on_train_end(self.args, self.state, self.control, model=self.model)


def kinds(callback, kind=None):
    records = callback.ledger.read()
    return [r for r in records if r["kind"] == kind] if kind else [r["kind"] for r in records]


def monitor(tmp_path, backend=None, *, rules=(), trainer=None, **kwargs):
    from trl.experimental.decision.jev_monitor import JevMonitorCallback

    kwargs.setdefault("finish_timeout", 2.0)
    callback = JevMonitorCallback(
        backend, kwargs.pop("questions", CQ), stop_when=rules, output_dir=tmp_path / "monitor", **kwargs
    )
    if trainer is not None:
        callback.attach(trainer)
    return callback


def rule(**kwargs):
    from trl.experimental.decision.jev_monitor import StopRule

    return StopRule(**{"question": "stop", **kwargs})


@requires_decision
def test_exports():
    from trl.experimental.decision import JevMonitorCallback, StopRule
    from trl.experimental.decision.jev_monitor import JevMonitorCallback as A
    from trl.experimental.decision.jev_monitor import StopRule as B

    assert JevMonitorCallback is A and StopRule is B


@requires_decision
def test_default_questions_are_valid_and_generic(tmp_path):
    from trl.experimental.decision.jev_monitor import DEFAULT_QUESTIONS, JevMonitorCallback

    validate_questions(DEFAULT_QUESTIONS)
    text = json.dumps(DEFAULT_QUESTIONS).lower()
    for word in ("clef", "jev", "qwen", "klm", "rlcd", "cosine", "joint", "typed-decisions", "eval_"):
        assert word not in text
    callback = JevMonitorCallback(FakeBackend(), output_dir=tmp_path)
    assert callback.questions == DEFAULT_QUESTIONS and callback.questions is not DEFAULT_QUESTIONS
    # User questions replace the defaults
    assert JevMonitorCallback(FakeBackend(), CQ, output_dir=tmp_path).questions == CQ


@requires_decision
def test_construction_errors(tmp_path):
    from trl.experimental.decision.jev_monitor import JevMonitorCallback, StopRule

    backend = FakeBackend()
    for kwargs in (
        {"mode": "auto"},
        {"stop_when": [StopRule("nope")]},  # unknown question
        {"stop_when": [StopRule("trend", option="sideways")]},  # unknown option of a choice question
        {"stop_when": [StopRule("trend")]},  # a choice rule names its option
        {"stop_when": [StopRule("stop", option="true")]},  # a noul rule reads `noul`, no option
        {"stop_when": [StopRule("health", option="3")]},  # score levels are "0", "1", "2"
        {"stop_when": ["stop"]},
        {"patience": 0},
        {"grad_norm_every": 0},
        {"metric_prefix": ""},
        {"finish_timeout": -1},
    ):
        with pytest.raises(ValueError):
            JevMonitorCallback(backend, CQ, output_dir=tmp_path, **kwargs)
    with pytest.raises(ValueError):
        JevMonitorCallback(backend, {"q": {"type": "bogus", "instructions": "x"}}, output_dir=tmp_path)
    with pytest.raises(ValueError):  # rules need a backend to ask
        JevMonitorCallback(None, CQ, stop_when=[StopRule("stop")], output_dir=tmp_path)
    for threshold in (0.0, 1.5, float("nan")):
        with pytest.raises(ValueError):
            StopRule("stop", threshold=threshold)
    # Valid: choice and score options, and enforce with no backend (sentinels only)
    JevMonitorCallback(
        backend, CQ, stop_when=[StopRule("trend", "worse"), StopRule("health", "0")], output_dir=tmp_path
    )
    JevMonitorCallback(None, CQ, mode="enforce", output_dir=tmp_path)


@requires_decision
def test_advise_never_changes_control(tmp_path):
    model = TinyModel(scale=9.0)
    callback = monitor(tmp_path, FakeBackend(), rules=[rule()], patience=1, grad_norm_every=1)
    run = Run(callback, tmp_path, model=model)

    def unchanged(event):
        before = dict(vars(run.control))
        event()
        assert dict(vars(run.control)) == before

    unchanged(run.begin)
    for step in (10, 20, 30, 40):
        model.set_grads(float("inf") if step in (20, 30) else 0.5)
        loss = float("nan") if step == 40 else 0.5
        unchanged(lambda step=step: run.optimizer_step(step - 1))
        unchanged(lambda step=step, loss=loss: run.evaluate(step, eval_loss=loss))
    unchanged(run.end)
    assert not run.control.should_training_stop
    assert kinds(callback, "advice")  # the rule and the sentinels held, and were only advised
    assert kinds(callback, "sentinel")


@requires_decision
@pytest.mark.parametrize("initial", [False, True])
def test_never_sets_save_or_evaluate_or_epoch_stop(tmp_path, initial):
    model = TinyModel(scale=9.0)
    trainer = FakeTrainer()
    callback = monitor(tmp_path, FakeBackend(), rules=[rule()], mode="enforce", patience=1, trainer=trainer)
    run = Run(callback, tmp_path, model=model, logging_steps=1)
    for flag in FLAGS:
        setattr(run.control, flag, initial)

    def check():
        assert all(getattr(run.control, flag) is initial for flag in FLAGS)

    run.begin()
    check()
    for step in (10, 20, 30):
        model.set_grads(float("nan") if step == 30 else 0.5)
        run.optimizer_step(step - 1)
        check()
        run.evaluate(step, eval_loss=float("inf") if step == 30 else 0.5)
        check()
    run.end()
    check()
    assert run.control.should_training_stop and trainer.saved


@requires_decision
def test_does_not_block(tmp_path):
    slow_backend = FakeBackend(delay=2.0)
    callback = monitor(tmp_path, slow_backend, finish_timeout=0.2)
    run = Run(callback, tmp_path).begin()
    t0 = time.perf_counter()
    run.evaluate(10, wait=False)
    assert time.perf_counter() - t0 < 1.0  # fsync'd ledger writes only: the backend takes 2 s
    assert slow_backend.started.wait(5)
    t0 = time.perf_counter()
    run.evaluate(20, wait=False)  # the first request is still running
    assert time.perf_counter() - t0 < 1.0
    t0 = time.perf_counter()
    run.end()  # waits for the worker at most `finish_timeout`
    assert time.perf_counter() - t0 < 1.0
    assert kinds(callback, "end")[0]["payload"]["worker_idle"] is False


@requires_decision
def test_nothing_written_after_training(tmp_path):
    gate = threading.Event()
    backend = FakeBackend(gate=gate)
    callback = monitor(tmp_path, backend, finish_timeout=0.05)
    run = Run(callback, tmp_path).begin()
    run.evaluate(10, wait=False)
    assert backend.started.wait(5)
    run.end()
    before = callback.ledger.read()
    assert before[-1]["kind"] == "end"
    gate.set()  # the answer arrives after training ended: dropped, not recorded
    assert callback.wait(5)
    assert callback.ledger.read() == before


@requires_decision
def test_verdict_applied_at_next_eval(tmp_path):
    backend = FakeBackend(lambda n: stop_answer(0.1 * n))
    callback = monitor(tmp_path, backend, rules=[rule()])
    run = Run(callback, tmp_path).begin()
    run.evaluate(10)
    assert kinds(callback, "verdict") == []
    assert [r["step"] for r in kinds(callback, "answer")] == [10]  # received, not applied yet
    run.evaluate(20)
    run.end()
    (verdict,) = kinds(callback, "verdict")
    assert verdict["step"] == 20 and verdict["payload"]["for_step"] == 10
    assert verdict["payload"]["rules"][0]["probability"] == pytest.approx(0.1)
    # The digest sent for each evaluation is the run's state at that step, its metrics without the prefix
    assert [state["step"] for state in backend.calls] == [10, 20]
    assert backend.calls[1]["current"]["accuracy"] == 0.8 and "runtime" not in backend.calls[1]["current"]
    assert backend.calls[0]["max_steps"] == 100
    # The final evaluation's verdict is never applied, but on_train_end waits so the ledger records it
    assert [r["step"] for r in kinds(callback, "answer")] == [10, 20]
    assert [r["step"] for r in kinds(callback, "eval")] == [10, 20]


@requires_decision
def test_late_verdict_is_recorded_missing(tmp_path):
    gate = threading.Event()
    backend = FakeBackend(gate=gate)
    trainer = FakeTrainer()
    callback = monitor(tmp_path, backend, rules=[rule()], mode="enforce", patience=1, trainer=trainer)
    run = Run(callback, tmp_path).begin()
    assert not run.evaluate(10, wait=False)
    assert backend.started.wait(5)  # the worker took the request for 10
    assert not run.evaluate(20, wait=False)  # the verdict for 10 is still running
    (missing,) = kinds(callback, "missing")
    assert missing["step"] == 20 and missing["payload"]["for_step"] == 10
    gate.set()
    assert callback.wait(5)
    # The verdict for 10 arrived late: recorded, discarded, never applied
    (late,) = kinds(callback, "late")
    assert late["step"] == 10
    assert not run.control.should_training_stop and trainer.saved == []
    assert run.evaluate(30)  # the verdict for 20 arrived in time and applies
    assert [r["payload"]["for_step"] for r in kinds(callback, "verdict")] == [20]
    run.end()


@requires_decision
def test_enforce_stops_after_patience(tmp_path):
    from transformers import TrainerControl

    control = TrainerControl()
    trainer = FakeTrainer(control)
    callback = monitor(tmp_path, FakeBackend(stop_answer(0.95)), rules=[rule()], mode="enforce", trainer=trainer)
    run = Run(callback, tmp_path)
    run.control = control
    run.begin()
    stopped = [run.evaluate(step) for step in (10, 20, 30, 40)]
    assert stopped == [False, False, False, True]
    assert trainer.saved == [str(tmp_path / "monitor-stop-40")]
    assert trainer.stop_flag_at_save == [False]  # saved before the flag was set
    record = json.loads((tmp_path / "monitor" / "intervened.json").read_text())
    assert record["step"] == 40 and record["reason"] == "rule"
    assert record["rule"] == {"question": "stop", "option": None, "threshold": 0.9}
    assert [v["for_step"] for v in record["verdicts"]] == [10, 20, 30]
    assert all(v["probability"] == 0.95 for v in record["verdicts"])
    assert record["checkpoint"] == str(tmp_path / "monitor-stop-40")
    # Once stopped, a further evaluation (the Trainer's end-of-epoch one) saves nothing more
    run.evaluate(50)
    run.end()
    assert len(trainer.saved) == 1


@requires_decision
def test_streak_resets_on_disagreement(tmp_path):
    sequence = [0.95, 0.95, 0.5, 0.95, 0.95, 0.95, 0.95]
    backend = FakeBackend(lambda n: stop_answer(sequence[n - 1]))
    callback = monitor(tmp_path, backend, rules=[rule()], mode="enforce", trainer=FakeTrainer())
    run = Run(callback, tmp_path).begin()
    stopped = [run.evaluate(step) for step in (10, 20, 30, 40, 50, 60, 70)]
    # Verdicts for 10, 20 agree, 30 disagrees (reset), 40, 50, 60 agree: the third applies at 70
    assert stopped == [False] * 6 + [True]
    statuses = [r["payload"]["rules"][0]["status"] for r in kinds(callback, "verdict")]
    assert statuses == ["agree", "agree", "disagree", "agree", "agree", "agree"]
    run.end()


@requires_decision
def test_missing_or_mistyped_answer_is_missing(tmp_path):
    answers = [
        stop_answer(0.95),
        {},  # lacks the rule's question
        {"stop": {"type": "score", "score": 1.0, "probabilities": {"0": 1.0}}},  # wrong answer type
        {"stop": "garbage"},
        {"stop": {"type": "noul", "noul": float("nan")}},
        {"stop": {"type": "noul", "noul": 1.5}},
        "not a mapping",
        None,  # the backend gave no answer
        stop_answer(0.95),
    ]
    backend = FakeBackend(lambda n: answers[n - 1])
    callback = monitor(tmp_path, backend, rules=[rule()], mode="enforce", patience=2, trainer=FakeTrainer())
    run = Run(callback, tmp_path).begin()
    steps = [10 * (i + 1) for i in range(len(answers) + 1)]
    stopped = [run.evaluate(step) for step in steps]
    # Missing answers neither break nor extend the streak: the two agreeing verdicts stop at the end
    assert stopped == [False] * len(answers) + [True]
    missing_for = sorted({r["payload"]["for_step"] for r in kinds(callback, "missing")})
    assert missing_for == steps[1:8]
    run.end()


@requires_decision
def test_choice_and_score_rules(tmp_path):
    answer = {
        "trend": {"type": "choice", "choice": "worse", "confidence": 0.9, "probabilities": {"worse": 0.92}},
        "health": {"type": "score", "score": 0.1, "confidence": 0.8, "probabilities": {"0": 0.5}},
    }
    rules = [rule(question="health", option="0", threshold=0.6), rule(question="trend", option="worse")]
    callback = monitor(tmp_path, FakeBackend(answer), rules=rules, mode="enforce", patience=1, trainer=FakeTrainer())
    run = Run(callback, tmp_path).begin()
    assert [run.evaluate(10), run.evaluate(20)] == [False, True]
    record = json.loads((tmp_path / "monitor" / "intervened.json").read_text())
    assert record["rule"]["question"] == "trend"
    run.end()


@requires_decision
def test_enforce_requires_attach(tmp_path):
    callback = monitor(tmp_path, FakeBackend(), rules=[rule()], mode="enforce")
    with pytest.raises(RuntimeError, match=r"attach\(trainer\)"):
        Run(callback, tmp_path).begin()
    # Advise needs no trainer
    Run(monitor(tmp_path, FakeBackend()), tmp_path).begin().end()


@requires_decision
def test_attach_adds_the_callback_once(tmp_path):
    trainer = FakeTrainer()
    callback = monitor(tmp_path, FakeBackend(), trainer=trainer)
    assert callback.attach(trainer) is callback
    assert trainer.callback_handler.callbacks == [callback]


@requires_decision
def test_only_configured_prefix_and_only_during_training(tmp_path):
    backend = FakeBackend()
    callback = monitor(tmp_path, backend, rules=[rule()], mode="enforce", patience=1, trainer=FakeTrainer())
    run = Run(callback, tmp_path)
    run.evaluate(10, eval_loss=float("nan"))  # before training: ignored
    assert not callback.ledger.path.exists()
    run.begin()
    count = len(kinds(callback))
    test_metrics = {"test_loss": float("nan"), "test_accuracy": 0.5}
    run.cb.on_evaluate(run.args, run.state, run.control, metrics=test_metrics)
    run.cb.on_evaluate(run.args, run.state, run.control, metrics=None)
    assert len(kinds(callback)) == count and backend.calls == []
    run.end()
    count = len(kinds(callback))
    run.evaluate(20, eval_loss=float("nan"))  # after training: ignored
    assert len(kinds(callback)) == count and backend.calls == []
    assert not run.control.should_training_stop


class Wire:
    """A fake `broadcast`: rank 0 sends, the other ranks receive what it sent, in order."""

    def __init__(self):
        self.sent, self.received = [], 0

    def rank0(self, value):
        self.sent.append(value)
        return value

    def other(self, value):
        assert value is None  # a non-zero rank proposes nothing
        self.received += 1
        return self.sent[self.received - 1]


@requires_decision
def test_decision_is_broadcast(tmp_path):
    wire = Wire()
    backend0, backend1 = FakeBackend(), FakeBackend()
    trainer0, trainer1 = FakeTrainer(), FakeTrainer()
    kwargs = {"rules": [rule()], "mode": "enforce", "patience": 1}
    cb0 = monitor(tmp_path / "r0", backend0, broadcast=wire.rank0, trainer=trainer0, **kwargs)
    cb1 = monitor(tmp_path / "r1", backend1, broadcast=wire.other, trainer=trainer1, **kwargs)
    run0 = Run(cb0, tmp_path / "r0").begin()
    run1 = Run(cb1, tmp_path / "r1", rank0=False).begin()
    for step in (10, 20):
        run0.evaluate(step)
        run1.evaluate(step)
        assert run0.control.should_training_stop == run1.control.should_training_stop == (step == 20)
    run0.end()
    run1.end()
    # Rank 0 sent its decision, then whether the checkpoint was saved
    assert wire.sent[0] is None and wire.sent[1]["reason"] == "rule" and wire.sent[2] is True
    # The other rank never asked the backend and wrote nothing; it joined the (collective) save
    assert backend1.calls == [] and not (tmp_path / "r1" / "monitor").exists()
    assert trainer1.saved == trainer0.saved == [str(tmp_path / "r0" / "monitor-stop-20")]
    assert (tmp_path / "r0" / "monitor" / "intervened.json").exists()


@requires_decision
def test_broadcast_value_wins_on_a_non_zero_rank(tmp_path):
    replies = iter([{"step": 10, "reason": "rule"}, True])  # rank 0's decision, then "the checkpoint was saved"

    def broadcast(value):
        assert value is None
        return next(replies)

    backend = FakeBackend(stop_answer(0.0))
    callback = monitor(tmp_path, backend, mode="enforce", broadcast=broadcast, trainer=FakeTrainer())
    run = Run(callback, tmp_path, rank0=False).begin()
    assert run.evaluate(10)
    run.end()
    assert backend.calls == [] and not (tmp_path / "monitor").exists()
    # A broadcast "no" keeps training whatever this rank's own view
    callback = monitor(tmp_path, backend, mode="enforce", broadcast=lambda value: None, trainer=FakeTrainer())
    run = Run(callback, tmp_path).begin()
    assert not run.evaluate(10, eval_loss=float("nan"))
    run.end()


@requires_decision
def test_save_failure_does_not_stop(tmp_path):
    trainer = FakeTrainer(fail=True)
    callback = monitor(tmp_path, FakeBackend(), rules=[rule()], mode="enforce", patience=1, trainer=trainer)
    run = Run(callback, tmp_path).begin()
    assert [run.evaluate(10), run.evaluate(20), run.evaluate(30)] == [False, False, False]
    assert len(trainer.saved) == 2  # tried at 20 and again at 30
    errors = [r["payload"] for r in kinds(callback, "error")]
    assert errors and all(e["where"] == "save" and "disk full" in e["error"] for e in errors)
    assert not (tmp_path / "monitor" / "intervened.json").exists()
    trainer.fail = False
    assert run.evaluate(40)
    run.end()


@requires_decision
def test_sentinels(tmp_path):
    # (i) Non-finite evaluation metrics: a stop at that evaluation, without a backend
    trainer = FakeTrainer()
    callback = monitor(tmp_path / "i", None, mode="enforce", trainer=trainer)
    run = Run(callback, tmp_path / "i").begin()
    assert not run.evaluate(10)
    assert run.evaluate(20, eval_kl=float("inf"))
    (sentinel,) = kinds(callback, "sentinel")
    assert sentinel["payload"]["sentinel"] == "nonfinite_metrics" and sentinel["payload"]["metrics"] == ["eval_kl"]
    assert json.loads((tmp_path / "i" / "monitor" / "intervened.json").read_text())["reason"] == "sentinel"
    run.end()

    # (ii) Non-finite gradients: ledgered at once, a stop at the next evaluation, on every rank
    wire = Wire()
    model = TinyModel()
    cb0 = monitor(tmp_path / "ii", None, mode="enforce", broadcast=wire.rank0, trainer=FakeTrainer())
    cb1 = monitor(tmp_path / "ii1", None, mode="enforce", broadcast=wire.other, trainer=FakeTrainer())
    run0 = Run(cb0, tmp_path / "ii", model=model, logging_steps=5).begin()
    run1 = Run(cb1, tmp_path / "ii1", model=model, logging_steps=5, rank0=False).begin()
    model.set_grads(0.5)
    run0.optimizer_step(4)
    model.backbone.weight.grad[0, 0] = float("nan")
    for step in (9, 14):  # two consecutive checks on taken steps (patience_gradients=2)
        run0.optimizer_step(step)
        run1.optimizer_step(step)
    sentinels = kinds(cb0, "sentinel")
    assert [r["step"] for r in sentinels] == [10, 15] and sentinels[0]["payload"]["groups"] == ["backbone"]
    assert not run0.control.should_training_stop
    run0.evaluate(15)
    run1.evaluate(15)
    assert run0.control.should_training_stop and run1.control.should_training_stop
    run0.end()
    run1.end()

    # (iii) A scale parameter above log(100): ledgered only, never a stop
    trainer = FakeTrainer()
    callback = monitor(tmp_path / "iii", None, mode="enforce", trainer=trainer)
    run = Run(callback, tmp_path / "iii", model=TinyModel(scale=math.log(100) + 0.1)).begin()
    assert not run.evaluate(10) and not run.evaluate(20)
    (sentinel,) = kinds(callback, "sentinel")  # once, when it crosses
    assert sentinel["payload"]["sentinel"] == "scale_above_limit" and sentinel["payload"]["name"] == "logit_scale"
    assert trainer.saved == []
    run.end()


@requires_decision
def test_scales_and_gradient_norms_reach_the_digest(tmp_path):
    backend = FakeBackend()
    model = TinyModel(scale=2.0)
    callback = monitor(tmp_path, backend, grad_norm_every=1)
    run = Run(callback, tmp_path, model=model).begin()
    model.set_grads(1.0)
    run.optimizer_step(0)
    run.evaluate(1)
    run.end()
    scales = backend.calls[0]["scales"]
    assert scales["logit_scale"] == pytest.approx(2.0)
    assert scales["grad_norm_backbone"] == pytest.approx(math.sqrt(6.0))  # 4 weights + 2 biases of 1.0
    assert scales["grad_norm_head"] == pytest.approx(math.sqrt(7.0))  # and the scale
    assert backend.calls[0]["sentinels"] == {
        "nonfinite_gradients": False,
        "nonfinite_metrics": False,
        "scale_above_limit": False,
    }


@requires_decision
def test_grad_norm_sync_cadence(tmp_path, monkeypatch):
    model = TinyModel()
    model.set_grads(0.5)
    callback = monitor(tmp_path, FakeBackend(), grad_norm_every=10)
    calls = []
    routine = callback._gradient_norms
    monkeypatch.setattr(callback, "_gradient_norms", lambda m: calls.append(1) or routine(m))
    run = Run(callback, tmp_path, model=model).begin()
    items = []
    original_item = torch.Tensor.item
    monkeypatch.setattr(torch.Tensor, "item", lambda t: items.append(1) or original_item(t))
    syncs = []
    for step in range(30):
        before = len(items)
        run.optimizer_step(step)
        syncs.append(len(items) - before)
    monkeypatch.setattr(torch.Tensor, "item", original_item)
    assert len(calls) == 3
    # One read per group (backbone, head) on the checked steps, none on the others
    assert syncs == [2 if (step + 1) % 10 == 0 else 0 for step in range(30)]
    run.end()
    # The default cadence is `logging_steps`
    callback = monitor(tmp_path, FakeBackend())
    calls.clear()
    routine = callback._gradient_norms
    monkeypatch.setattr(callback, "_gradient_norms", lambda m: calls.append(1) or routine(m))
    run = Run(callback, tmp_path, model=model, logging_steps=5).begin()
    for step in range(30):
        run.optimizer_step(step)
    assert len(calls) == 6
    run.end()


@requires_decision
def test_resume_restores_streak_from_ledger(tmp_path):
    first = monitor(tmp_path, FakeBackend(stop_answer(0.95)), rules=[rule()])
    run = Run(first, tmp_path).begin()
    for step in (10, 20, 30, 40):
        run.evaluate(step)
    run.end()
    assert max(r["step"] for r in first.ledger.read()) == 40
    # Resume from the checkpoint at step 30: what came after it is dropped, the streak is rebuilt (verdicts for 10, 20)
    second = monitor(tmp_path, FakeBackend(stop_answer(0.95)), rules=[rule()], mode="enforce", trainer=FakeTrainer())
    run = Run(second, tmp_path).begin(step=30)
    assert all(r["step"] <= 30 for r in second.ledger.read() if r["kind"] != "start")
    assert [r["payload"]["for_step"] for r in kinds(second, "verdict")] == [10, 20]
    assert [r["payload"]["from_step"] for r in kinds(second, "resume")] == [30]
    assert not run.evaluate(40)  # nothing pending to apply
    assert run.evaluate(50)  # the verdict for 40 is the third agreeing one
    record = json.loads((tmp_path / "monitor" / "intervened.json").read_text())
    assert [v["for_step"] for v in record["verdicts"]] == [10, 20, 40]
    run.end()


@requires_decision
def test_control_arm_included_when_given(tmp_path):
    control = Ledger(tmp_path / "control.jsonl")
    for step, accuracy in ((10, 0.5), (20, 0.6), (30, 0.9)):
        control.append("eval", step, {"metrics": {"accuracy": accuracy, "selection_score": accuracy}})
    control.append("verdict", 20, {"for_step": 10})
    backend = FakeBackend()
    callback = monitor(tmp_path, backend, control_ledger=control.path)
    run = Run(callback, tmp_path).begin()
    run.evaluate(20)
    run.end()
    # Matched steps only: the control arm's rows up to the current step
    assert backend.calls[0]["control"]["step"] == 20
    assert backend.calls[0]["control"]["current"]["accuracy"] == 0.6
    # Unreadable control file: a ledgered warning, once, and no control in the digest
    backend = FakeBackend()
    callback = monitor(tmp_path / "other", backend, control_ledger=tmp_path)  # a directory
    run = Run(callback, tmp_path / "other").begin()
    run.evaluate(10)
    run.evaluate(20)
    run.end()
    assert backend.calls[0]["control"] is None
    assert len(kinds(callback, "warning")) == 1


@requires_decision
def test_backend_exception_never_reaches_trainer(tmp_path, monkeypatch):
    from trl.experimental.decision.jev_monitor import callback as callback_module

    backend = FakeBackend(error=RuntimeError("backend exploded"))
    callback = monitor(tmp_path, backend, rules=[rule()], mode="enforce", trainer=FakeTrainer())
    run = Run(callback, tmp_path).begin()
    run.evaluate(10)
    run.evaluate(20)
    (answer, *_) = kinds(callback, "answer")
    assert "backend exploded" in answer["payload"]["error"]
    assert [r["payload"]["for_step"] for r in kinds(callback, "missing")] == [10]
    # An exception inside a hook is ledgered, not raised
    monkeypatch.setattr(callback_module, "build_digest", lambda *a, **k: 1 / 0)
    run.evaluate(30)
    monkeypatch.setattr(callback, "_gradient_norms", lambda model: 1 / 0)
    run.model = TinyModel()
    run.optimizer_step(9)
    run.end()
    assert {r["payload"]["where"] for r in kinds(callback, "error")} >= {"on_evaluate", "on_pre_optimizer_step"}
    assert not run.control.should_training_stop


@requires_decision
def test_ledger_failure_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    from trl.experimental.decision.jev_monitor import JevMonitorCallback

    callback = JevMonitorCallback(FakeBackend(), CQ, output_dir=blocker / "monitor", finish_timeout=1.0)
    run = Run(callback, tmp_path).begin()
    run.evaluate(10)
    run.evaluate(20)
    run.end()
    assert callback.ledger.last_error


def clef_fixtures():
    pytest.importorskip("transformers.decision")
    from transformers import AutoTokenizer

    from .test_decision_clef import MODEL_ID, clef_decision_config, make_trainer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer, clef_decision_config(), make_trainer


@requires_decision
def test_real_trainer_advise_run(tmp_path):
    from trl.experimental.decision import compute_calibration_metrics
    from trl.experimental.decision.jev_monitor import JevMonitorCallback

    tokenizer, decision_config, make_trainer = clef_fixtures()
    trainer, _, _ = make_trainer(
        str(tmp_path), tokenizer, decision_config, compute_metrics=compute_calibration_metrics, eval_steps=5
    )
    states = []
    callback = JevMonitorCallback(lambda state, questions: states.append(state) or stop_answer(0.99), CQ)
    callback.attach(trainer)
    trainer.train()
    assert trainer.state.global_step == 20
    evals = kinds(callback, "eval")
    assert [r["step"] for r in evals] == [5, 10, 15, 20]
    assert set(evals[0]["payload"]["metrics"]) >= {"loss", "accuracy", "kl", "brier", "ece", "selection_score"}
    assert callback.ledger.path == tmp_path / "jev_monitor" / "ledger.jsonl"
    # A request replaced before the worker took it is never sent, so only the order of the steps sent is certain
    steps = [state["step"] for state in states]
    assert steps and steps == sorted(set(steps)) and set(steps) <= {5, 10, 15, 20}
    # An evaluation after training is ignored
    trainer.evaluate()
    assert len(kinds(callback, "eval")) == 4


@requires_decision
def test_real_trainer_enforced_stop(tmp_path):
    from transformers import TrainerCallback

    from trl.experimental.decision.jev_monitor import JevMonitorCallback, StopRule

    tokenizer, decision_config, make_trainer = clef_fixtures()
    trainer, _, _ = make_trainer(str(tmp_path), tokenizer, decision_config, eval_steps=5, save_strategy="no")
    callback = JevMonitorCallback(
        lambda state, questions: stop_answer(0.99),
        CQ,
        mode="enforce",
        stop_when=[StopRule("stop")],
        patience=1,
    )

    class WaitForVerdict(TrainerCallback):  # runs before the monitor: the verdict is ready when it applies
        def on_evaluate(self, *args, **kwargs):
            callback.wait(5)

    trainer.add_callback(WaitForVerdict())
    callback.attach(trainer)
    trainer.train()
    assert trainer.state.global_step == 10  # the verdict for 5 applied at 10
    assert (tmp_path / "monitor-stop-10" / "config.json").exists()
    record = json.loads((tmp_path / "jev_monitor" / "intervened.json").read_text())
    assert record["step"] == 10 and record["reason"] == "rule"


@requires_decision
def test_skipped_overflow_step_is_a_note_not_a_sentinel(tmp_path):
    model = TinyModel()
    trainer = FakeTrainer()
    trainer.accelerator.optimizer_step_was_skipped = True  # fp16: the scaler skipped the step
    callback = monitor(tmp_path, None, mode="enforce", trainer=trainer, grad_norm_every=1, patience_gradients=1)
    run = Run(callback, tmp_path, model=model, fp16=True).begin()
    model.set_grads(float("inf"))
    for step in range(5):
        run.optimizer_step(step)
    assert not run.evaluate(5)
    assert kinds(callback, "sentinel") == [] and len(kinds(callback, "note")) == 5
    assert trainer.saved == []
    run.end()


@requires_decision
def test_overflow_without_a_trainer_under_fp16_is_a_note(tmp_path):
    model = TinyModel()
    backend = FakeBackend()
    callback = monitor(tmp_path, backend, grad_norm_every=1)
    run = Run(callback, tmp_path, model=model, fp16=True).begin()
    model.set_grads(float("nan"))
    run.optimizer_step(0)
    run.evaluate(1)
    run.end()
    assert kinds(callback, "sentinel") == [] and kinds(callback, "note")
    assert backend.calls[0]["sentinels"]["nonfinite_gradients"] is False


@requires_decision
def test_nonfinite_gradients_on_taken_steps_need_consecutive_checks(tmp_path):
    model = TinyModel()
    trainer = FakeTrainer()
    callback = monitor(tmp_path, None, mode="enforce", trainer=trainer, grad_norm_every=1)
    run = Run(callback, tmp_path, model=model).begin()
    model.set_grads(float("nan"))
    run.optimizer_step(0)
    assert not run.evaluate(1)  # one check is not enough (patience_gradients=2)
    model.set_grads(0.5)
    run.optimizer_step(1)  # a clean check clears it
    model.set_grads(float("nan"))
    run.optimizer_step(2)
    assert not run.evaluate(3)
    run.optimizer_step(3)  # the second consecutive non-finite check
    assert run.evaluate(4)
    assert [r["payload"]["consecutive"] for r in kinds(callback, "sentinel")] == [1, 1, 2]
    assert json.loads((tmp_path / "monitor" / "intervened.json").read_text())["sentinel"]["step"] == 4
    run.end()


@requires_decision
def test_gradient_flag_clears_after_a_clean_check(tmp_path):
    model = TinyModel()
    backend = FakeBackend()
    callback = monitor(tmp_path, backend, grad_norm_every=1)
    run = Run(callback, tmp_path, model=model).begin()
    model.set_grads(float("inf"))
    run.optimizer_step(0)
    run.evaluate(1)
    model.set_grads(0.5)
    run.optimizer_step(1)
    run.evaluate(2)
    run.end()
    assert [state["sentinels"]["nonfinite_gradients"] for state in backend.calls] == [True, False]


@requires_decision
def test_stop_needs_a_saved_checkpoint(tmp_path):
    trainer = FakeTrainer(write=False)  # save_model returns without writing (as with a sharded state dict)
    callback = monitor(tmp_path, FakeBackend(), rules=[rule()], mode="enforce", patience=1, trainer=trainer)
    run = Run(callback, tmp_path).begin()
    assert [run.evaluate(10), run.evaluate(20)] == [False, False]
    assert trainer.saved == [str(tmp_path / "monitor-stop-20")]
    (error,) = kinds(callback, "error")
    assert error["payload"]["where"] == "save" and "no weights" in error["payload"]["error"]
    assert not (tmp_path / "monitor" / "intervened.json").exists()
    # A directory with only the training arguments holds no weights either
    os.makedirs(tmp_path / "monitor-stop-30")
    (tmp_path / "monitor-stop-30" / "training_args.bin").write_bytes(b"args")
    assert not run.evaluate(30)
    run.end()


@requires_decision
def test_answer_from_an_earlier_run_is_ignored(tmp_path):
    gate = threading.Event()
    backend = FakeBackend(lambda n: stop_answer(0.95 if n == 1 else 0.1), gate=gate)
    trainer = FakeTrainer()
    callback = monitor(tmp_path, backend, rules=[rule()], mode="enforce", patience=1, trainer=trainer)
    callback.finish_timeout = 0.05
    first = Run(callback, tmp_path).begin()
    first.evaluate(10, wait=False)
    assert backend.started.wait(5)
    first.end()  # the first run's request is still running
    second = Run(callback, tmp_path).begin()
    second.evaluate(10, wait=False)  # the second run's own request for 10
    gate.set()
    assert callback.wait(5)
    assert not second.evaluate(20)  # its own verdict (0.1) applies, not the first run's (0.95)
    records = callback.ledger.read()
    start = max(i for i, r in enumerate(records) if r["kind"] == "start")
    answers = [r for r in records[start:] if r["kind"] in ("answer", "late")]
    assert [r["payload"]["answers"]["stop"]["noul"] for r in answers] == [0.1, 0.1]
    assert [r["payload"]["rules"][0]["probability"] for r in kinds(callback, "verdict")] == [0.1]
    assert trainer.saved == []
    second.end()


@requires_decision
def test_several_evaluations_at_one_step_are_merged(tmp_path):
    gate = threading.Event()
    backend = FakeBackend(gate=gate)
    callback = monitor(tmp_path, backend, rules=[rule()])
    run = Run(callback, tmp_path).begin()
    first = {"eval_a_loss": 0.5, "eval_a_accuracy": 0.7, "eval_a_runtime": 1.0}
    second = {"eval_b_loss": 0.6, "eval_b_accuracy": 0.8, "eval_b_runtime": 1.0}
    run.state.global_step = 10
    run.cb.on_evaluate(run.args, run.state, run.control, metrics=first)
    assert backend.started.wait(5)
    run.cb.on_evaluate(run.args, run.state, run.control, metrics=second)
    gate.set()
    assert callback.wait(5)
    run.state.global_step = 20
    run.cb.on_evaluate(run.args, run.state, run.control, metrics={**first, **second})
    run.end()
    assert kinds(callback, "missing") == []  # the second evaluation at 10 did not discard the verdict for 10
    # Once per run, a note says the request already sent covers the first dataset only
    (note,) = [r for r in kinds(callback, "note") if r["payload"]["note"] == "several_evaluations"]
    assert note["step"] == 10 and "metric_prefix" in note["payload"]["reason"]
    assert [r["payload"]["for_step"] for r in kinds(callback, "verdict")] == [10]
    assert [state["step"] for state in backend.calls] == [10, 20]  # one request per step
    assert backend.calls[1]["history"][0] == {
        "step": 10,
        "a_loss": 0.5,
        "a_accuracy": 0.7,
        "b_loss": 0.6,
        "b_accuracy": 0.8,
    }
    # On resume the two records of step 10 merge again
    resumed = monitor(tmp_path, FakeBackend())
    Run(resumed, tmp_path).begin(step=10)
    assert resumed._history == [backend.calls[1]["history"][0]]
    resumed.close()


@requires_decision
def test_evaluation_after_a_failed_train_is_ignored(tmp_path):
    backend = FakeBackend()
    callback = monitor(tmp_path, backend)
    run = Run(callback, tmp_path).begin()
    run.optimizer_step(9)
    run.evaluate(10)
    run.cb.on_step_begin(run.args, run.state, run.control)  # the next step raises: no on_step_end, no on_train_end
    before, calls = kinds(callback), len(backend.calls)
    run.evaluate(10)  # a standalone trainer.evaluate() afterwards
    run.evaluate(10)
    assert kinds(callback) == before + ["note", "note"] and len(backend.calls) == calls
    assert callback._worker is not None and callback._worker.is_alive()  # nothing was closed


# -- never raise into training, never switch the monitor off ----------------------------------------------------


class Regression(torch.nn.Module):
    """A one-layer model the Trainer can train on CPU in milliseconds, with no download."""

    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(4, 1)

    def forward(self, x, labels=None):
        y = self.layer(x).squeeze(-1)
        return {"loss": ((y - labels) ** 2).mean(), "logits": y}


class Rows(torch.utils.data.Dataset):
    def __init__(self, n=8):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"x": torch.full((4,), float(i) / self.n), "labels": torch.tensor(float(i % 2))}


class Stream(torch.utils.data.IterableDataset):
    def __init__(self, n):
        self.n = n

    def __iter__(self):
        for i in range(self.n):
            yield {"x": torch.full((4,), float(i) / self.n), "labels": torch.tensor(float(i % 2))}


def tiny_trainer(tmp_path, train_dataset=None, eval_dataset=None, callbacks=(), **kwargs):
    from transformers import Trainer, TrainingArguments

    defaults = {
        "per_device_train_batch_size": 2,
        "eval_strategy": "epoch",
        "save_strategy": "no",
        "report_to": [],
        "use_cpu": True,
        "logging_steps": 1,
        "disable_tqdm": True,
        "num_train_epochs": 3,
    }
    args = TrainingArguments(output_dir=str(tmp_path), **{**defaults, **kwargs})
    return Trainer(
        model=Regression(),
        args=args,
        train_dataset=Rows() if train_dataset is None else train_dataset,
        eval_dataset=Rows() if eval_dataset is None else eval_dataset,
        callbacks=list(callbacks),
    )


class WaitFor:
    """A callback that runs before the monitor's `on_evaluate` and waits for the pending verdict."""

    def __new__(cls, callback):
        from transformers import TrainerCallback

        class _Wait(TrainerCallback):
            def on_evaluate(self, *args, **kwargs):
                callback.wait(5)

        return _Wait()


@requires_decision
def test_checkpoint_check_error_is_recorded_and_both_ranks_go_on(tmp_path, monkeypatch):
    from trl.experimental.decision.jev_monitor import callback as callback_module

    def walk_error(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(callback_module, "_has_weights", walk_error)
    wire = Wire()
    kwargs = {"rules": [rule()], "mode": "enforce", "patience": 1}
    cb0 = monitor(tmp_path / "r0", FakeBackend(), broadcast=wire.rank0, trainer=FakeTrainer(), **kwargs)
    cb1 = monitor(tmp_path / "r1", FakeBackend(), broadcast=wire.other, trainer=FakeTrainer(), **kwargs)
    run0 = Run(cb0, tmp_path / "r0").begin()
    run1 = Run(cb1, tmp_path / "r1", rank0=False).begin()
    for step in (10, 20):
        assert not run0.evaluate(step)
        assert not run1.evaluate(step)  # it received the second broadcast: it would wait forever otherwise
    run0.end()
    run1.end()
    assert wire.sent[1]["reason"] == "rule" and wire.sent[2] is False
    errors = [r["payload"] for r in kinds(cb0, "error")]
    assert any("PermissionError" in e["error"] for e in errors)
    assert not (tmp_path / "r0" / "monitor" / "intervened.json").exists()


@requires_decision
def test_checkpoint_check_error_never_reaches_a_real_trainer(tmp_path, monkeypatch):
    from trl.experimental.decision.jev_monitor import JevMonitorCallback, StopRule
    from trl.experimental.decision.jev_monitor import callback as callback_module

    def walk_error(path):
        raise PermissionError(13, "Permission denied", str(path))

    monkeypatch.setattr(callback_module, "_has_weights", walk_error)
    callback = JevMonitorCallback(
        lambda state, questions: stop_answer(0.99), CQ, mode="enforce", stop_when=[StopRule("stop")], patience=1
    )
    trainer = tiny_trainer(tmp_path, callbacks=[WaitFor(callback)])
    callback.attach(trainer)
    trainer.train()  # does not raise
    assert trainer.state.global_step == 12
    assert any("PermissionError" in r["payload"]["error"] for r in kinds(callback, "error"))


@requires_decision
def test_real_trainer_streaming_epoch_evaluations_are_watched(tmp_path):
    from trl.experimental.decision.jev_monitor import JevMonitorCallback

    seen = []
    callback = JevMonitorCallback(lambda state, questions: seen.append(state["step"]) or stop_answer(0.1), CQ)
    # 5 batches of 2 per pass with accumulation 2: every pass ends inside a window (a step begun, never ended)
    trainer = tiny_trainer(
        tmp_path, train_dataset=Stream(10), callbacks=[WaitFor(callback)], gradient_accumulation_steps=2, max_steps=6
    )
    callback.attach(trainer)
    trainer.train()
    assert [r["step"] for r in kinds(callback, "eval")] == [2, 4, 6]
    assert seen == [2, 4, 6]
    assert [r["payload"]["for_step"] for r in kinds(callback, "verdict")] == [2, 4]


@requires_decision
def test_epoch_end_rearms(tmp_path):
    backend = FakeBackend()
    callback = monitor(tmp_path, backend)
    run = Run(callback, tmp_path).begin()
    run.cb.on_step_begin(run.args, run.state, run.control)  # the last, partial window of a pass
    run.cb.on_epoch_end(run.args, run.state, run.control)
    run.evaluate(10)
    run.end()
    assert [r["step"] for r in kinds(callback, "eval")] == [10] and len(backend.calls) == 1


@requires_decision
def test_disarmed_evaluation_is_ignored_with_a_note_and_closes_nothing(tmp_path):
    backend = FakeBackend()
    callback = monitor(tmp_path, backend)
    run = Run(callback, tmp_path).begin()
    run.cb.on_step_begin(run.args, run.state, run.control)
    run.evaluate(10)
    run.evaluate(10)
    assert backend.calls == [] and kinds(callback, "eval") == []
    notes = kinds(callback, "note")
    assert len(notes) == 2 and all(n["payload"]["note"] == "evaluation_ignored" for n in notes)
    assert "step" in notes[0]["payload"]["reason"]
    # The monitor is still on: the next step that ends re-arms it
    run.cb.on_step_end(run.args, run.state, run.control)
    run.evaluate(20)
    run.end()
    assert [r["step"] for r in kinds(callback, "eval")] == [20] and len(backend.calls) == 1


@requires_decision
def test_resume_failure_still_starts_the_worker(tmp_path, monkeypatch):
    first = monitor(tmp_path, FakeBackend())
    run = Run(first, tmp_path).begin()
    run.evaluate(10)
    run.end()

    def full_disk(self, step):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(Ledger, "truncate_after", full_disk)
    backend = FakeBackend()
    second = monitor(tmp_path, backend)
    run = Run(second, tmp_path).begin(step=10)
    run.evaluate(20)
    run.end()
    assert [state["step"] for state in backend.calls] == [20]
    records = second.ledger.read()
    assert any(r["kind"] == "error" and r["payload"]["where"] == "resume" for r in records)
    assert sum(r["kind"] == "start" for r in records) == 2


@requires_decision
def test_a_stop_is_attempted_at_most_once_per_step(tmp_path):
    trainer = FakeTrainer(fail=True)
    callback = monitor(tmp_path, FakeBackend(), rules=[rule()], mode="enforce", patience=1, trainer=trainer)
    run = Run(callback, tmp_path).begin()
    run.evaluate(10)
    run.evaluate(20)
    run.evaluate(20)  # a second evaluation dataset at the same step
    run.end()
    assert trainer.saved == [str(tmp_path / "monitor-stop-20")]


@requires_decision
def test_no_stop_at_the_final_evaluation(tmp_path):
    trainer = FakeTrainer()
    callback = monitor(tmp_path, FakeBackend(), rules=[rule()], mode="enforce", patience=1, trainer=trainer)
    run = Run(callback, tmp_path, max_steps=20).begin()
    run.evaluate(10)
    run.control.should_training_stop = True  # max_steps reached: the run ends after this evaluation anyway
    run.evaluate(20)
    run.end()
    assert trainer.saved == [] and not (tmp_path / "monitor" / "intervened.json").exists()
    assert not (tmp_path / "monitor-stop-20").exists()
    (note,) = [n for n in kinds(callback, "note") if n["payload"]["note"] == "stop_not_needed"]
    assert note["step"] == 20 and note["payload"]["proposal"]["reason"] == "rule"


# -- record what happened ---------------------------------------------------------------------------------------

LOGGER = "trl.experimental.decision.jev_monitor.callback"


def messages(caplog, level):
    import logging

    return [r.getMessage() for r in caplog.records if r.name == LOGGER and r.levelno == getattr(logging, level)]


@requires_decision
def test_stop_is_not_enforced_unless_recorded(tmp_path, caplog):
    blocker = tmp_path / "file"
    blocker.write_text("x")  # the monitor's directory cannot be created
    wire = Wire()
    kwargs = {"rules": [rule()], "mode": "enforce", "patience": 1, "finish_timeout": 1.0}
    from trl.experimental.decision.jev_monitor import JevMonitorCallback

    trainer = FakeTrainer()
    cb0 = JevMonitorCallback(
        FakeBackend(),
        CQ,
        output_dir=blocker / "monitor",
        broadcast=wire.rank0,
        stop_when=kwargs.pop("rules"),
        **kwargs,
    ).attach(trainer)
    cb1 = monitor(
        tmp_path / "r1", FakeBackend(), rules=[rule()], broadcast=wire.other, trainer=FakeTrainer(), **kwargs
    )
    run0 = Run(cb0, tmp_path).begin()
    run1 = Run(cb1, tmp_path / "r1", rank0=False).begin()
    with caplog.at_level("WARNING", logger=LOGGER):
        for step in (10, 20):
            assert not run0.evaluate(step)
            assert not run1.evaluate(step)
    run0.end()
    run1.end()
    assert trainer.saved == [str(tmp_path / "monitor-stop-20")]  # saved, but nothing could record the stop
    assert wire.sent[2] is False  # what the other ranks receive: "saved and recorded"
    (error,) = messages(caplog, "ERROR")
    assert "not enforced" in error and "20" in error
    ledger_warnings = [m for m in messages(caplog, "WARNING") if "ledger" in m]
    assert len(ledger_warnings) == 1  # once, on the first ledger failure


@requires_decision
@pytest.mark.parametrize("broken", ["intervened.json", "ledger"])
def test_stop_is_enforced_when_either_record_is_written(tmp_path, monkeypatch, caplog, broken):
    callback = monitor(tmp_path, FakeBackend(), rules=[rule()], mode="enforce", patience=1, trainer=FakeTrainer())
    run = Run(callback, tmp_path).begin()
    if broken == "intervened.json":
        (tmp_path / "monitor" / "intervened.json").mkdir(parents=True)  # cannot be replaced by a file
    else:
        append = callback.ledger.append
        monkeypatch.setattr(callback.ledger, "append", lambda k, s, p: False if k == "intervened" else append(k, s, p))
    run.evaluate(10)
    with caplog.at_level("WARNING", logger=LOGGER):
        assert run.evaluate(20)
    run.end()
    assert any("stopped training at step 20" in m for m in messages(caplog, "WARNING"))
    if broken == "intervened.json":
        assert [r["step"] for r in kinds(callback, "intervened")] == [20]
    else:
        assert json.loads((tmp_path / "monitor" / "intervened.json").read_text())["step"] == 20


@requires_decision
def test_failed_save_and_first_backend_failure_are_logged(tmp_path, server, caplog, monkeypatch):
    monkeypatch.setenv(ENV, "sekrit-key")
    server.respond(("status", 401))
    client = SystemOneClient(url=server.url, model="m", api_key_env=ENV, timeout=0.5)
    trainer = FakeTrainer(fail=True)
    callback = monitor(tmp_path, client, mode="enforce", trainer=trainer)
    run = Run(callback, tmp_path).begin()
    with caplog.at_level("WARNING", logger=LOGGER):
        for step in (10, 20, 30):
            run.evaluate(step, eval_loss=float("nan") if step == 30 else 0.5)
    run.end()
    warnings = messages(caplog, "WARNING")
    judge = [m for m in warnings if "judge" in m]
    assert len(judge) == 1 and "HTTP 401" in judge[0]  # once, whatever the number of failures
    assert any("could not save" in m and "disk full" in m for m in warnings)
    assert all("sekrit" not in r.getMessage() for r in caplog.records)


@requires_decision
def test_answer_records_hold_the_digest_sent(tmp_path):
    gate = threading.Event()
    backend = FakeBackend(gate=gate)
    callback = monitor(tmp_path, backend, budget_bytes=2048)
    run = Run(callback, tmp_path).begin()
    run.evaluate(10, wait=False)
    assert backend.started.wait(5)
    run.evaluate(20, wait=False)  # the request for 10 is late
    gate.set()
    assert callback.wait(5)
    run.evaluate(30)
    run.end()
    records = {r["step"]: r for r in kinds(callback, "answer") + kinds(callback, "late")}
    assert records[10]["kind"] == "late"
    sent = {state["step"]: state for state in backend.calls}
    for step, record in records.items():
        assert record["payload"]["digest"] == sent[step]
        assert len(compact(record["payload"]["digest"]).encode()) <= 2048
    assert "slopes" in records[30]["payload"]["digest"] and "sentinels" in records[30]["payload"]["digest"]


def moved_files(directory, stem):
    return sorted(p.name for p in directory.iterdir() if p.name.startswith(stem + ".moved-"))


@requires_decision
def test_a_new_run_in_a_reused_directory_moves_the_earlier_records_aside(tmp_path):
    def stop_run():
        trainer = FakeTrainer()
        callback = monitor(tmp_path, FakeBackend(), rules=[rule()], mode="enforce", patience=1, trainer=trainer)
        run = Run(callback, tmp_path).begin()
        run.evaluate(10)
        assert run.evaluate(20)
        run.end()
        return callback

    first = stop_run()
    earlier = first.ledger.read()
    assert any(r["kind"] == "intervened" for r in earlier)
    directory = tmp_path / "monitor"
    second = monitor(tmp_path, FakeBackend())
    Run(second, tmp_path).begin().end()
    # Nothing is deleted: the first run's records, and its intervened.json, are kept beside the new ledger
    (aside,) = moved_files(directory, "ledger")
    assert Ledger(directory / aside).read() == earlier
    assert not (directory / "intervened.json").exists()
    (old_json,) = moved_files(directory, "intervened")
    assert json.loads((directory / old_json).read_text())["step"] == 20
    (note,) = [r for r in kinds(second, "note") if r["payload"]["note"] == "earlier_records_moved"]
    assert note["payload"]["records"] == len(earlier) and note["payload"]["to"].endswith(aside)
    assert note["payload"]["intervened_json_to"].endswith(old_json)
    # A third run never overwrites what the second moved aside
    stop_run()
    Run(monitor(tmp_path, FakeBackend()), tmp_path).begin().end()
    assert len(moved_files(directory, "ledger")) == 3 and len(moved_files(directory, "intervened")) == 2


@requires_decision
def test_resume_from_before_a_stop_moves_its_record_aside(tmp_path):
    trainer = FakeTrainer()
    first = monitor(tmp_path, FakeBackend(), rules=[rule()], mode="enforce", patience=1, trainer=trainer)
    run = Run(first, tmp_path).begin()
    for step in (10, 20):
        run.evaluate(step)
    run.end()
    resumed = monitor(tmp_path, FakeBackend())
    Run(resumed, tmp_path).begin(step=10).end()
    directory = tmp_path / "monitor"
    assert not (directory / "intervened.json").exists() and len(moved_files(directory, "intervened")) == 1
    (aside,) = moved_files(directory, "ledger")
    assert {r["kind"] for r in Ledger(directory / aside).read()} >= {"intervened", "verdict", "eval"}
    assert all(r["step"] <= 10 for r in resumed.ledger.read() if r["kind"] not in ("end",))


@requires_decision
def test_resume_after_a_stop_is_stopped_only_by_new_verdicts(tmp_path):
    trainer = FakeTrainer()
    first = monitor(tmp_path, FakeBackend(stop_answer(0.99)), rules=[rule()], mode="enforce", trainer=trainer)
    run = Run(first, tmp_path).begin()
    assert [run.evaluate(step) for step in (10, 20, 30, 40)] == [False, False, False, True]
    run.end()
    # Resume from step 40 with the same rule: the streak that stopped the first run is not restored
    second = monitor(tmp_path, FakeBackend(stop_answer(0.0)), rules=[rule()], mode="enforce", trainer=FakeTrainer())
    run = Run(second, tmp_path).begin(step=40)
    assert (tmp_path / "monitor" / "intervened.json").exists()  # step 40 is kept, and so is its record
    assert [run.evaluate(step) for step in (50, 60, 70)] == [False, False, False]
    run.end()
    new = [r for r in kinds(second, "verdict") if r["step"] > 40]
    assert len(new) == 2 and all(rule["streak"] == 0 for r in new for rule in r["payload"]["rules"])


@requires_decision
def test_persistent_nonfinite_gradients_record_transitions_only(tmp_path):
    model = TinyModel()
    callback = monitor(tmp_path, None, grad_norm_every=1)
    run = Run(callback, tmp_path, model=model).begin()
    model.set_grads(float("nan"))
    for step in range(10):
        run.optimizer_step(step)
    model.set_grads(0.5)
    run.optimizer_step(10)
    run.end()
    assert [r["payload"]["consecutive"] for r in kinds(callback, "sentinel")] == [1, 2]
    (cleared,) = [r for r in kinds(callback, "note") if r["payload"]["note"] == "finite_gradients"]
    assert cleared["step"] == 11 and cleared["payload"]["after"] == 10


# -- client hardening -------------------------------------------------------------------------------------------


def test_key_surrounding_whitespace_is_stripped(server, client):
    c = client(key="  k-with-newline\n")
    assert c.ask({}, Q) == GOOD["answers"]
    assert server.requests[0]["headers"]["Authorization"] == "Bearer k-with-newline"


@requires_decision
def test_key_with_inner_whitespace_is_refused_and_never_leaks(server, client, tmp_path):
    c = client(key="tok3nhead tok3ntail")
    assert c.ask({}, Q) is None
    assert server.requests == []
    assert ENV in c.last_error and "whitespace" in c.last_error and "tok3n" not in c.last_error
    # Nor through the monitor, which writes the client's error into its ledger
    callback = monitor(tmp_path, c)
    run = Run(callback, tmp_path).begin()
    run.evaluate(10)
    run.end()
    assert kinds(callback, "answer") and "tok3n" not in callback.ledger.path.read_text()


def test_abandoned_attempt_ends_against_a_slow_drip(server, client):
    server.respond(("drip_body",))  # one byte every 50 ms, never the whole body
    c = client(key="k", timeout=0.3, retries=0)
    before = set(threading.enumerate())
    assert c.ask({}, Q) is None
    attempts = [t for t in set(threading.enumerate()) - before if t.name == "systemone-request"]
    # The server is still dripping: the attempt stops at its own deadline instead of waiting for 64 KiB
    for thread in attempts:
        thread.join(2)
    assert not any(t.is_alive() for t in attempts)


# -- several datasets and small fixes ---------------------------------------------------------------------------


@requires_decision
def test_metrics_are_matched_by_exact_name_or_dataset(tmp_path):
    backend = FakeBackend()
    callback = monitor(tmp_path, backend)
    run = Run(callback, tmp_path).begin()
    run.evaluate(10, eval_kl=0.1, eval_reverse_kl=0.3, eval_soft_nll=0.2)  # one dataset: `eval_runtime`
    run.state.global_step = 20
    datasets = {
        "eval_validation_kl": 0.1,
        "eval_validation_reverse_kl": 0.3,
        "eval_validation_selection_score": 0.5,
        "eval_validation_runtime": 1.0,
    }
    run.cb.on_evaluate(run.args, run.state, run.control, metrics=datasets)
    run.end()
    first, second = (r["payload"]["metrics"] for r in kinds(callback, "eval"))
    assert "kl" in first and "soft_nll" in first and "reverse_kl" not in first
    assert set(second) == {"validation_kl", "validation_selection_score"}
    # `metric_prefix` selects one dataset: its metrics are named as with one dataset, and the objective scores them
    backend = FakeBackend()
    callback = monitor(tmp_path / "one", backend, metric_prefix="eval_validation")
    run = Run(callback, tmp_path / "one").begin()
    run.state.global_step = 10
    run.cb.on_evaluate(run.args, run.state, run.control, metrics=datasets)
    assert callback.wait(5)
    run.end()
    assert backend.calls[0]["current"] == {"kl": 0.1, "selection_score": 0.5}
    assert backend.calls[0]["best_step"] == 10


@requires_decision
def test_large_gradients_are_finite_norms_not_sentinels(tmp_path):
    model = TinyModel()
    backend = FakeBackend()
    callback = monitor(tmp_path, backend, grad_norm_every=1)
    run = Run(callback, tmp_path, model=model).begin()
    model.set_grads(1e20)  # its square overflows float32
    run.optimizer_step(0)
    run.evaluate(1)
    run.end()
    assert kinds(callback, "sentinel") == []
    assert backend.calls[0]["scales"]["grad_norm_backbone"] == pytest.approx(1e20 * math.sqrt(6.0), rel=1e-5)


@requires_decision
def test_worker_clears_itself_on_exit_and_a_new_run_gets_one(tmp_path):
    backend = FakeBackend()
    callback = monitor(tmp_path, backend)
    Run(callback, tmp_path).begin()
    worker = callback._worker
    callback.close()
    worker.join(5)
    assert not worker.is_alive() and callback._worker is None
    run = Run(callback, tmp_path).begin()
    assert callback._worker is not worker and callback._worker.is_alive()
    run.evaluate(10)
    run.end()
    assert len(backend.calls) == 1


def nested(depth):
    root = current = []
    for _ in range(depth):
        current.append([])
        current = current[0]
    return root


def test_digest_never_raises_on_huge_integers_or_deep_nesting():
    history = [
        {"step": 1, "accuracy": 0.5, "huge": 10**5000, "deep": nested(5000)},
        {"step": 2, "accuracy": 0.6},
    ]
    d = build_digest(
        history, step=2, max_steps=3, objective={"accuracy": 1.0}, scales={"s": -(10**5000)}, sentinels={}
    )
    assert len(compact(d).encode()) <= 8192
    assert d["best_step"] == 2 and "huge" in d["nonfinite"] and "s" in d["nonfinite"]
    assert d["history"][0]["huge"] is None


def test_ledger_accepts_deep_or_huge_payloads(tmp_path):
    ledger = Ledger(tmp_path / "l.jsonl")
    assert ledger.append("answer", 1, {"answers": {"junk": nested(5000), "n": 10**5000}}) is True
    assert ledger.read()[0]["payload"]["answers"]["n"] is None
