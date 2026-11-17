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
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

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


def test_env_proxies_are_used_when_asked(server, other_server, client, monkeypatch):
    monkeypatch.setenv("http_proxy", other_server.base)
    c = client(key="k", use_env_proxies=True)
    assert c.ask({}, Q) == GOOD["answers"]
    assert server.requests == [] and len(other_server.requests) == 1


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

    assert client(key="k").ask({"x": Weird()}, Q) is None


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
    ledger.truncate_after(20)
    assert [r["step"] for r in ledger.read()] == [10, 20]
    ledger.append("ask", 25, {})
    assert [r["step"] for r in ledger.read()] == [10, 20, 25]
    Ledger(tmp_path / "missing.jsonl").truncate_after(0)


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
    ledger.truncate_after(1)
    assert [r["step"] for r in ledger.read()] == [1]


def test_ledger_append_never_raises(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    ledger = Ledger(blocker / "sub" / "l.jsonl")  # the parent is a file
    assert ledger.append("ask", 1, {}) is False
    assert ledger.last_error
    assert ledger.append("ask", 1, {"bad": object()}) in (True, False)  # unserialisable payloads become strings


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
