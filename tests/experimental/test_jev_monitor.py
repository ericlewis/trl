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
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


pytest.importorskip("transformers.decision")

from trl.experimental.decision.jev_monitor import Ledger, SystemOneClient, build_digest, validate_questions


Q = {"q": {"type": "noul", "instructions": "Is the run healthy?"}}
GOOD = {"model": "m", "answers": {"q": {"type": "noul", "noul": 0.75}}, "usage": {"tokens": 3}}
OBJ = {"accuracy": 1.0, "ece": -1.0}
ENV = "JEV_TEST_KEY"


class Server:
    """Local HTTP server with a script of behaviours, one per request; the last one repeats."""

    def __init__(self):
        self.requests = []
        self.script = [("json", 200, GOOD)]
        self.release = threading.Event()
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                outer.requests.append({"headers": dict(self.headers), "body": body, "path": self.path})
                index = min(len(outer.requests) - 1, len(outer.script) - 1)
                outer.behave(self, outer.script[index])

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/v1/systemone"
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
        elif kind == "raw":
            h.send_response(200)
            h.send_header("Content-Length", str(len(step[1])))
            h.end_headers()
            h.wfile.write(step[1])
        elif kind == "redirect":
            h.send_response(302)
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
        elif kind == "status":
            h.send_response(step[1])
            h.send_header("Content-Length", "0")
            h.end_headers()

    def close(self):
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def server():
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


def client(server, key=None, monkeypatch=None, **kwargs):
    import os

    if key is not None:
        os.environ[ENV] = key
    kwargs.setdefault("timeout", 0.5)
    return SystemOneClient(url=server.url, model="m", api_key_env=ENV, **kwargs)


def test_parses_real_answer_shape(server):
    server.respond(("json", 200, GOOD))
    c = client(server, key="secret")
    assert c.ask({}, Q) == {"q": {"type": "noul", "noul": 0.75}}
    request = server.requests[0]
    assert request["headers"]["Authorization"] == "Bearer secret"
    assert request["headers"]["Content-Type"] == "application/json"
    body = json.loads(request["body"])
    assert body["model"] == "m" and body["questions"] == Q and body["state"] == {}


def test_choice_and_score_shapes_pass_through(server):
    answers = {
        "c": {"type": "choice", "choice": "a", "confidence": 0.6, "probabilities": {"a": 0.6, "b": 0.4}},
        "s": {"type": "score", "score": 3.2, "legend": {"1": "bad"}, "confidence": 0.5, "probabilities": {"0": 1.0}},
    }
    server.respond(("json", 200, {"model": "m", "answers": answers, "usage": {}}))
    assert client(server, key="k").ask({}, Q) == answers


def test_redirect_is_refused_and_key_not_forwarded(server, other_server):
    server.respond(("redirect", other_server.url))
    c = client(server, key="secret")
    assert c.ask({}, Q) is None
    assert other_server.requests == []
    assert "secret" not in c.last_error


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
        "http_500_then_500_then_500",
    ],
)
def test_failures_return_none_and_never_raise(server, behaviour):
    script = {
        "close_mid_body": [("close_mid_body",)],
        "close_before_response": [("close_before_response",)],
        "not_json": [("raw", b"<html>nope</html>")],
        "json_list": [("json", 200, [1, 2])],
        "answers_not_dict": [("json", 200, {"answers": [1]})],
        "nan_constant": [("raw", b'{"answers": {"q": NaN}}')],
        "hang": [("hang",)],
        "oversized_body": [("raw", b'{"answers": {}, "pad": "' + b"x" * 5000 + b'"}')],
        "http_500_then_500_then_500": [("status", 500)],
    }[behaviour]
    server.respond(*script)
    c = client(server, key="secret", timeout=0.3, retries=1, max_response_bytes=1000)
    t0 = time.monotonic()
    assert c.ask({"a": 1}, Q) is None
    assert time.monotonic() - t0 < 4
    assert c.last_error and "secret" not in c.last_error


def test_500_exhausts_retries(server):
    server.respond(("status", 500))
    assert client(server, key="k").ask({}, Q) is None
    assert len(server.requests) == 3  # 1 + retries


def test_401_is_not_retried(server):
    server.respond(("status", 401))
    c = client(server, key="k")
    assert c.ask({}, Q) is None
    assert len(server.requests) == 1
    assert c.last_error == "HTTP 401"


def test_503_is_retried_then_succeeds(server):
    server.respond(("status", 503), ("json", 200, GOOD))
    assert client(server, key="k").ask({}, Q) == GOOD["answers"]
    assert len(server.requests) == 2


def test_circuit_breaker(server):
    server.respond(("status", 401))
    c = client(server, key="k")
    for _ in range(3):
        assert c.ask({}, Q) is None
    assert len(server.requests) == 3
    assert c.ask({}, Q) is None
    assert len(server.requests) == 3
    assert "circuit" in c.last_error


def test_success_resets_breaker(server):
    c = client(server, key="k", retries=0)
    server.respond(("status", 500))
    c.ask({}, Q)
    c.ask({}, Q)
    server.respond(("json", 200, GOOD))
    assert c.ask({}, Q) is not None
    server.respond(("status", 500))
    c.ask({}, Q)
    c.ask({}, Q)
    assert c.ask({}, Q) is None and len(server.requests) == 3  # the third call still goes out


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
    assert (c.url, c.model, c.api_key_env) == ("https://api.typesafe.ai/v1/systemone", "jev-1.13.0", "TYPESAFE_API_KEY")
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


def test_key_is_redacted_from_errors(server, monkeypatch):
    c = client(server, key="topsecret")
    monkeypatch.setattr(c, "_attempt", lambda *a: (_ for _ in ()).throw(RuntimeError("boom Bearer topsecret topsecret")))
    assert c.ask({}, Q) is None
    assert "topsecret" not in c.last_error and "REDACTED" in c.last_error


def test_request_is_strict_json(server):
    c = client(server, key="k")
    assert c.ask({"loss": float("nan"), "lr": float("inf"), "ok": [1.5]}, Q) is not None

    def refuse(name):
        raise AssertionError(name)

    body = json.loads(server.requests[0]["body"], parse_constant=refuse)
    assert body["state"] == {"loss": None, "lr": None, "ok": [1.5]}


def test_invalid_questions_make_no_request(server):
    c = client(server, key="k")
    assert c.ask({}, {"q": {"type": "bogus", "instructions": "x"}}) is None
    assert c.ask({}, {f"q{i}": Q["q"] for i in range(65)}) is None
    assert server.requests == [] and "invalid request" in c.last_error
    assert c._failures == 0


def test_unserialisable_state_never_raises(server):
    class Weird:
        def __str__(self):
            raise RuntimeError("no")

    assert client(server, key="k").ask({"x": Weird()}, Q) is None


def history_of(n):
    return [{"step": 25 * (i + 1), "accuracy": 0.5 + i / (2 * n), "ece": 0.2 - i / (10 * n), "loss": 1.0 / (i + 1)} for i in range(n)]


def test_digest_respects_budget():
    d = build_digest(history_of(500), step=12500, max_steps=20000, objective=OBJ, scales={}, sentinels={})
    assert len(json.dumps(d, separators=(",", ":")).encode()) <= 8192 and d["history_dropped"] > 0
    assert d["history"][-1]["step"] == 12500  # the newest evaluation is kept
    assert d["history_dropped"] + len(d["history"]) == 500


@pytest.mark.parametrize("budget", [50, 300, 1000, 4096])
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
    assert len(json.dumps(d, separators=(",", ":"), allow_nan=False).encode()) <= budget


def test_digest_content():
    h = history_of(10)
    d = build_digest(h, step=250, max_steps=1000, objective=OBJ, scales={"t": 2.0}, sentinels={"flat": False}, last_n=4)
    assert d["best_step"] == 250 and d["current"]["accuracy"] == h[-1]["accuracy"]
    assert d["slopes"]["accuracy"] > 0 and d["slopes"]["ece"] < 0
    assert d["scales"] == {"t": 2.0} and d["sentinels"] == {"flat": False}
    assert d["history_dropped"] == 0 and d["nonfinite"] == []
    h[-1]["mean_confidence"] = 0.9
    d = build_digest(h, step=250, max_steps=1000, objective=OBJ, scales={}, sentinels={})
    assert d["confidence_gap"] == pytest.approx(0.9 - h[-1]["accuracy"])


def test_digest_control():
    d = build_digest(history_of(5), step=125, max_steps=1000, objective=OBJ, scales={}, sentinels={}, control=history_of(3))
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


def test_ledger_roundtrip_and_truncate(tmp_path):
    ledger = Ledger(tmp_path / "sub" / "ledger.jsonl")
    assert ledger.read() == []
    for step in (10, 20, 30):
        ledger.append("ask", step, {"loss": float("nan"), "step": step})
    records = ledger.read()
    assert [r["step"] for r in records] == [10, 20, 30]
    assert records[0]["kind"] == "ask" and records[0]["payload"] == {"loss": None, "step": 10}
    with open(ledger.path, "a") as f:
        f.write('{"torn": ')  # a torn last write is skipped
    assert len(ledger.read()) == 3
    ledger.truncate_after(20)
    assert [r["step"] for r in ledger.read()] == [10, 20]
    ledger.append("ask", 25, {})
    assert [r["step"] for r in ledger.read()] == [10, 20, 25]
    Ledger(tmp_path / "missing.jsonl").truncate_after(0)


def test_validate_questions_limits():
    validate_questions(Q)
    validate_questions({f"q{i}": Q["q"] for i in range(64)})
    with pytest.raises(ValueError):
        validate_questions({f"q{i}": Q["q"] for i in range(65)})
    with pytest.raises(ValueError):
        validate_questions({"q": {"type": "bogus", "instructions": "x"}})
    many = {"q": {"type": "choice", "instructions": "x", "criteria": {f"o{i}": "d" for i in range(30)}}}
    validate_questions(many)  # the default 26-option cap does not apply
