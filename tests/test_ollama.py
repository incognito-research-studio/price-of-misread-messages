"""The Ollama adapter, against a stubbed HTTP layer. No server is contacted."""

from __future__ import annotations

import io
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from cdd.adapters.ollama import OllamaClient, OllamaError, make_client
from cdd.interpreters.typed import TypedQuestion


class _FakeServer:
    def __init__(self, reply: str, models: tuple[str, ...] = ("qwen2.5:7b-instruct",)) -> None:
        self.reply = reply
        self.models = models
        self.bodies: list[dict[str, Any]] = []

    def __call__(self, request: urllib.request.Request, timeout: float) -> io.BytesIO:
        if request.full_url.endswith("/api/tags"):
            models = [{"name": m, "digest": f"sha-{m}"} for m in self.models]
            return io.BytesIO(json.dumps({"models": models}).encode())
        if request.full_url.endswith("/api/version"):
            return io.BytesIO(json.dumps({"version": "0.0.1"}).encode())
        assert request.data is not None
        self.bodies.append(json.loads(request.data))
        return io.BytesIO(json.dumps({"message": {"content": self.reply}}).encode())


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> _FakeServer:
    fake = _FakeServer('{"q1": "b", "q2": "y"}')
    monkeypatch.setattr(urllib.request, "urlopen", fake)
    return fake


def test_complete_sends_seed_temperature_and_schema(server: _FakeServer) -> None:
    client = make_client("qwen2.5:7b-instruct", seed=7)
    client.complete("system", "user", {"type": "object"})
    body = server.bodies[0]
    assert body["options"]["seed"] == 7 and body["options"]["temperature"] == 0.0
    assert body["options"]["num_predict"] == 1024
    assert body["format"] == {"type": "object"} and body["stream"] is False
    assert client.requests == 1


def test_ask_constrains_each_answer_to_its_options(server: _FakeServer) -> None:
    client = OllamaClient("qwen2.5:7b-instruct")
    questions = [
        TypedQuestion("q1", "First?", ("a", "b")),
        TypedQuestion("q2", "Second?", ("x", "y")),
    ]
    answers = client.ask("context", questions)
    assert [(a.key, a.choice, a.probability) for a in answers] == [
        ("q1", "b", 1.0),
        ("q2", "y", 1.0),
    ]
    schema = server.bodies[0]["format"]
    assert schema["properties"]["q1"]["enum"] == ["a", "b"]
    assert schema["required"] == ["q1", "q2"]


def test_cache_reuses_identical_requests(server: _FakeServer) -> None:
    client = OllamaClient("qwen2.5:7b-instruct", cache=True)
    client.complete("s", "u")
    client.complete("s", "u")
    assert client.requests == 1 and len(server.bodies) == 1


def test_missing_model_is_reported(server: _FakeServer) -> None:
    with pytest.raises(OllamaError, match="ollama pull"):
        OllamaClient("not-pulled:1b")


def test_capped_replies_and_prompt_size_are_recorded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def capped(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        if request.full_url.endswith("/api/tags"):
            return io.BytesIO(json.dumps({"models": [{"name": "qwen2.5:7b-instruct"}]}).encode())
        payload = {
            "message": {"content": '{"deltas": [{"type"'},
            "done_reason": "length",
            "prompt_eval_count": 812,
            "eval_count": 1024,
        }
        return io.BytesIO(json.dumps(payload).encode())

    monkeypatch.setattr(urllib.request, "urlopen", capped)
    client = OllamaClient("qwen2.5:7b-instruct", log_path=str(tmp_path / "log.jsonl"))
    client.log_tag = "s/c"
    client.complete("s", "u")
    assert client.capped == 1 and client.max_prompt_tokens == 812
    record = json.loads((tmp_path / "log.jsonl").read_text())
    assert record["tag"] == "s/c" and record["done_reason"] == "length"
    assert record["prompt_tokens"] == 812 and record["reply_tokens"] == 1024


def test_transient_failures_are_retried(
    server: _FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    failures: list[BaseException] = [
        TimeoutError("timed out"),
        urllib.error.HTTPError("http://x/api/chat", 500, "error", {}, io.BytesIO(b"busy")),  # type: ignore[arg-type]
    ]

    def flaky(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        if request.full_url.endswith("/api/chat") and failures:
            raise failures.pop(0)
        return server(request, timeout)

    monkeypatch.setattr(urllib.request, "urlopen", flaky)
    client = OllamaClient("qwen2.5:7b-instruct", retry_wait=0)
    assert client.complete("s", "u") == server.reply
    assert client.retried == 2

    failures.extend([TimeoutError("timed out")] * 3)
    with pytest.raises(OllamaError, match="failed after"):
        client.complete("s", "u")


def test_client_errors_are_not_retried(
    server: _FakeServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    def bad_request(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        if request.full_url.endswith("/api/chat"):
            raise urllib.error.HTTPError(request.full_url, 400, "bad", {}, io.BytesIO(b"no"))  # type: ignore[arg-type]
        return server(request, timeout)

    monkeypatch.setattr(urllib.request, "urlopen", bad_request)
    client = OllamaClient("qwen2.5:7b-instruct", retry_wait=0)
    with pytest.raises(OllamaError, match="HTTP 400"):
        client.complete("s", "u")
    assert client.retried == 0


def test_unreachable_server_is_reported(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(request: urllib.request.Request, timeout: float) -> io.BytesIO:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with pytest.raises(OllamaError, match="ollama serve"):
        OllamaClient("qwen2.5:7b-instruct")


def test_disk_cache_reuses_replies_across_clients(server: _FakeServer, tmp_path: Path) -> None:
    cache = tmp_path / "replies.sqlite"
    first = OllamaClient("qwen2.5:7b-instruct", cache_path=str(cache))
    assert first.complete("s", "u") == server.reply
    second = OllamaClient("qwen2.5:7b-instruct", cache_path=str(cache))
    second.log_path = tmp_path / "log.jsonl"
    assert second.complete("s", "u") == server.reply
    assert (len(server.bodies), second.requests, second.cache_hits) == (1, 0, 1)
    logged = json.loads((tmp_path / "log.jsonl").read_text())
    assert logged["cached"] is True and logged["reply"] == server.reply
    # Another seed is another request.
    OllamaClient("qwen2.5:7b-instruct", seed=1, cache_path=str(cache)).complete("s", "u")
    assert len(server.bodies) == 2


def test_provenance_records_the_digest_and_server(server: _FakeServer) -> None:
    info = OllamaClient("qwen2.5:7b-instruct").provenance()
    assert (info["digest"], info["server_version"]) == ("sha-qwen2.5:7b-instruct", "0.0.1")
