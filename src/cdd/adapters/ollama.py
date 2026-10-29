"""Local models served by Ollama, through its HTTP API and the standard library only.

One client serves both model interfaces: ``complete`` for the text conditions and ``ask`` for
the typed-question conditions. Typed answers come from schema-constrained decoding: every
question becomes a JSON property whose value must be one of its options. A general instruction
model used this way imitates the *output* of a typed decision model, not its calibration, so
every answer reports probability 1.0.

Start the server with ``ollama serve`` (or the Ollama app) before running an experiment.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from cdd.adapters.cache import ReplyCache, request_key
from cdd.interpreters.typed import TypedAnswer, TypedQuestion

DEFAULT_HOST = "http://127.0.0.1:11434"

TYPED_SYSTEM_PROMPT = """\
You answer multiple-choice questions about operational messages and planning data.
For every question, choose exactly one of its options and copy it exactly."""


class OllamaError(RuntimeError):
    """The server is unreachable, the model is missing, or a request failed."""


class OllamaClient:
    """Chat client for one Ollama model.

    ``temperature`` and ``seed`` are sent with every request. ``max_tokens`` caps each reply, so
    a model stuck repeating itself returns truncated (unparseable) output instead of running into
    ``timeout`` and stopping the experiment. A request that times out, loses its connection or
    gets a server error (HTTP 5xx) is retried ``retries`` times, waiting ``retry_wait`` seconds
    and then twice that; the final failure raises :class:`OllamaError`. With ``cache`` on, identical
    requests within a process return the first reply instead of calling the model again;
    ``cache_path`` keeps replies on disk across runs (:mod:`cdd.adapters.cache`), keyed by the
    request and the model digest. Keep both off for repeat runs that measure decision
    consistency. ``log_path`` appends every request and reply as one JSON line, labelled with
    ``log_tag``; a reply served from the disk cache is logged with ``"cached": true``. The
    experiment runner sets ``log_path``, ``log_tag`` and ``cache_path``.
    """

    def __init__(
        self,
        model: str,
        *,
        host: str = DEFAULT_HOST,
        temperature: float = 0.0,
        seed: int = 0,
        num_ctx: int = 8192,
        max_tokens: int = 1024,
        timeout: float = 300.0,
        retries: int = 2,
        retry_wait: float = 5.0,
        cache: bool = False,
        cache_path: str | None = None,
        log_path: str | None = None,
        check: bool = True,
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.temperature = temperature
        self.seed = seed
        self.num_ctx = num_ctx
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.retries = retries
        self.retry_wait = retry_wait
        self.cache = cache
        self.cache_path = Path(cache_path) if cache_path else None
        self.log_path = Path(log_path) if log_path else None
        self.log_tag: str | None = None
        self.requests = 0
        self.seconds = 0.0
        self.retried = 0
        self.capped = 0
        self.cache_hits = 0
        self.max_prompt_tokens = 0
        self.digest: str | None = None
        self.server_version: str | None = None
        self._replies: dict[str, str] = {}
        self._store: ReplyCache | None = None
        if check:
            self._check_model()

    @property
    def name(self) -> str:
        return f"ollama:{self.model}"

    def provenance(self) -> dict[str, Any]:
        """What produced the replies: model name and digest, and the server version."""
        if self.digest is None:
            self._check_model()
        return {
            "client": "ollama",
            "model": self.model,
            "digest": self.digest,
            "server_version": self.server_version,
            "temperature": self.temperature,
            "seed": self.seed,
            "num_ctx": self.num_ctx,
            "max_tokens": self.max_tokens,
        }

    def complete(self, system: str, user: str, schema: dict[str, Any] | None = None) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return self._chat(messages, schema)

    def ask(self, context: str, questions: Sequence[TypedQuestion]) -> list[TypedAnswer]:
        schema = {
            "type": "object",
            "properties": {q.key: {"type": "string", "enum": list(q.options)} for q in questions},
            "required": [q.key for q in questions],
        }
        lines = [context, "", "Answer every question below. Reply with JSON keyed by question id."]
        for question in questions:
            lines.append(f"\nQuestion {json.dumps(question.key)}: {question.prompt}")
            if question.definitions:
                lines.append("Options (answer with the label only):")
                lines.extend(
                    f"- {json.dumps(option)}: {definition}"
                    for option, definition in zip(
                        question.options, question.definitions, strict=True
                    )
                )
            else:
                lines.append(f"Options: {json.dumps(list(question.options))}")
        reply = self._chat(
            [
                {"role": "system", "content": TYPED_SYSTEM_PROMPT},
                {"role": "user", "content": "\n".join(lines)},
            ],
            schema,
        )
        answers = json.loads(reply)
        if not isinstance(answers, dict):
            raise ValueError("typed reply is not a JSON object")
        return [TypedAnswer(q.key, str(answers.get(q.key, "")), 1.0) for q in questions]

    def _chat(self, messages: list[dict[str, str]], schema: dict[str, Any] | None) -> str:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "seed": self.seed,
                "num_ctx": self.num_ctx,
                "num_predict": self.max_tokens,
            },
        }
        if schema is not None:
            body["format"] = schema
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if self.cache and key in self._replies:
            return self._replies[key]
        store = self._reply_store()
        stored_key = request_key(body, self.digest) if store is not None else ""
        stored = store.get(stored_key) if store is not None else None
        if stored is not None:
            self.cache_hits += 1
            self._log(body, stored, cached=True)
            return str(stored["reply"])

        started = time.perf_counter()
        payload = self._post("/api/chat", body)
        elapsed = time.perf_counter() - started
        reply = str(payload["message"]["content"])
        self.requests += 1
        self.seconds += elapsed
        # Ollama silently drops the start of a prompt longer than num_ctx; the token count
        # shows how close a prompt came. A reply cut by max_tokens ends with "length".
        prompt_tokens = int(payload.get("prompt_eval_count") or 0)
        self.max_prompt_tokens = max(self.max_prompt_tokens, prompt_tokens)
        self.capped += payload.get("done_reason") == "length"
        if self.cache:
            self._replies[key] = reply
        record = {
            "reply": reply,
            "seconds": elapsed,
            "done_reason": payload.get("done_reason"),
            "prompt_tokens": prompt_tokens,
            "reply_tokens": payload.get("eval_count"),
        }
        if store is not None:
            store.put(stored_key, record)
        self._log(body, record, cached=False)
        return reply

    def _reply_store(self) -> ReplyCache | None:
        if self.cache_path is None:
            return None
        if self._store is None or self._store.path != self.cache_path:
            if self.digest is None:
                self._check_model()
            self._store = ReplyCache(self.cache_path)
        return self._store

    def _log(self, body: dict[str, Any], record: dict[str, Any], *, cached: bool) -> None:
        if self.log_path is None:
            return
        entry = {"tag": self.log_tag, "model": self.model, "request": body, **record}
        if cached:
            entry["cached"] = True
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def _post(self, path: str, body: dict[str, Any] | None = None, *, retry: bool = True) -> Any:
        data = None if body is None else json.dumps(body).encode()
        attempts = 1 + (self.retries if retry else 0)
        for attempt in range(attempts):
            request = urllib.request.Request(
                f"{self.host}{path}",
                data=data,
                headers={"Content-Type": "application/json"},
                method="GET" if body is None else "POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.load(response)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode(errors="replace")
                error = OllamaError(f"{path} failed with HTTP {exc.code}: {detail}")
                if exc.code < 500:
                    raise error from exc
                cause: Exception = exc
            except urllib.error.URLError as exc:
                error = OllamaError(
                    f"cannot reach Ollama at {self.host} ({exc.reason}); "
                    "start it with `ollama serve`"
                )
                cause = exc
            except (TimeoutError, ConnectionError) as exc:
                error = OllamaError(
                    f"{path} failed after {self.timeout:g} s or lost its connection"
                )
                cause = exc
            if attempt + 1 < attempts:
                self.retried += 1
                time.sleep(self.retry_wait * (attempt + 1))
        raise error from cause

    def _check_model(self) -> None:
        tags = self._post("/api/tags", retry=False)
        models = {model["name"]: model for model in tags.get("models", [])}
        found = models.get(self.model) or models.get(f"{self.model}:latest")
        if found is None:
            available = ", ".join(sorted(models)) or "none"
            raise OllamaError(
                f"model {self.model!r} is not pulled (available: {available}); "
                f"run `ollama pull {self.model}`"
            )
        self.digest = found.get("digest")
        try:
            self.server_version = self._post("/api/version", retry=False).get("version")
        except OllamaError:
            self.server_version = None


def make_client(model: str, **options: Any) -> OllamaClient:
    """Factory for experiment configs: ``client: cdd.adapters.ollama:make_client``."""
    return OllamaClient(model, **options)
