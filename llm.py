"""Thin async wrapper around the OpenAI API: chat + embeddings, with retries,
a concurrency cap, usage accounting, and an offline fake mode for plumbing tests."""
from __future__ import annotations

import asyncio
import hashlib
import json
import random
from dataclasses import dataclass, field

from openai import AsyncOpenAI

from config import CFG


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0      # includes reasoning tokens, which are billed as output
    reasoning_tokens: int = 0
    calls: int = 0
    embed_calls: int = 0

    def add(self, u) -> None:
        self.calls += 1
        if u is not None:
            self.prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
            self.completion_tokens += getattr(u, "completion_tokens", 0) or 0
            details = getattr(u, "completion_tokens_details", None)
            self.reasoning_tokens += getattr(details, "reasoning_tokens", 0) or 0

    def __str__(self) -> str:
        return (f"{self.calls} chat calls, {self.embed_calls} embed calls, "
                f"{self.prompt_tokens} prompt + {self.completion_tokens} completion tokens "
                f"({self.reasoning_tokens} reasoning)")


def is_reasoning_model(model: str) -> bool:
    """gpt-5 / o-series reject `temperature` and `max_tokens` (verified against the
    API for gpt-5-mini-2025-08-07). The gpt-5 *chat* aliases are not reasoning models."""
    return model.startswith(("gpt-5", "o1", "o3", "o4")) and "chat" not in model


def sampling_params(cfg, model: str) -> dict:
    if is_reasoning_model(model):
        return dict(reasoning_effort=cfg.reasoning_effort,
                    max_completion_tokens=cfg.max_completion_tokens)
    return dict(temperature=cfg.temperature, max_tokens=cfg.max_tokens)


class LLMClient:
    def __init__(self, cfg=CFG):
        self.cfg = cfg
        self.usage = Usage()
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self._client = None
        if not cfg.fake_llm:
            if not cfg.api_key:
                raise RuntimeError("OPENAI_API_KEY is not set (or run with --fake-llm)")
            self._client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    # ------------------------------------------------------------------ chat
    def for_run(self, usage: Usage) -> "RunLLM":
        """A view of this client that also accounts usage to one run. The
        transport (semaphore, retries) stays shared, so concurrent runs still
        respect one process-wide concurrency cap."""
        return RunLLM(self, usage)

    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   usage: Usage | None = None):
        """Returns the raw assistant message object (may carry .tool_calls)."""
        if self.cfg.fake_llm:
            return _FakeMessage(messages, tools)

        kwargs = dict(
            model=self.cfg.model,
            messages=messages,
            **sampling_params(self.cfg, self.cfg.model),
        )
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        async with self._sem:
            resp = await self._with_retry(lambda: self._client.chat.completions.create(**kwargs))
        for u in (self.usage, usage):
            if u is not None:
                u.add(getattr(resp, "usage", None))
        return resp.choices[0].message

    # ------------------------------------------------------------ embeddings
    @property
    def embed_space(self) -> str:
        """Identifies the vector space embeddings live in. Vectors from different
        spaces are not comparable, so state snapshots record and check this."""
        return f"hash-{_HASH_DIM}" if self.cfg.fake_llm else self.cfg.embed_model

    async def embed(self, texts: list[str], usage: Usage | None = None) -> list[list[float]]:
        if self.cfg.fake_llm:
            return [_hash_embed(t) for t in texts]
        try:
            async with self._sem:
                resp = await self._with_retry(
                    lambda: self._client.embeddings.create(model=self.cfg.embed_model, input=texts)
                )
        except Exception as e:
            # Hashed vectors live in a different space from real ones, so a silent
            # fallback would quietly change what retrieval returns mid-experiment.
            if not self.cfg.embed_fallback:
                raise RuntimeError(f"embedding failed and embed_fallback is off: {e}") from e
            print(f"[llm] embedding failed ({e}); falling back to hashed embeddings")
            return [_hash_embed(t) for t in texts]
        for u in (self.usage, usage):
            if u is not None:
                u.embed_calls += 1
        return [d.embedding for d in resp.data]

    # ----------------------------------------------------------------- retry
    async def _with_retry(self, thunk):
        delay = 1.0
        last = None
        for attempt in range(self.cfg.max_retries):
            try:
                return await thunk()
            except Exception as e:
                last = e
                if attempt == self.cfg.max_retries - 1:
                    break
                await asyncio.sleep(delay + random.random() * 0.3)
                delay *= 2
        raise last


class RunLLM:
    """Per-run view of a shared LLMClient: same transport, separate usage."""

    def __init__(self, client: LLMClient, usage: Usage):
        self.client = client
        self.usage = usage

    @property
    def embed_space(self) -> str:
        return self.client.embed_space

    async def chat(self, messages: list[dict], tools: list[dict] | None = None):
        return await self.client.chat(messages, tools, usage=self.usage)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self.client.embed(texts, usage=self.usage)


# --------------------------------------------------------------------- utils
_HASH_DIM = 256


def _hash_embed(text: str, dim: int = _HASH_DIM) -> list[float]:
    """Deterministic bag-of-words hashing embedding. Not semantic, but it keeps
    retrieval functional offline and when the embedding endpoint errors."""
    vec = [0.0] * dim
    for tok in text.lower().split():
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = sum(v * v for v in vec) ** 0.5 or 1.0
    return [v / norm for v in vec]


class _FakeMessage:
    """Stands in for an assistant message when MAS_FAKE_LLM=1.

    Exercises the tool loop once (first turn calls a tool if any are offered),
    then answers in plain text, so the whole pipeline can be smoke-tested for free.
    """
    def __init__(self, messages, tools):
        self.role = "assistant"
        already_used_tool = any(m.get("role") == "tool" for m in messages)
        if tools and not already_used_tool:
            spec = tools[0]["function"]
            args = _fake_args(spec.get("parameters", {}))
            self.content = None
            self.tool_calls = [_FakeToolCall(spec["name"], args)]
        else:
            self.content = "[fake-llm] Synthetic response for offline pipeline testing."
            self.tool_calls = None


class _FakeToolCall:
    def __init__(self, name, args):
        self.id = "call_" + hashlib.md5(f"{name}{args}".encode()).hexdigest()[:8]
        self.type = "function"
        self.function = _FakeFn(name, args)


class _FakeFn:
    def __init__(self, name, args):
        self.name = name
        self.arguments = json.dumps(args)


def _fake_args(schema: dict) -> dict:
    out = {}
    for key, prop in (schema.get("properties") or {}).items():
        if key not in (schema.get("required") or []):
            continue
        t = prop.get("type")
        out[key] = {"string": "test", "integer": 1, "number": 1.0,
                    "boolean": True, "array": [], "object": {}}.get(t, "test")
    return out
