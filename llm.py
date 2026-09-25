"""Thin async wrapper around the OpenAI API: chat + embeddings, with retries,
a concurrency cap, record-replay caching (cache.py), a spending cap, usage
accounting, and an offline fake mode for plumbing tests."""
from __future__ import annotations

import asyncio
import hashlib
import itertools
import json
import random
from dataclasses import dataclass

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

from cache import (MODES, BudgetExceeded, CacheMiss, CacheStats, ResponseCache,
                   chat_cost_upper, chat_key, embed_cost_upper, embed_key, price_for)
from config import CFG, PRICES_PER_1M


@dataclass
class Usage:
    """Real API spend only. Responses served from the cache cost nothing and are
    not counted here (CacheStats counts them)."""
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0   # served from OpenAI's prompt cache (billed at a discount)
    completion_tokens: int = 0      # includes reasoning tokens, which are billed as output
    reasoning_tokens: int = 0
    calls: int = 0
    embed_calls: int = 0
    embed_tokens: int = 0
    cost_usd_upper: float = 0.0     # upper bound: see cache.chat_cost_upper

    def add_chat(self, u, cost: float) -> None:
        self.calls += 1
        self.cost_usd_upper += cost
        if u is not None:
            self.prompt_tokens += u.prompt_tokens or 0
            self.completion_tokens += u.completion_tokens or 0
            self.reasoning_tokens += getattr(u.completion_tokens_details, "reasoning_tokens", 0) or 0
            self.cached_prompt_tokens += getattr(u.prompt_tokens_details, "cached_tokens", 0) or 0

    def add_embed(self, u, cost: float) -> None:
        self.embed_calls += 1
        self.cost_usd_upper += cost
        if u is not None:
            self.embed_tokens += u.prompt_tokens or 0

    def __str__(self) -> str:
        return (f"{self.calls} chat calls, {self.embed_calls} embed calls, "
                f"{self.prompt_tokens} prompt + {self.completion_tokens} completion tokens "
                f"({self.reasoning_tokens} reasoning), <= ${self.cost_usd_upper:.4f}")


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
    """Shared by every run in a process: one connection, one concurrency cap, one
    budget. Runs talk to it through RunLLM views (for_run), which carry their own
    usage, cache mode, replica and call counter."""

    def __init__(self, cfg=CFG, cache: ResponseCache | None = None):
        self.cfg = cfg
        self.usage = Usage()            # whole process: what the budget cap checks
        self._sem = asyncio.Semaphore(cfg.max_concurrency)
        self._cache = cache
        self._client = None
        if not cfg.fake_llm:
            # fail now, not mid-run, if the budget cap could not price a call
            self._chat_price = price_for(PRICES_PER_1M, cfg.model)
            self._embed_price = price_for(PRICES_PER_1M, cfg.embed_model)
            if cfg.api_key:             # replay-strict needs no key
                self._client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)

    @property
    def cache(self) -> ResponseCache:
        if self._cache is None:
            self._cache = ResponseCache(self.cfg.cache_path)
        return self._cache

    def for_run(self, usage: Usage, stats: CacheStats | None = None, mode: str | None = None,
                replica: int = 0, replica_from_call: int = 0) -> "RunLLM":
        """A per-run view: shares the transport, cache and budget, but accounts usage
        and cache stats to one run and numbers that run's chat calls."""
        return RunLLM(self, usage, stats or CacheStats(), mode or self.cfg.cache_mode,
                      replica, replica_from_call)

    # ------------------------------------------------------------------ chat
    async def chat(self, messages: list[dict], tools: list[dict] | None = None,
                   usage: Usage | None = None, *, idx: int = 0, replica: int = 0,
                   mode: str | None = None, stats: CacheStats | None = None):
        """Returns the assistant message object (may carry .tool_calls)."""
        if self.cfg.fake_llm:
            return _FakeMessage(messages, tools)
        mode = mode or self.cfg.cache_mode
        if mode not in MODES:
            raise ValueError(f"unknown cache mode {mode!r}; expected one of {MODES}")
        stats = stats if stats is not None else CacheStats(mode=mode)

        request = dict(model=self.cfg.model, messages=messages,
                       **sampling_params(self.cfg, self.cfg.model))
        if tools:
            request["tools"] = tools
            request["tool_choice"] = "auto"

        if mode == "off":
            stats.chat_miss(idx)
            data = await self._call_chat(request, usage)
        else:
            key = chat_key(request, replica)
            data = self.cache.get(key)
            if data is not None:
                stats.chat_hit()
            else:
                stats.chat_miss(idx)
                if mode == "replay-strict":
                    raise CacheMiss(f"chat call {idx}: no stored response for key {key[:16]}... "
                                    f"(replica {replica})")
                fresh = await self._call_chat(request, usage)
                # first write wins: if another process stored this request meanwhile,
                # continue with its response so this run and its replay agree
                data = self.cache.put(key, "chat", request, fresh)

        # rebuilt from stored JSON on every path, so a recorded run and its replay
        # hand the agent objects built the same way
        choice = ChatCompletion.model_validate(data).choices[0]
        if choice.finish_reason == "length":
            stats.truncated += 1
        return choice.message

    async def _call_chat(self, request: dict, usage: Usage | None) -> dict:
        client = self._require_client()
        async with self._sem:
            self._check_budget()
            resp = await self._with_retry(lambda: client.chat.completions.create(**request))
        cost = chat_cost_upper(resp.usage, self._chat_price)
        for u in (self.usage, usage):
            if u is not None:
                u.add_chat(resp.usage, cost)
        return resp.model_dump(mode="json")

    # ------------------------------------------------------------ embeddings
    @property
    def embed_space(self) -> str:
        """Identifies the vector space embeddings live in. Vectors from different
        spaces are not comparable, so state snapshots record and check this."""
        return f"hash-{_HASH_DIM}" if self.cfg.fake_llm else self.cfg.embed_model

    async def embed(self, texts: list[str], usage: Usage | None = None, *,
                    mode: str | None = None, stats: CacheStats | None = None) -> list[list[float]]:
        if self.cfg.fake_llm:
            return [_hash_embed(t) for t in texts]
        mode = mode or self.cfg.cache_mode
        if mode not in MODES:
            raise ValueError(f"unknown cache mode {mode!r}; expected one of {MODES}")
        stats = stats if stats is not None else CacheStats(mode=mode)
        model = self.cfg.embed_model

        if mode == "off":
            stats.embed_misses += len(texts)
            vecs, _ = await self._call_embed(texts, usage)
            return vecs

        keys = [embed_key(model, t) for t in texts]
        out: list = [self.cache.get(k) for k in keys]
        missing = [i for i, d in enumerate(out) if d is None]
        stats.embed_hits += len(texts) - len(missing)
        stats.embed_misses += len(missing)
        if missing:
            if mode == "replay-strict":
                raise CacheMiss(f"embedding: no stored vector for {len(missing)} text(s)")
            vecs, fallback = await self._call_embed([texts[i] for i in missing], usage)
            for i, v in zip(missing, vecs):
                # a hashed fallback vector must never enter the store
                out[i] = {"embedding": v} if fallback else self.cache.put(
                    keys[i], "embed", {"model": model, "input": texts[i]}, {"embedding": v})
        return [d["embedding"] for d in out]

    async def _call_embed(self, texts: list[str], usage: Usage | None):
        """Returns (vectors, used_fallback)."""
        try:
            client = self._require_client()
            async with self._sem:
                self._check_budget()
                resp = await self._with_retry(
                    lambda: client.embeddings.create(model=self.cfg.embed_model, input=texts))
        except BudgetExceeded:
            raise
        except Exception as e:
            # Hashed vectors live in a different space from real ones, so a silent
            # fallback would quietly change what retrieval returns mid-experiment.
            if not self.cfg.embed_fallback:
                raise RuntimeError(f"embedding failed and embed_fallback is off: {e}") from e
            print(f"[llm] embedding failed ({e}); falling back to hashed embeddings")
            return [_hash_embed(t) for t in texts], True
        cost = embed_cost_upper(resp.usage, self._embed_price)
        for u in (self.usage, usage):
            if u is not None:
                u.add_embed(resp.usage, cost)
        return [d.embedding for d in resp.data], False

    # ------------------------------------------------------------- guards
    def _require_client(self):
        if self._client is None:
            raise RuntimeError("OPENAI_API_KEY is not set: only replay-strict runs (or "
                               "--fake-llm) work without it")
        return self._client

    def _check_budget(self) -> None:
        """Checked inside the concurrency slot, right before each API call. Calls
        already in flight still finish, so spend can pass the cap by at most
        max_concurrency calls."""
        if self.usage.cost_usd_upper >= self.cfg.budget_usd:
            raise BudgetExceeded(f"spent <= ${self.usage.cost_usd_upper:.4f} (upper bound), "
                                 f"cap ${self.cfg.budget_usd:.2f}; no further API calls")

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
    """Per-run view of a shared LLMClient: same transport, cache and budget;
    separate usage, cache stats, cache mode and replica."""

    def __init__(self, client: LLMClient, usage: Usage, stats: CacheStats, mode: str,
                 replica: int, replica_from_call: int):
        if mode not in MODES:
            raise ValueError(f"unknown cache mode {mode!r}; expected one of {MODES}")
        if replica < 0 or replica_from_call < 0:
            raise ValueError("replica and replica_from_call must be >= 0")
        self.client, self.usage, self.stats = client, usage, stats
        self.mode, self.replica, self.replica_from_call = mode, replica, replica_from_call
        stats.mode, stats.replica, stats.replica_from_call = mode, replica, replica_from_call
        self._calls = itertools.count()

    @property
    def embed_space(self) -> str:
        return self.client.embed_space

    async def chat(self, messages: list[dict], tools: list[dict] | None = None):
        # Numbered before the first await, so concurrent agents started in index
        # order (asyncio.gather) always get the same numbers.
        idx = next(self._calls)
        replica = self.replica if idx >= self.replica_from_call else 0
        return await self.client.chat(messages, tools, usage=self.usage, idx=idx,
                                      replica=replica, mode=self.mode, stats=self.stats)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return await self.client.embed(texts, usage=self.usage, mode=self.mode, stats=self.stats)


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
