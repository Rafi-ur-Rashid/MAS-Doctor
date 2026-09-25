"""Record-replay store for LLM responses, and the budget guard.

Why: gpt-5-mini cannot be made to repeat itself (the API rejects `temperature`),
so two runs of the same experiment would differ by chance. With every response
stored under a fingerprint of the exact request, a run can be replayed exactly,
and a paired run (clean vs. attacked) shares every call up to the first one whose
request the intervention changed -- the divergence point.

Modes (per run):
  replay-or-record  stored response if there is one, else call the API and store it
  replay-strict     stored response or CacheMiss; never calls the API
  off               no cache (demos only; the run cannot be replayed)

There is deliberately no "record-only" mode: re-recording an existing request
would either overwrite the response earlier runs used (breaking their replay) or
behave exactly like replay-or-record. Fresh samples of the same request come
from a different replica index instead.

The stored response is the one every run uses. If two processes miss on the same
request at once, both call the API, the first write wins, and the loser discards
its own response and continues with the stored one -- so the run and its later
replay always agree.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

MODES = ("replay-or-record", "replay-strict", "off")
KEY_VERSION = 1


class CacheMiss(RuntimeError):
    """replay-strict found no stored response for a request."""


class BudgetExceeded(RuntimeError):
    """The process has reached its spending cap; no further API calls are made."""


# ------------------------------------------------------------------ fingerprints
def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def chat_key(request: dict, replica: int) -> str:
    """request is the exact kwargs sent to chat.completions.create: model snapshot,
    messages, tools, tool_choice and sampling parameters."""
    return hashlib.sha256(canonical(
        {"v": KEY_VERSION, "kind": "chat", "request": request, "replica": replica}).encode()).hexdigest()


def embed_key(model: str, text: str) -> str:
    """Embeddings are not sampled, so they are never replicated."""
    return hashlib.sha256(canonical(
        {"v": KEY_VERSION, "kind": "embed", "model": model, "input": text}).encode()).hexdigest()


# ------------------------------------------------------------------------ store
class ResponseCache:
    """SQLite, safe for several processes: WAL journal, busy timeout, and
    first-write-wins inserts. Payloads are zlib-compressed JSON, kept exact."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=60, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("""CREATE TABLE IF NOT EXISTS responses (
                               key      TEXT PRIMARY KEY,
                               kind     TEXT NOT NULL,
                               request  BLOB NOT NULL,
                               response BLOB NOT NULL,
                               created  REAL NOT NULL)""")
        self.db.commit()

    @staticmethod
    def _pack(obj) -> bytes:
        return zlib.compress(canonical(obj).encode())

    @staticmethod
    def _unpack(blob: bytes):
        return json.loads(zlib.decompress(blob))

    def get(self, key: str) -> dict | None:
        row = self.db.execute("SELECT response FROM responses WHERE key = ?", (key,)).fetchone()
        return self._unpack(row[0]) if row else None

    def put(self, key: str, kind: str, request, response) -> dict:
        """Store unless already present; return whatever is stored now."""
        self.db.execute("INSERT OR IGNORE INTO responses VALUES (?, ?, ?, ?, ?)",
                        (key, kind, self._pack(request), self._pack(response), time.time()))
        self.db.commit()
        return self.get(key)

    def request(self, key: str):
        """The stored request, for audits and debugging."""
        row = self.db.execute("SELECT request FROM responses WHERE key = ?", (key,)).fetchone()
        return self._unpack(row[0]) if row else None

    def __len__(self) -> int:
        return self.db.execute("SELECT COUNT(*) FROM responses").fetchone()[0]


# ------------------------------------------------------------------ run stats
@dataclass
class CacheStats:
    """Per run. first_miss_call is the index of the first chat call that was not
    served from the store: for a paired run, the divergence point."""
    mode: str = "replay-or-record"
    replica: int = 0
    replica_from_call: int = 0
    chat_hits: int = 0
    chat_misses: int = 0
    embed_hits: int = 0
    embed_misses: int = 0
    first_miss_call: int | None = None
    truncated: int = 0                     # responses cut off by max_completion_tokens
    missed_calls: list[int] = field(default_factory=list)

    def chat_hit(self) -> None:
        self.chat_hits += 1

    def chat_miss(self, idx: int) -> None:
        self.chat_misses += 1
        self.missed_calls.append(idx)
        if self.first_miss_call is None or idx < self.first_miss_call:
            self.first_miss_call = idx


# ---------------------------------------------------------------------- prices
def price_for(prices: dict, model: str) -> dict:
    if model not in prices:
        raise KeyError(f"no price for model {model!r}; add it to config.PRICES_PER_1M "
                       f"before making API calls, so the budget cap can work")
    return prices[model]


def chat_cost_upper(usage, price: dict) -> float:
    """Upper bound in USD: every prompt token at the full input price (ignores the
    cached-input discount), every completion token, reasoning included, at output."""
    if usage is None:
        return 0.0
    return ((usage.prompt_tokens or 0) * price["input"]
            + (usage.completion_tokens or 0) * price["output"]) / 1e6


def embed_cost_upper(usage, price: dict) -> float:
    if usage is None:
        return 0.0
    return (usage.prompt_tokens or 0) * price["input"] / 1e6
