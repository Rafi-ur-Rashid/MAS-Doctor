"""Memory for the MAS, in three real layers.

1. Working memory  - the rolling chat transcript an agent sends to the model.
2. Episodic memory - per-agent vector store; embedded, persisted, retrieved by
                     cosine similarity and injected into the prompt each turn.
3. Blackboard      - a shared key/value store every agent can read and write
                     through tools. This is the MAS's common workspace.
"""
from __future__ import annotations

import asyncio
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field, asdict
from pathlib import Path

from config import CFG
from tools import REGISTRY


# ============================================================== vector store
def _cosine(a: list[float], b: list[float]) -> float:
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    na = sum(x * x for x in a[:n]) ** 0.5 or 1.0
    nb = sum(x * x for x in b[:n]) ** 0.5 or 1.0
    return dot / (na * nb)


@dataclass
class MemoryItem:
    text: str
    meta: dict = field(default_factory=dict)
    ts: float = field(default_factory=time.time)
    vec: list[float] = field(default_factory=list)


class VectorStore:
    """Small embedded store. Linear scan is fine at MAS transcript scale."""

    def __init__(self, llm, path: Path | None = None):
        self.llm = llm
        self.path = path
        self.items: list[MemoryItem] = []
        if path and path.exists():
            self.items = [MemoryItem(**d) for d in json.loads(path.read_text())]

    async def add(self, text: str, meta: dict | None = None) -> None:
        if not text.strip():
            return
        vec = (await self.llm.embed([text]))[0]
        self.items.append(MemoryItem(text=text, meta=meta or {}, vec=vec))
        self._flush()

    async def search(self, query: str, k: int = 3,
                     min_similarity: float = 0.0) -> list[tuple[float, MemoryItem]]:
        """Top-k by cosine, dropping anything below min_similarity. Returns
        (score, item) pairs so callers can see how relevant a hit actually was."""
        if not self.items:
            return []
        qvec = (await self.llm.embed([query]))[0]
        scored = [(_cosine(qvec, it.vec), it) for it in self.items]
        scored = [p for p in scored if p[0] >= min_similarity]
        scored.sort(key=lambda p: -p[0])
        return scored[:k]

    def _flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps([asdict(i) for i in self.items]))


# ============================================================== agent memory
class AgentMemory:
    def __init__(self, agent_id: int, system_prompt: str, llm, persist: bool = True):
        self.agent_id = agent_id
        self.working: list[dict] = [{"role": "system", "content": system_prompt}]
        path = (CFG.state_dir / f"agent_{agent_id}_episodic.json") if persist else None
        self.episodic = VectorStore(llm, path)

    # -- working ---------------------------------------------------------
    def append(self, message: dict) -> None:
        self.working.append(message)

    def prompt_messages(self) -> list[dict]:
        """System prompt + the most recent turns, so context stays bounded."""
        system, rest = self.working[0], self.working[1:]
        keep = CFG.working_memory_turns * 2
        if len(rest) <= keep:
            return [system] + rest
        window = rest[-keep:]
        # never start a window on an orphaned tool result
        while window and window[0].get("role") == "tool":
            window = window[1:]
        return [system] + window

    # -- episodic --------------------------------------------------------
    async def remember(self, text: str, meta: dict | None = None) -> None:
        await self.episodic.add(text, meta)

    async def recall(self, query: str, k: int | None = None) -> str:
        hits = await self.episodic.search(query, k or CFG.recall_k,
                                          min_similarity=CFG.recall_min_similarity)
        if not hits:
            return ""   # nothing relevant beats injecting noise the agent will trust
        lines = [f"- ({it.meta.get('kind', 'note')}, round {it.meta.get('round', '?')}, "
                 f"relevance {score:.2f}) {it.text}" for score, it in hits]
        return "Relevant things you recall from earlier work:\n" + "\n".join(lines)


# ================================================================ blackboard
class Blackboard:
    """Shared state across agents. Singleton so the tools can reach it."""
    _instance: "Blackboard | None" = None

    def __init__(self, persist: bool = True):
        self.path = (CFG.state_dir / "blackboard.json") if persist else None
        self.data: dict[str, str] = {}
        self.log: list[dict] = []
        if self.path and self.path.exists():
            saved = json.loads(self.path.read_text())
            self.data, self.log = saved.get("data", {}), saved.get("log", [])
        self._lock = asyncio.Lock()
        Blackboard._instance = self

    @classmethod
    def get(cls) -> "Blackboard":
        return cls._instance or cls()

    def write(self, key: str, value: str, author: str = "unknown") -> dict:
        self.data[key] = value
        self.log.append({"ts": time.time(), "key": key, "author": author, "value": value})
        self._flush()
        return {"status": "written", "key": key}

    def read(self, key: str | None = None) -> dict:
        if key is None:
            return {"keys": sorted(self.data)}
        if key not in self.data:
            return {"error": f"no entry '{key}'", "keys": sorted(self.data)}
        return {"key": key, "value": self.data[key]}

    def reset(self) -> None:
        self.data, self.log = {}, []
        self._flush()

    def _flush(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"data": self.data, "log": self.log}, indent=2))


# Set by Agent.act so blackboard writes are attributable. A ContextVar (not a plain
# dict) because agents run concurrently -- each asyncio task gets its own copy.
CURRENT_AUTHOR: ContextVar[str] = ContextVar("current_author", default="unknown")


@REGISTRY.register(
    "blackboard_write", "Publish a finding to the shared blackboard so other agents can read it.",
    {"type": "object",
     "properties": {"key": {"type": "string", "description": "Short identifier, e.g. 'q2_revenue'."},
                    "value": {"type": "string", "description": "The content to share."}},
     "required": ["key", "value"]})
def blackboard_write(key: str, value: str):
    return Blackboard.get().write(key, value, author=CURRENT_AUTHOR.get())


@REGISTRY.register(
    "blackboard_read", "Read a shared blackboard entry, or omit 'key' to list available keys.",
    {"type": "object", "properties": {"key": {"type": "string"}}, "required": []})
def blackboard_read(key: str | None = None):
    return Blackboard.get().read(key)
