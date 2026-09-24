"""Memory for the MAS, in three real layers.

1. Working memory  - the rolling chat transcript an agent sends to the model.
2. Episodic memory - per-agent vector store; embedded, retrieved by cosine
                     similarity and injected into the prompt each turn.
3. Blackboard      - a shared key/value store every agent can read and write
                     through tools. This is the MAS's common workspace.

None of these objects is global. Each run owns its own instances through its
RunContext (run_context.py), and they start from a named state snapshot, so
nothing carries over from one run to the next unless a run asks for it.
Nothing here records wall-clock time: ordering uses the run's logical clock.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from config import CFG


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
    seq: int = 0                       # logical time: the run clock when it was written
    vec: list[float] = field(default_factory=list)


class VectorStore:
    """Small embedded store. Linear scan is fine at MAS transcript scale."""

    def __init__(self, llm, clock, items: list[MemoryItem] | None = None):
        self.llm = llm
        self.clock = clock
        self.items: list[MemoryItem] = list(items or [])

    async def add(self, text: str, meta: dict | None = None) -> None:
        if not text.strip():
            return
        vec = (await self.llm.embed([text]))[0]
        self.items.append(MemoryItem(text=text, meta=meta or {}, seq=self.clock(), vec=vec))

    async def search(self, query: str, k: int = 3,
                     min_similarity: float = 0.0) -> list[tuple[float, MemoryItem]]:
        """Top-k by cosine, dropping anything below min_similarity. Returns
        (score, item) pairs so callers can see how relevant a hit actually was."""
        if not self.items:
            return []
        qvec = (await self.llm.embed([query]))[0]
        scored = [(_cosine(qvec, it.vec), it) for it in self.items]
        scored = [p for p in scored if p[0] >= min_similarity]
        scored.sort(key=lambda p: -p[0])       # stable: ties keep insertion order
        return scored[:k]


# ============================================================== agent memory
class AgentMemory:
    def __init__(self, system_prompt: str, episodic: VectorStore):
        self.working: list[dict] = [{"role": "system", "content": system_prompt}]
        self.episodic = episodic

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
    """Shared state across the agents of one run."""

    def __init__(self, clock, data: dict | None = None, log: list | None = None):
        self.clock = clock
        self.data: dict[str, str] = dict(data or {})
        self.log: list[dict] = list(log or [])

    def write(self, key: str, value: str, author: str, round_idx: int) -> dict:
        self.data[key] = value
        self.log.append({"seq": self.clock(), "round": round_idx,
                         "key": key, "author": author, "value": value})
        return {"status": "written", "key": key}

    def read(self, key: str | None = None) -> dict:
        if key is None:
            return {"keys": sorted(self.data)}
        if key not in self.data:
            return {"error": f"no entry '{key}'", "keys": sorted(self.data)}
        return {"key": key, "value": self.data[key]}
