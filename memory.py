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
from events import Composite


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
    write_id: str = ""                 # "w<seq>": the artifact id in the event log


class VectorStore:
    """Small embedded store. Linear scan is fine at MAS transcript scale."""

    def __init__(self, llm, clock, items: list[MemoryItem] | None = None):
        self.llm = llm
        self.clock = clock
        self.items: list[MemoryItem] = list(items or [])

    async def add(self, text: str, meta: dict | None = None) -> MemoryItem | None:
        if not text.strip():
            return None
        vec = (await self.llm.embed([text]))[0]
        seq = self.clock()
        item = MemoryItem(text=text, meta=meta or {}, seq=seq, vec=vec, write_id=f"w{seq}")
        self.items.append(item)
        return item

    async def score(self, query: str) -> list[tuple[float, MemoryItem]]:
        """Every item with its cosine similarity to the query, best first."""
        if not self.items:
            return []
        qvec = (await self.llm.embed([query]))[0]
        scored = [(_cosine(qvec, it.vec), it) for it in self.items]
        scored.sort(key=lambda p: -p[0])       # stable: ties keep insertion order
        return scored

    async def search(self, query: str, k: int = 3,
                     min_similarity: float = 0.0) -> list[tuple[float, MemoryItem]]:
        """Top-k by cosine, dropping anything below min_similarity. Returns
        (score, item) pairs so callers can see how relevant a hit actually was."""
        return [p for p in await self.score(query) if p[0] >= min_similarity][:k]


# ============================================================== agent memory
class AgentMemory:
    """working holds the chat messages; working_ids holds each one's artifact id
    in the event log, index for index."""

    def __init__(self, system_prompt: str, episodic: VectorStore, system_id: str):
        self.working: list[dict] = [{"role": "system", "content": system_prompt}]
        self.working_ids: list[str] = [system_id]
        self.episodic = episodic

    # -- working ---------------------------------------------------------
    def append(self, message: dict, aid: str) -> None:
        self.working.append(message)
        self.working_ids.append(aid)

    def window(self) -> tuple[list[dict], list[str]]:
        """System prompt + the most recent turns, so context stays bounded.
        Returns the messages and their artifact ids."""
        system, rest = 0, list(range(1, len(self.working)))
        keep = CFG.working_memory_turns * 2
        if len(rest) > keep:
            rest = rest[-keep:]
            # never start a window on an orphaned tool result
            while rest and self.working[rest[0]].get("role") == "tool":
                rest = rest[1:]
        idx = [system] + rest
        return [self.working[i] for i in idx], [self.working_ids[i] for i in idx]

    def prompt_messages(self) -> list[dict]:
        return self.window()[0]

    # -- episodic --------------------------------------------------------
    async def remember(self, text: str, meta: dict | None = None) -> MemoryItem | None:
        return await self.episodic.add(text, meta)

    async def recall(self, query: str, k: int | None = None) -> tuple[Composite | None, dict]:
        """Returns the recall block (None if nothing relevant) and what the
        retrieval saw, for the mem_read event."""
        k = k or CFG.recall_k
        scored = await self.episodic.score(query)
        hits = [p for p in scored if p[0] >= CFG.recall_min_similarity][:k]
        read = {"k": k, "threshold": CFG.recall_min_similarity,
                "n_candidates": len(scored),
                "top": [{"write_id": it.write_id, "similarity": sc} for sc, it in scored[:k + 1]],
                "returned": [it.write_id for _, it in hits],
                "margin": (scored[0][0] - scored[1][0]) if len(scored) > 1 else None}
        if not hits:
            return None, read   # nothing relevant beats injecting noise the agent will trust
        block = Composite().text("Relevant things you recall from earlier work:\n")
        for n, (score, it) in enumerate(hits):
            if n:
                block.text("\n")
            block.text(f"- ({it.meta.get('kind', 'note')}, round {it.meta.get('round', '?')}, "
                       f"relevance {score:.2f}) ")
            block.ref(it.write_id, it.text)
        return block, read


# ================================================================ blackboard
class Blackboard:
    """Shared state across the agents of one run."""

    def __init__(self, clock, data: dict | None = None, log: list | None = None):
        self.clock = clock
        self.data: dict[str, str] = dict(data or {})
        self.log: list[dict] = list(log or [])
        # artifact id of each key's current value: "bb<seq>" of its latest write
        self.ids: dict[str, str] = {e["key"]: f"bb{e['seq']}" for e in self.log}

    def write(self, key: str, value: str, author: str, round_idx: int) -> dict:
        seq = self.clock()
        self.data[key] = value
        self.ids[key] = f"bb{seq}"
        self.log.append({"seq": seq, "round": round_idx,
                         "key": key, "author": author, "value": value})
        return {"status": "written", "key": key}

    def read(self, key: str | None = None) -> dict:
        if key is None:
            return {"keys": sorted(self.data)}
        if key not in self.data:
            return {"error": f"no entry '{key}'", "keys": sorted(self.data)}
        return {"key": key, "value": self.data[key]}
