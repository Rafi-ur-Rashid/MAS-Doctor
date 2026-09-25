"""Event log: every artifact that enters or leaves a model context, with an id.

The log is the substrate the monitor reads (EXPERIMENT_PLAN.md §2.1, §4.1). Two
kinds of record go into runs/<id>/events.jsonl, one JSON object per line:

  artifact  a piece of content, registered once: the task, a system prompt, a
            model output, a tool result, a memory item, a blackboard value, a
            workspace file, an agent's final response, or a chat message.
  event     something that happened: run_start, prompt_delivery, llm_call,
            tool_dispatch, tool_result, mem_read, mem_write, msg_send, msg_recv.

Chat messages are stored as *recipes*, not copies: a list of parts, each either
runtime-authored literal text ({"text": ...}) or a reference to another artifact
({"ref": id}, optionally with "field"). So the log says exactly which pieces went
into every prompt, and which text the runtime itself wrote. Every llm_call lists
the message ids that formed its context; verify_run() rebuilds those messages
from the recipes and checks them against a hash of what was actually sent and,
for real runs, against the request stored in the replay cache.

Determinism: ids come from positions the scheduler fixes (call index, message
index, tool-call position, run-clock values ticked only in sequential phases),
and events from concurrent phases are buffered per agent and flushed in agent
order. The same run therefore yields a byte-identical log.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from cache import canonical, chat_key

SCHEMA_VERSION = 1


def sha256(obj) -> str:
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


# ------------------------------------------------------------------ recipes
class Composite:
    """Builds a string and its recipe together, so the two cannot drift apart.
    Each piece keeps the text it rendered to when it was added."""

    def __init__(self):
        self._pieces: list[tuple[dict, str]] = []

    def text(self, s: str) -> "Composite":
        if s:
            if self._pieces and "text" in self._pieces[-1][0]:
                merged = self._pieces[-1][1] + s
                self._pieces[-1] = ({"text": merged}, merged)
            else:
                self._pieces.append(({"text": s}, s))
        return self

    def ref(self, aid: str, content: str, field: str | None = None) -> "Composite":
        part = {"ref": aid} if field is None else {"ref": aid, "field": field}
        self._pieces.append((part, content))
        return self

    def extend(self, other: "Composite") -> "Composite":
        for part, rendered in other._pieces:
            if "text" in part:
                self.text(rendered)
            else:
                self._pieces.append((dict(part), rendered))
        return self

    @property
    def parts(self) -> list[dict]:
        return [dict(p) for p, _ in self._pieces]

    @property
    def content(self) -> str:
        return "".join(t for _, t in self._pieces)


def render(parts: list[dict] | None, artifacts: dict) -> str | None:
    """Rebuild a string from a recipe. parts=None stands for content None."""
    if parts is None:
        return None
    out = []
    for p in parts:
        if "text" in p:
            out.append(p["text"])
        else:
            out.append(artifacts[p["ref"]][p.get("field", "content")])
    return "".join(out)


# --------------------------------------------------------------------- log
class EventLog:
    """One per run. emit() appends immediately; code running in a concurrent
    phase emits into its own buffer and the scheduler flushes buffers in a fixed
    order (flush())."""

    def __init__(self):
        self.records: list[dict] = []
        self.artifacts: dict[str, dict] = {}

    def emit(self, record: dict, buffer: list | None = None) -> None:
        if buffer is not None:
            buffer.append(record)
        else:
            self._commit(record)

    def flush(self, buffer: list) -> None:
        for r in buffer:
            self._commit(r)
        buffer.clear()

    def _commit(self, record: dict) -> None:
        if record["type"] == "artifact":
            aid = record["id"]
            if aid in self.artifacts:
                raise ValueError(f"artifact id {aid!r} registered twice")
            self.artifacts[aid] = record
        self.records.append({"seq": len(self.records), **record})

    def write(self, path: Path) -> None:
        with open(path, "w") as f:
            for r in self.records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")


def artifact(aid: str, kind: str, producer: str, content, **meta) -> dict:
    return {"type": "artifact", "id": aid, "kind": kind, "producer": producer,
            "content": content, **meta}


def event(ev: str, **fields) -> dict:
    return {"type": "event", "ev": ev, **fields}


def message_artifact(aid: str, producer: str, role: str, parts: list | None,
                     fields: dict) -> dict:
    """fields: the message's keys other than role/content (tool_calls,
    tool_call_id, name), kept verbatim so the message rebuilds exactly."""
    return {"type": "artifact", "id": aid, "kind": "message", "producer": producer,
            "role": role, "parts": parts, "fields": fields}


def rebuild_message(aid: str, artifacts: dict) -> dict:
    a = artifacts[aid]
    if a["kind"] != "message":
        raise ValueError(f"{aid!r} is a {a['kind']}, not a message")
    return {"role": a["role"], "content": render(a["parts"], artifacts), **a["fields"]}


# ------------------------------------------------------------------ verify
def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def verify_records(records: list[dict], cache=None) -> dict:
    """Checks a run's log is complete and exact:
      - every artifact id is registered once, and every reference resolves;
      - every llm_call's messages, rebuilt from recipes, hash to what was sent;
      - with a cache: the rebuilt request has the recorded cache key, and that
        key is present in the store (the stored request is what was sent).
    Returns counts; raises AssertionError on the first failure."""
    from tools import REGISTRY   # tool specs are rebuilt from the registry

    artifacts: dict[str, dict] = {}
    for r in records:
        if r["type"] == "artifact":
            assert r["id"] not in artifacts, f"artifact {r['id']} registered twice"
            artifacts[r["id"]] = r
    for r in records:
        for p in (r.get("parts") or []):
            if "ref" in p:
                assert p["ref"] in artifacts, f"{r['id']}: unresolved ref {p['ref']}"
                assert p.get("field", "content") in artifacts[p["ref"]], \
                    f"{r['id']}: {p['ref']} has no field {p.get('field')}"

    start = next(r for r in records if r.get("ev") == "run_start")
    calls = [r for r in records if r.get("ev") == "llm_call"]
    keys_checked = 0
    for c in calls:
        msgs = [rebuild_message(m, artifacts) for m in c["context"]]
        assert sha256(msgs) == c["messages_sha256"], \
            f"llm_call {c['call_idx']}: rebuilt messages differ from what was sent"
        if c.get("cache_key"):
            request = dict(model=start["model"], messages=msgs, **start["sampling"])
            if c["tools"]:
                request["tools"] = REGISTRY.specs(c["tools"])
                request["tool_choice"] = "auto"
            assert chat_key(request, c["replica"]) == c["cache_key"], \
                f"llm_call {c['call_idx']}: rebuilt request has a different cache key"
            if cache is not None:
                assert cache.get(c["cache_key"]) is not None, \
                    f"llm_call {c['call_idx']}: cache key not in store"
            keys_checked += 1
        assert c["output"] in artifacts, f"llm_call {c['call_idx']}: output not registered"
    return {"records": len(records), "artifacts": len(artifacts),
            "llm_calls": len(calls), "cache_keys_checked": keys_checked}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Verify a run's event log.")
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--no-cache", action="store_true", help="skip the cache-store lookup")
    args = ap.parse_args()
    cache = None
    if not args.no_cache:
        from cache import ResponseCache
        from config import CFG
        cache = ResponseCache(CFG.cache_path)
    print(json.dumps(verify_records(load(args.run_dir / "events.jsonl"), cache), indent=2))
    print("OK")
