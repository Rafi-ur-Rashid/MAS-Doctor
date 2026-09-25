"""RunContext: everything mutable that belongs to one run.

A run never touches global state. It gets its own blackboard, workspace files,
per-agent episodic stores, usage counter and logical clock, all built from a
*named state snapshot* ("empty" unless the caller names another). A run's end
state is saved only if the caller asks for it, under a new name, and existing
snapshots are never overwritten. So nothing carries over between runs
implicitly, and any run can be repeated from the exact state it started from.

It also carries the run's cache settings (C03): the cache mode, and the replica
index with the call it takes effect from. Calls numbered below replica_from_call
reuse the base recording (replica 0); calls from there on are sampled afresh
under the given replica. CacheStats records which calls were served from the
store and where the run first diverged from it.

The run's event log (C04, events.py) lives here too. State loaded from a snapshot
is registered in it first, as artifacts with producer "snapshot", so every later
reference to a loaded memory item, blackboard value or file resolves.
"""
from __future__ import annotations

import itertools
import json
import re
from dataclasses import asdict
from pathlib import Path

from cache import CacheStats
from config import CFG
from events import EventLog, artifact
from llm import Usage
from memory import Blackboard, MemoryItem, VectorStore

EMPTY = "empty"
SNAPSHOT_FORMAT = 2   # 2: memory items carry write_id; files carry artifact ids
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class SnapshotError(ValueError):
    pass


def snapshot_path(name: str, state_dir: Path | None = None) -> Path:
    if not _NAME_RE.match(name):
        raise SnapshotError(f"invalid snapshot name {name!r}")
    return (state_dir or CFG.state_dir) / "snapshots" / f"{name}.json"


class RunContext:
    def __init__(self, client, agent_names: list[str], state_from: str = EMPTY,
                 state_dir: Path | None = None, cache_mode: str | None = None,
                 replica: int = 0, replica_from_call: int = 0):
        """client: a shared LLMClient (or any object with the same for_run).
        agent_names: the roster; episodic stores are keyed by agent name."""
        self.state_dir = state_dir or CFG.state_dir
        self.state_from = state_from
        self.usage = Usage()
        self.cache_stats = CacheStats()
        self.llm = client.for_run(self.usage, stats=self.cache_stats,
                                  mode=cache_mode or CFG.cache_mode,
                                  replica=replica, replica_from_call=replica_from_call)

        snap = self._load(state_from)
        unknown = sorted(set(snap["agents"]) - set(agent_names))
        if unknown:
            raise SnapshotError(f"snapshot {state_from!r} has memory for agents not in this "
                                f"roster: {unknown}")
        if any(snap["agents"].values()) and snap["embed_space"] != self.llm.embed_space:
            raise SnapshotError(f"snapshot {state_from!r} was embedded in "
                                f"{snap['embed_space']!r}, this run uses {self.llm.embed_space!r}")
        # continue the logical clock after whatever the snapshot already holds
        self._seq = itertools.count(snap["next_seq"])
        # a run started from a snapshot of generation g is generation g + 1; ids of
        # content that persists across runs (files) include it, so a chain of runs
        # through snapshots never reuses an id
        self.generation = snap["generation"] + 1

        self.blackboard = Blackboard(self.clock, snap["blackboard"]["data"],
                                     snap["blackboard"]["log"])
        self.files: dict[str, str] = dict(snap["files"])
        self.file_ids: dict[str, str] = dict(snap["file_ids"])
        self.stores: dict[str, VectorStore] = {
            name: VectorStore(self.llm, self.clock,
                              [MemoryItem(**d) for d in snap["agents"].get(name, [])])
            for name in agent_names
        }

        self.events = EventLog()
        for name, store in self.stores.items():
            for it in store.items:
                self.events.emit(artifact(it.write_id, "memory_item", "snapshot", it.text,
                                          owner=name, meta=it.meta))
        for key, value in self.blackboard.data.items():
            self.events.emit(artifact(self.blackboard.ids[key], "blackboard_value", "snapshot",
                                      value, key=key))
        for path, content in self.files.items():
            self.events.emit(artifact(self.file_ids[path], "file", "snapshot", content, path=path))

    # ------------------------------------------------------------ clock
    def clock(self) -> int:
        """Logical time. Strictly increasing within a run; never wall-clock."""
        return next(self._seq)

    # -------------------------------------------------------- snapshots
    def _load(self, name: str) -> dict:
        if name == EMPTY:
            return {"format": SNAPSHOT_FORMAT, "embed_space": None, "next_seq": 1, "generation": 0,
                    "agents": {}, "blackboard": {"data": {}, "log": []}, "files": {},
                    "file_ids": {}}
        path = snapshot_path(name, self.state_dir)
        if not path.exists():
            raise SnapshotError(f"no state snapshot named {name!r} ({path})")
        snap = json.loads(path.read_text())
        if snap.get("format") != SNAPSHOT_FORMAT:
            raise SnapshotError(f"snapshot {name!r} has format {snap.get('format')}, "
                                f"expected {SNAPSHOT_FORMAT}")
        return snap

    def state(self) -> dict:
        """The run's current mutable state, in snapshot form."""
        last = next(self._seq)
        self._seq = itertools.count(last)      # peek without consuming a tick
        return {
            "format": SNAPSHOT_FORMAT,
            "embed_space": self.llm.embed_space,
            "next_seq": last,
            "generation": self.generation,
            "agents": {n: [asdict(it) for it in s.items] for n, s in self.stores.items()},
            "blackboard": {"data": self.blackboard.data, "log": self.blackboard.log},
            "files": self.files,
            "file_ids": self.file_ids,
        }

    def save_snapshot(self, name: str) -> Path:
        if name == EMPTY:
            raise SnapshotError(f"{EMPTY!r} is reserved")
        path = snapshot_path(name, self.state_dir)
        if path.exists():
            raise SnapshotError(f"snapshot {name!r} already exists; snapshots are immutable")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.state()))
        return path
